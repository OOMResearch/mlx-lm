# Copyright © 2025 Apple Inc.

import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optimizers
import numpy as np
from mlx.utils import tree_flatten

from mlx_lm.models import llama
from mlx_lm.quant.dwq import _topk_indices, compute_dwq_targets, dwq_quantize
from mlx_lm.tuner.trainer import iterate_batches


class TestDWQ(unittest.TestCase):
    def setUp(self):
        mx.random.seed(0)
        args = llama.ModelArgs(
            model_type="llama",
            hidden_size=64,
            num_hidden_layers=2,
            intermediate_size=128,
            num_attention_heads=2,
            rms_norm_eps=1e-5,
            vocab_size=128,
            tie_word_embeddings=False,
        )
        self.teacher = llama.Model(args)
        mx.eval(self.teacher.parameters())
        self.student = copy.deepcopy(self.teacher)
        nn.quantize(self.student, group_size=32, bits=4)

        # Mixed lengths so batches are padded to different shapes.
        rng = np.random.default_rng(0)
        lengths = [20, 24, 28, 30, 40, 44, 50, 60, 70, 76, 80, 90]
        self.data = [(rng.integers(0, 128, size=n).tolist(), 0) for n in lengths]

    def run_dwq(self, target_fn):
        student = copy.deepcopy(self.student)
        dwq_quantize(
            student,
            target_fn,
            optimizers.Adam(learning_rate=1e-4, bias_correction=True),
            self.data,
            self.data[:4],
            batch_size=4,
            max_seq_length=128,
            seed=0,
            dtype=mx.float32,
        )
        return student

    def test_compiled_step_matches_uncompiled(self):
        def target_fn(batch, idx, split):
            return self.teacher(batch)

        compiled = self.run_dwq(target_fn)
        with mock.patch("mlx_lm.quant.dwq.mx.compile", lambda fn, **kwargs: fn):
            reference = self.run_dwq(target_fn)

        before = dict(tree_flatten(self.student.parameters()))
        expected = dict(tree_flatten(reference.trainable_parameters()))
        actual = dict(tree_flatten(compiled.trainable_parameters()))
        self.assertEqual(actual.keys(), expected.keys())
        self.assertGreater(len(actual), 0)
        for k, v in expected.items():
            # Training moved the parameters, and by the same amount either way.
            self.assertFalse(mx.array_equal(v, before[k]).item())
            self.assertTrue(mx.allclose(actual[k], v, rtol=1e-3, atol=1e-4).item())

        # The compiled step must leave the model usable.
        logits = compiled(mx.array([self.data[0][0]]))
        self.assertTrue(mx.isfinite(logits).all().item())

    def test_compiled_step_with_top_k_targets(self):
        def target_fn(batch, idx, split):
            logits = self.teacher(batch)
            ids = mx.argpartition(logits, kth=-16, axis=-1)[..., -16:]
            return mx.take_along_axis(logits, ids, axis=-1), ids

        student = self.run_dwq(target_fn)
        before = dict(tree_flatten(self.student.parameters()))
        for k, v in tree_flatten(student.trainable_parameters()):
            self.assertTrue(mx.isfinite(v).all().item())
            self.assertFalse(mx.array_equal(v, before[k]).item())


class TestDWQTargets(unittest.TestCase):
    def assert_topk(self, x, idx, k):
        self.assertEqual(idx.shape, (*x.shape[:-1], k))
        self.assertLess(idx.max().item(), x.shape[-1])
        ordered = mx.sort(idx.astype(mx.int64), axis=-1)
        self.assertTrue((ordered[..., 1:] != ordered[..., :-1]).all().item())
        expected = mx.sort(x, axis=-1)[..., -k:]
        actual = mx.sort(mx.take_along_axis(x, idx, axis=-1), axis=-1)
        self.assertTrue(mx.array_equal(actual, expected).item())

    def test_topk_indices(self):
        mx.random.seed(0)
        cases = [
            ((2, 3, 4096), 64),  # blocked, no padding
            ((2, 3, 4099), 64),  # blocked, padded
            ((3, 1000), 8),
            ((2, 2, 200), 64),  # too small to block
            ((1, 2, 64), 64),  # k equal to the axis size
        ]
        for shape, k in cases:
            for dtype in [mx.float32, mx.bfloat16]:
                x = mx.random.normal(shape).astype(dtype)
                self.assert_topk(x, _topk_indices(x, k), k)
                # Rounded values tie heavily, including across the k-th value.
                x = mx.round(mx.random.normal(shape) * 2).astype(dtype)
                self.assert_topk(x, _topk_indices(x, k), k)

    def test_compute_dwq_targets(self):
        mx.random.seed(0)
        args = llama.ModelArgs(
            model_type="llama",
            hidden_size=64,
            num_hidden_layers=1,
            intermediate_size=128,
            num_attention_heads=2,
            rms_norm_eps=1e-5,
            vocab_size=20_000,
            tie_word_embeddings=False,
        )
        model = llama.Model(args)
        mx.eval(model.parameters())
        rng = np.random.default_rng(0)
        data = [(rng.integers(0, 20_000, size=n).tolist(), 0) for n in [12, 16, 20, 24]]

        with tempfile.TemporaryDirectory() as save_dir:
            compute_dwq_targets(model, Path(save_dir), data, data, 2, 64, seed=0)
            for split in ["train", "valid"]:
                files = sorted((Path(save_dir) / split).glob("*.safetensors"))
                self.assertEqual(len(files), 2)
                batches = iterate_batches(data, 2, 64, seed=0)
                for file, (batch, _) in zip(files, batches):
                    targets = mx.load(str(file))
                    logits = model(batch[:, :-1])
                    self.assert_topk(logits, targets["indices"], 1024)
                    self.assertTrue(
                        mx.array_equal(
                            targets["logits"],
                            mx.take_along_axis(logits, targets["indices"], axis=-1),
                        ).item()
                    )


if __name__ == "__main__":
    unittest.main()
