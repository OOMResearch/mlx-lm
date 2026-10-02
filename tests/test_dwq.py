# Copyright © 2025 Apple Inc.

import copy
import unittest
from unittest import mock

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optimizers
import numpy as np
from mlx.utils import tree_flatten

from mlx_lm.models import llama
from mlx_lm.quant.dwq import dwq_quantize


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


if __name__ == "__main__":
    unittest.main()
