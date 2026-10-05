# Copyright © 2025 Apple Inc.

import copy
import unittest

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_lm.models import llama
from mlx_lm.tuner.quantized import (
    _DenseBackwardQuantizedEmbedding,
    _DenseBackwardQuantizedLinear,
    dense_quantized_backward,
)
from mlx_lm.tuner.trainer import default_loss
from mlx_lm.tuner.utils import linear_to_lora_layers


def make_model(tie_word_embeddings):
    mx.random.seed(0)
    args = llama.ModelArgs(
        model_type="llama",
        hidden_size=64,
        num_hidden_layers=2,
        intermediate_size=128,
        num_attention_heads=2,
        rms_norm_eps=1e-5,
        vocab_size=256,
        tie_word_embeddings=tie_word_embeddings,
    )
    model = llama.Model(args)
    nn.quantize(model, group_size=32, bits=4)
    model.freeze()
    linear_to_lora_layers(model, 2, {"rank": 4, "scale": 10.0, "dropout": 0.0})
    # Give the adapters a non-zero output so every gradient is non-trivial.
    for name, module in model.named_modules():
        if hasattr(module, "lora_b"):
            module.lora_b = mx.random.normal(module.lora_b.shape) * 0.05
    mx.eval(model.parameters())
    model.train()
    return model


def types_of(model):
    return {type(m) for _, m in model.named_modules()}


class TestDenseQuantizedBackward(unittest.TestCase):
    def setUp(self):
        self.batch = mx.random.randint(0, 256, (2, 17))
        self.lengths = mx.array([[0, 16], [0, 12]])

    def loss_and_grads(self, model):
        (loss, _), grads = nn.value_and_grad(model, default_loss)(
            model, self.batch, self.lengths
        )
        return loss, dict(tree_flatten(grads))

    def check_matches_reference(self, tie_word_embeddings):
        reference = make_model(tie_word_embeddings)
        model = copy.deepcopy(reference)
        dense_quantized_backward(model)

        self.assertIn(_DenseBackwardQuantizedLinear, types_of(model))
        self.assertNotIn(nn.QuantizedLinear, types_of(model))
        self.assertEqual(
            tree_flatten(model.parameters())[0][0],
            tree_flatten(reference.parameters())[0][0],
        )

        expected_loss, expected = self.loss_and_grads(reference)
        loss, grads = self.loss_and_grads(model)
        self.assertTrue(mx.array_equal(loss, expected_loss).item())
        self.assertEqual(grads.keys(), expected.keys())
        for k, v in expected.items():
            scale = mx.abs(v).max().item()
            self.assertGreater(scale, 0)
            # The dense matmul and the quantized kernel round differently, so
            # compare to a tolerance a wrong gradient could not meet.
            self.assertLess(mx.abs(grads[k] - v).max().item(), 1e-2 * scale, k)

    def test_gradients_match_untied(self):
        self.check_matches_reference(tie_word_embeddings=False)

    def test_gradients_match_tied(self):
        model = make_model(tie_word_embeddings=True)
        dense_quantized_backward(model)
        self.assertIn(_DenseBackwardQuantizedEmbedding, types_of(model))
        self.check_matches_reference(tie_word_embeddings=True)

    def test_size_limit_skips_large_layers(self):
        model = make_model(tie_word_embeddings=True)
        # q_proj is 64x64 float32 (16 KiB dense); the embedding is 256x64 (64 KiB).
        dense_quantized_backward(model, max_dense_bytes=32 * 1024)
        self.assertIn(_DenseBackwardQuantizedLinear, types_of(model))
        self.assertIn(nn.QuantizedEmbedding, types_of(model))
        self.assertNotIn(_DenseBackwardQuantizedEmbedding, types_of(model))

    def test_is_idempotent(self):
        model = make_model(tie_word_embeddings=True)
        dense_quantized_backward(model)
        before = types_of(model)
        dense_quantized_backward(model)
        self.assertEqual(types_of(model), before)

    def test_trainable_quantization_parameters_keep_their_gradients(self):
        reference = make_model(tie_word_embeddings=False)
        for _, m in reference.named_modules():
            if isinstance(m, nn.QuantizedLinear):
                m.unfreeze(keys=["scales", "biases"], recurse=False)
        model = copy.deepcopy(reference)
        dense_quantized_backward(model)

        _, expected = self.loss_and_grads(reference)
        _, grads = self.loss_and_grads(model)
        scale_keys = [k for k in expected if k.endswith("scales")]
        self.assertGreater(len(scale_keys), 0)
        for k in scale_keys:
            self.assertGreater(mx.abs(expected[k]).max().item(), 0)
            self.assertTrue(mx.array_equal(grads[k], expected[k]).item(), k)

    def test_eval_mode_uses_the_plain_forward(self):
        model = make_model(tie_word_embeddings=True)
        reference = copy.deepcopy(model)
        dense_quantized_backward(model)
        model.eval()
        reference.eval()
        inputs = self.batch[:, :-1]
        self.assertTrue(mx.array_equal(model(inputs), reference(inputs)).item())


if __name__ == "__main__":
    unittest.main()
