# Copyright © 2025 Apple Inc.

import copy
import unittest

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map, tree_unflatten

from mlx_lm.models import llama
from mlx_lm.quant.dynamic_quant import estimate_sensitivities
from mlx_lm.tuner.losses import kl_div_loss


def reference_sensitivities(model, data, low, high, group_size, batch_size):
    """Uncompiled, step-by-step version of estimate_sensitivities."""

    def qdq(w, bits):
        w, s, b = mx.quantize(w, bits=bits, group_size=group_size)
        return mx.dequantize(w, scales=s, biases=b, bits=bits, group_size=group_size)

    layers = tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module)
    q_layers = copy.deepcopy({k: l for k, l in layers if hasattr(l, "to_quantized")})
    q_model = copy.deepcopy(model)
    for l in q_layers.values():
        l.weight = qdq(l.weight, low)
        l.freeze()
        l.unfreeze(keys=["weight"])
    q_model.freeze()
    q_model.update_modules(tree_unflatten(list(q_layers.items())))

    def loss_fn(batch, targets):
        return kl_div_loss(q_model(batch), targets).mean()

    grad_accum = tree_map(mx.zeros_like, q_model.trainable_parameters())
    n_batches = 0
    for s in range(0, len(data), batch_size):
        batch = data[s : s + batch_size]
        _, grads = nn.value_and_grad(q_model, loss_fn)(batch, model(batch))
        grad_accum = tree_map(lambda x, y: x + y, grad_accum, grads)
        mx.eval(grad_accum)
        n_batches += 1

    def sensitivity(gradient, low_q_weight, weight):
        delta = low_q_weight - qdq(weight, high)
        return (gradient / n_batches * delta).sum() / (weight.size / 1e6)

    sensitivities = tree_map(
        sensitivity, grad_accum, q_model.parameters(), model.parameters()
    )
    return {k[:-7]: s.item() for k, s in tree_flatten(sensitivities)}


class TestDynamicQuant(unittest.TestCase):
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
        self.model = llama.Model(args)
        mx.eval(self.model.parameters())

    def check(self, data, batch_size):
        probe = data[:1]
        before = self.model(probe)
        mx.eval(before)

        expected = reference_sensitivities(self.model, data, 4, 5, 64, batch_size)
        actual = dict(
            estimate_sensitivities(self.model, data, 4, 64, 5, 64, batch_size)
        )

        self.assertEqual(actual.keys(), expected.keys())
        scale = max(abs(v) for v in expected.values())
        self.assertGreater(scale, 0)
        for k, v in expected.items():
            self.assertAlmostEqual(actual[k], v, delta=1e-3 * scale)

        # The compiled step must leave the original model usable.
        self.assertTrue(mx.array_equal(self.model(probe), before))

    def test_estimate_sensitivities_matches_reference(self):
        self.check(mx.random.randint(0, 128, (8, 16)), batch_size=4)

    def test_estimate_sensitivities_partial_last_batch(self):
        self.check(mx.random.randint(0, 128, (6, 16)), batch_size=4)


if __name__ == "__main__":
    unittest.main()
