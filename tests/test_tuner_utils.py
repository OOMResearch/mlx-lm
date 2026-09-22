# Copyright © 2024 Apple Inc.

import json
import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.tuner.lora import LoRALinear
from mlx_lm.tuner.utils import (
    fuse_adapters,
    load_adapters,
    print_trainable_parameters,
)


class TestTunerUtils(unittest.TestCase):
    def setUp(self):
        self.capturedOutput = StringIO()
        sys.stdout = self.capturedOutput

    def tearDown(self):
        sys.stdout = sys.__stdout__

    def test_quantized_print_trainable_parameters(self):
        model = MagicMock()
        quantized_linear = MagicMock(spec=nn.QuantizedLinear)
        quantized_linear.weight = MagicMock(size=1e6)
        quantized_linear.bits = 8
        lora_linear = MagicMock(spec=LoRALinear)
        lora_linear.weight = MagicMock(size=2e6)
        lora_linear.parameters.return_value = [lora_linear.weight]

        linear = MagicMock(spec=nn.Linear)
        linear.weight = MagicMock(size=3e6)
        linear.parameters.return_value = [linear.weight]

        model.leaf_modules.return_value = {
            "quantized_linear": quantized_linear,
            "lora_linear": lora_linear,
            "linear": linear,
        }

        model.trainable_parameters.return_value = {
            "layer1.weight": MagicMock(size=1e6),
            "layer3.weight": MagicMock(size=2e6),
        }
        expected_output_8bits = "Trainable parameters: 33.333% (3.000M/9.000M)\n"
        print_trainable_parameters(model)
        self.assertEqual(self.capturedOutput.getvalue(), expected_output_8bits)
        self.capturedOutput.truncate(0)
        self.capturedOutput.seek(0)

        quantized_linear.weight = MagicMock(size=1e6)
        quantized_linear.bits = 4
        expected_output_4bits = "Trainable parameters: 23.077% (3.000M/13.000M)\n"
        print_trainable_parameters(model)
        self.assertEqual(self.capturedOutput.getvalue(), expected_output_4bits)
        self.capturedOutput.truncate(0)
        self.capturedOutput.seek(0)

    def test_print_trainable_parameters(self):
        model = MagicMock()
        linear1 = MagicMock(spec=nn.Linear)
        linear1.weight = MagicMock(size=1e6)
        linear1.parameters.return_value = [linear1.weight]
        linear2 = MagicMock(spec=nn.Linear)
        linear2.weight = MagicMock(size=2e6)
        linear2.parameters.return_value = [linear2.weight]
        lora_linear = MagicMock(spec=LoRALinear)
        lora_linear.weight = MagicMock(size=3e6)
        lora_linear.parameters.return_value = [lora_linear.weight]
        model.leaf_modules.return_value = {
            "linear1": linear1,
            "linear2": linear2,
            "lora_linear": lora_linear,
        }

        model.trainable_parameters.return_value = {
            "layer1.weight": MagicMock(size=1e6),
            "layer3.weight": MagicMock(size=2e6),
        }
        expected_output = "Trainable parameters: 50.000% (3.000M/6.000M)\n"
        print_trainable_parameters(model)
        self.assertEqual(self.capturedOutput.getvalue(), expected_output)


class TinyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 3, bias=False)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [TinyLayer()]


def _make_adapter(weights):
    path = Path(tempfile.mkdtemp())
    config = {
        "fine_tune_type": "lora",
        "num_layers": 1,
        "lora_parameters": {
            "keys": ["proj"],
            "rank": 2,
            "scale": 1.0,
            "dropout": 0.0,
        },
    }
    (path / "adapter_config.json").write_text(json.dumps(config))
    mx.save_safetensors(str(path / "adapters.safetensors"), weights)
    return path


class TestLoadAdapters(unittest.TestCase):
    def test_unknown_tensor_name_raises(self):
        path = _make_adapter(
            {
                "layers.0.proj.lora_a": mx.zeros((4, 2)),
                "layers.0.proj.lora_b": mx.zeros((2, 3)),
                "layers.0.proj.lora_typo": mx.zeros((1,)),
            }
        )
        with self.assertRaises(ValueError) as cm:
            load_adapters(TinyModel(), path)
        self.assertIn("lora_typo", str(cm.exception))

    def test_wrong_tensor_shape_raises(self):
        path = _make_adapter(
            {
                "layers.0.proj.lora_a": mx.zeros((5, 2)),
                "layers.0.proj.lora_b": mx.zeros((2, 3)),
            }
        )
        with self.assertRaises(ValueError) as cm:
            load_adapters(TinyModel(), path)
        self.assertIn("shape", str(cm.exception))

    def test_valid_adapter_loads(self):
        path = _make_adapter(
            {
                "layers.0.proj.lora_a": mx.zeros((4, 2)),
                "layers.0.proj.lora_b": mx.zeros((2, 3)),
            }
        )
        model = load_adapters(TinyModel(), path)
        self.assertIsInstance(model, nn.Module)

    def test_base_weight_tensor_rejected(self):
        path = _make_adapter(
            {
                "layers.0.proj.lora_a": mx.zeros((4, 2)),
                "layers.0.proj.lora_b": mx.zeros((2, 3)),
                "layers.0.proj.linear.weight": mx.full((3, 4), 42.0),
            }
        )
        with self.assertRaises(ValueError) as cm:
            load_adapters(TinyModel(), path)
        self.assertIn("linear.weight", str(cm.exception))

    def test_valid_dora_adapter_loads(self):
        path = Path(tempfile.mkdtemp())
        config = {
            "fine_tune_type": "dora",
            "num_layers": 1,
            "lora_parameters": {
                "keys": ["proj"],
                "rank": 2,
                "scale": 1.0,
                "dropout": 0.0,
            },
        }
        (path / "adapter_config.json").write_text(json.dumps(config))
        mx.save_safetensors(
            str(path / "adapters.safetensors"),
            {
                "layers.0.proj.lora_a": mx.zeros((4, 2)),
                "layers.0.proj.lora_b": mx.zeros((2, 3)),
                "layers.0.proj.m": mx.zeros((3,)),
            },
        )
        model = load_adapters(TinyModel(), path)
        self.assertIsInstance(model, nn.Module)


class TestFuseAdapters(unittest.TestCase):
    # Metal float32 matmuls are not bit-exact against the unfused path; the
    # CPU stream makes these equivalence checks exact rather than tolerant.
    def setUp(self):
        self._device = mx.default_device()
        mx.set_default_device(mx.cpu)

    def tearDown(self):
        mx.set_default_device(self._device)

    def _adapter(self, seed):
        mx.random.seed(seed)
        return _make_adapter(
            {
                "layers.0.proj.lora_a": mx.random.normal((4, 2)),
                "layers.0.proj.lora_b": mx.random.normal((2, 3)),
            }
        )

    def test_fused_model_matches_unfused_output(self):
        base = TinyModel()
        x = mx.random.normal((2, 4))
        path = self._adapter(0)

        unfused = load_adapters(base, path)
        expected = unfused.layers[0].proj(x)

        fused = fuse_adapters(unfused)
        self.assertIsInstance(fused.layers[0].proj, nn.Linear)
        self.assertNotIsInstance(fused.layers[0].proj, LoRALinear)
        self.assertTrue(mx.allclose(fused.layers[0].proj(x), expected, atol=1e-5))

    def test_fusing_several_adapters_sums_their_deltas(self):
        model = TinyModel()
        weight = model.layers[0].proj.weight
        paths = [self._adapter(1), self._adapter(2)]

        expected = weight
        for path in paths:
            adapter = mx.load(str(path / "adapters.safetensors"))
            a = adapter["layers.0.proj.lora_a"]
            b = adapter["layers.0.proj.lora_b"]
            expected = expected + (b.T @ a.T)

        for path in paths:
            model = fuse_adapters(load_adapters(model, path))
        self.assertTrue(mx.allclose(model.layers[0].proj.weight, expected, atol=1e-5))

    def test_fuse_keeps_quantized_base_quantized(self):
        model = TinyModel()
        model.layers[0].proj = nn.Linear(64, 32, bias=False)
        nn.quantize(model, group_size=32, bits=4)
        path = _make_adapter(
            {
                "layers.0.proj.lora_a": mx.zeros((64, 2)),
                "layers.0.proj.lora_b": mx.zeros((2, 32)),
            }
        )
        model = fuse_adapters(load_adapters(model, path))
        self.assertIsInstance(model.layers[0].proj, nn.QuantizedLinear)

        model = TinyModel()
        model.layers[0].proj = nn.Linear(64, 32, bias=False)
        nn.quantize(model, group_size=32, bits=4)
        model = fuse_adapters(load_adapters(model, path), dequantize=True)
        self.assertNotIsInstance(model.layers[0].proj, nn.QuantizedLinear)


if __name__ == "__main__":
    unittest.main()
