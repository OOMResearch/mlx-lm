# Copyright © 2025 Apple Inc.

import mlx.core as mx
import mlx.nn as nn

# Largest dequantized weight, in bytes, that the backward pass may materialize.
MAX_DENSE_BYTES = 256 << 20


def _frozen_quantized_matmul(layer, x):
    """
    ``x @ W.T`` for a frozen quantized weight, with a faster backward pass.

    Backpropagating through ``mx.quantized_matmul`` multiplies the cotangent
    by the quantized weight in its untransposed orientation, which is several
    times slower than the forward orientation. Dequantizing the weight and
    using a dense matmul computes the same input gradient much faster. The
    weight, scales and biases are treated as constants, so this must only be
    used when they are frozen.
    """
    weight, scales, biases = layer["weight"], layer["scales"], layer.get("biases")
    kwargs = dict(group_size=layer.group_size, bits=layer.bits, mode=layer.mode)

    @mx.custom_function
    def matmul(x):
        return mx.quantized_matmul(
            x, weight, scales=scales, biases=biases, transpose=True, **kwargs
        )

    @matmul.vjp
    def matmul_vjp(x, cotangent, output):
        return cotangent @ mx.dequantize(weight, scales=scales, biases=biases, **kwargs)

    return matmul(x)


def _use_dense_backward(layer):
    return layer.training and not layer.trainable_parameters()


class _DenseBackwardQuantizedLinear(nn.QuantizedLinear):
    def __call__(self, x):
        if not _use_dense_backward(self):
            return super().__call__(x)
        x = _frozen_quantized_matmul(self, x)
        if "bias" in self:
            x = x + self["bias"]
        return x


class _DenseBackwardQuantizedEmbedding(nn.QuantizedEmbedding):
    def as_linear(self, x):
        if not _use_dense_backward(self):
            return super().as_linear(x)
        return _frozen_quantized_matmul(self, x)


_DENSE_BACKWARD_CLASSES = {
    nn.QuantizedLinear: _DenseBackwardQuantizedLinear,
    nn.QuantizedEmbedding: _DenseBackwardQuantizedEmbedding,
}


def dense_quantized_backward(model: nn.Module, max_dense_bytes: int = MAX_DENSE_BYTES):
    """
    Speed up backpropagation through the frozen quantized layers of ``model``.

    Each plain ``nn.QuantizedLinear`` and ``nn.QuantizedEmbedding`` instance
    is switched to a subclass with the same parameters and forward pass, whose
    backward pass uses a dequantized copy of the weight while the layer is
    frozen and in training mode. Layers whose dequantized weight would be
    larger than ``max_dense_bytes`` (typically a large vocabulary projection)
    are left alone to keep peak memory unchanged.
    """
    for _, module in model.named_modules():
        cls = _DENSE_BACKWARD_CLASSES.get(type(module))
        if cls is None:
            continue
        scales = module["scales"]
        dense_bytes = scales.size * module.group_size * scales.dtype.size
        if dense_bytes <= max_dense_bytes:
            module.__class__ = cls
