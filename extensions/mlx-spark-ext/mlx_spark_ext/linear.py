"""Inference layers backed by the extension's INT8 TensorOps primitive."""

import mlx.core as mx
from mlx import nn

from ._ext import w8a8_linear


class W8A8Linear(nn.Module):
    """Linear layer using existing symmetric per-channel INT8 weights.

    ``weight`` is an INT8 array of shape ``[out_features, in_features]``.
    ``scales`` is an FP32 array of shape ``[out_features]`` or
    ``[out_features, 1]``. No weights are re-quantized. Input activations are
    quantized per token, including single-token decode calls. Autodiff is
    unsupported; use this layer for inference.
    """

    def __init__(
        self, weight: mx.array, scales: mx.array, bias: mx.array | None = None
    ):
        super().__init__()
        if weight.ndim != 2 or weight.dtype != mx.int8:
            raise ValueError("weight must be an INT8 matrix [N, K]")
        n, k = weight.shape
        if n <= 0 or k <= 0 or k % 32 or k > 131040:
            raise ValueError(
                "N must be positive; K must be a multiple of 32 in [32, 131040]"
            )
        if scales.dtype != mx.float32 or scales.shape not in ((n,), (n, 1)):
            raise ValueError("scales must be FP32 with shape [N] or [N, 1]")
        if bias is not None and (
            bias.shape != (n,)
            or bias.dtype not in (mx.float16, mx.bfloat16, mx.float32)
        ):
            raise ValueError("bias must be a floating array of shape [N]")
        self.weight = weight
        self.scales = scales
        if bias is not None:
            self.bias = bias
        self.freeze()

    @property
    def input_dims(self) -> int:
        return self.weight.shape[1]

    @property
    def output_dims(self) -> int:
        return self.weight.shape[0]

    def __call__(self, x: mx.array) -> mx.array:
        return w8a8_linear(x, self.weight, self.scales, self.get("bias"))

    def _extra_repr(self) -> str:
        return f"input_dims={self.input_dims}, output_dims={self.output_dims}, bias={'bias' in self}"
