"""Single-row affine W4/BF16 QMV with eight threadgroup-local K splits.

The compact P2 activation-loading and affine-dot expressions are specialized
from MLX 0.32.4 quantized.h (Copyright © 2023-2024 Apple Inc., MIT). The split-K schedule and reduction are local.
Parameters retain native unsigned W4/group-32 packing. Products, partial
sums, and their merge use FP32; the output is rounded to BF16 once.

This inference operator returns a lazy graph and has no autodiff rule.
Splitting K changes FP32 accumulation order relative to native MLX QMV.
"""

import math
from functools import lru_cache
from string import Template

import mlx.core as mx

from ._types import StreamOrDevice
from ._validation import gpu_stream
from ._w4a16_sources import _BODY, _HEADER


@lru_cache(maxsize=64)
def _kernel(n, k):
    # Cache only the shape-specialized callable, never arrays or streams.
    return mx.fast.metal_kernel(
        name=f"spark_w4a16_split8_r4_n{n}_k{k}",
        input_names=["weight", "scales", "biases", "x"],
        output_names=["out"],
        header=_HEADER,
        source=Template(_BODY).substitute(K=k),
        ensure_row_contiguous=True,
        compile_options={"math_mode": "safe"},
    )


def w4a16_qmv(
    x: mx.array,
    weight: mx.array,
    scales: mx.array,
    biases: mx.array,
    *,
    stream: StreamOrDevice = None,
) -> mx.array:
    """Return one-row affine W4/G32 QMV as a lazy BF16 MLX array.

    ``x`` must be BF16 with a positive last dimension K divisible by 512.
    Any leading dimensions must have product one, including no leading
    dimensions. ``weight`` is native unsigned UINT32-packed W4 ``[N,K/8]``;
    BF16 ``scales`` and ``biases`` are ``[N,K/32]``. N must be a positive
    multiple of four. The output preserves x's leading dimensions and ends
    in N. The affine biases are quantization offsets, not a linear bias.

    Every call consumes the supplied arrays and selected GPU stream/device;
    inputs are never modified. Noncontiguous inputs are copied lazily by
    MLX's Metal-kernel wrapper. There is no internal eval, synchronization,
    activation quantization, parameter conversion, or automatic fallback.
    Unsupported dtypes, shapes, or CPU streams raise an error.

    The eight-split policy has confirmed isolated GPU gains for weight
    shapes ``[6144,2560]`` and ``[2560,10240]`` on the recorded M5 Pro setup.
    Other supported shapes have no implied performance benefit. Splits
    change FP32 accumulation order, so outputs may differ from native MLX;
    inference quality and end-to-end speed require separate validation.
    """
    if any(not isinstance(value, mx.array) for value in (x, weight, scales, biases)):
        raise TypeError("x, weight, scales and biases must be MLX arrays")
    if (
        x.dtype != mx.bfloat16
        or scales.dtype != mx.bfloat16
        or biases.dtype != mx.bfloat16
    ):
        raise ValueError("W4A16 QMV requires BF16 x, scales and biases")
    if weight.dtype != mx.uint32:
        raise ValueError("W4A16 QMV requires UINT32-packed W4 weights")
    if x.ndim < 1 or math.prod(x.shape[:-1]) != 1:
        raise ValueError("W4A16 QMV requires exactly one activation row")
    k = x.shape[-1]
    if k <= 0 or k % 512:
        raise ValueError("W4A16 QMV requires positive K divisible by 512")
    if weight.ndim != 2:
        raise ValueError("W4A16 packed weight must have shape [N,K/8]")
    n = weight.shape[0]
    if n <= 0 or n % 4 or weight.shape[1] != k // 8:
        raise ValueError(
            "W4A16 QMV requires positive N divisible by 4 and weight [N,K/8]"
        )
    if scales.shape != (n, k // 32) or biases.shape != scales.shape:
        raise ValueError("W4A16 QMV requires scales and biases with shape [N,K/32]")
    stream = gpu_stream(stream, operator="W4A16 QMV")
    return _kernel(n, k)(
        inputs=[weight, scales, biases, x],
        output_shapes=[(*x.shape[:-1], n)],
        output_dtypes=[mx.bfloat16],
        grid=(32, (n // 4) * 8, 1),
        threadgroup=(32, 8, 1),
        stream=stream,
    )[0]
