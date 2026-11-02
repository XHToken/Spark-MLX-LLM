"""Multirow signed-INT8 projection with original FP32 per-channel scales.

This inference-only primitive retains BF16 activations and FP32 dot products,
then multiplies one FP32 scale per output channel. The output may be BF16 or
FP32. It does not perform A8 activation quantization or internal evaluation.
"""

import math
from functools import lru_cache
from string import Template

import mlx.core as mx

from ._ext import is_w8a8_available
from ._types import StreamOrDevice
from ._validation import gpu_stream
from ._w8a16_multirow_sources import HEADER, MATRIX, NAX, NAX_HEADER, WIDE

__all__ = ["w8a16_multirow"]


@lru_cache(maxsize=128)
def _kernel(n, k, m, variant, tile_rows, k_lanes, simdgroups):
    if variant == "nax":
        source = Template(NAX).substitute(K=k, N=n, M=m, SG=simdgroups)
    elif variant == "matrix":
        source = Template(MATRIX).substitute(K=k, N=n, M=m, SG=simdgroups)
    else:
        source = Template(WIDE).substitute(
            K=k, N=n, M=m, TM=tile_rows, KL=k_lanes, SG=simdgroups
        )
    return mx.fast.metal_kernel(
        name=f"spark_w8a16_multirow_{variant}_n{n}_k{k}_m{m}_tm{tile_rows}_kl{k_lanes}_sg{simdgroups}",
        input_names=["x", "weight", "scales"],
        output_names=["out"],
        header=NAX_HEADER if variant == "nax" else HEADER,
        source=source,
        ensure_row_contiguous=True,
        compile_options={"math_mode": "safe"},
    )


def w8a16_multirow(
    x: mx.array,
    weight: mx.array,
    scales: mx.array,
    *,
    output_dtype=None,
    variant="matrix",
    tile_rows=8,
    k_lanes=8,
    simdgroups=4,
    stream: StreamOrDevice = None,
) -> mx.array:
    """Return lazy ``x @ signed_weight.T * scales`` for 2..8 BF16 rows.

    ``weight`` must be signed INT8 [N,K], ``scales`` FP32 [N] or [N,1], and
    x must have positive K divisible by 32. Its leading dimensions may have
    product 2..8. Output preserves those leading dimensions and ends in N.
    ``output_dtype`` defaults to BF16 and also accepts FP32 for output heads.

    ``matrix`` uses explicit FP32 simdgroup 8x8 fragments. ``wide`` reuses
    each loaded INT8 quartet across tile_rows (2/4/8) activation rows. These
    layout options support opt-in measurements; neither implies a speedup.
    K-lanes may be 8/16/32 and simdgroups 2/4/8. Inputs are never modified;
    noncontiguous values are copied lazily by MLX's Metal wrapper. Unsupported
    inputs or CPU streams raise rather than selecting an implicit fallback.

    The reduction ordering differs from native affine MLX arithmetic. This
    operator has no autodiff rule and provides no generation-quality claim.
    ``nax`` is an explicit M5+ candidate: BF16 times signed INT8 MPP matmul,
    FP32 accumulation and original channel scale, with relaxed_precision=false.
    It requires N divisible by32 and K divisible by64. No A8 quantization occurs.
    """
    if any(not isinstance(v, mx.array) for v in (x, weight, scales)):
        raise TypeError("x, weight and scales must be MLX arrays")
    if x.dtype != mx.bfloat16 or weight.dtype != mx.int8 or scales.dtype != mx.float32:
        raise ValueError(
            "W8A16 multirow requires BF16 x, signed INT8 weight and FP32 scales"
        )
    if x.ndim < 2:
        raise ValueError("W8A16 multirow requires 2..8 activation rows")
    m = math.prod(x.shape[:-1])
    if not 2 <= m <= 8:
        raise ValueError("W8A16 multirow requires 2..8 activation rows")
    k = x.shape[-1]
    if k <= 0 or k % 32:
        raise ValueError("W8A16 multirow requires positive K divisible by 32")
    if weight.ndim != 2 or weight.shape[0] <= 0 or weight.shape[1] != k:
        raise ValueError("W8A16 multirow requires signed weight [N,K] with positive N")
    n = weight.shape[0]
    if scales.shape not in ((n,), (n, 1)):
        raise ValueError("W8A16 multirow requires scales [N] or [N,1]")
    output_dtype = mx.bfloat16 if output_dtype is None else output_dtype
    if output_dtype not in (mx.bfloat16, mx.float32):
        raise ValueError("W8A16 multirow output_dtype must be BF16 or FP32")
    if variant not in ("matrix", "wide", "nax"):
        raise ValueError("W8A16 multirow variant must be matrix, wide or nax")
    if variant == "nax" and (n % 32 or k % 64):
        raise ValueError("W8A16 NAX requires N divisible by32 and K divisible by64")
    if type(tile_rows) is not int or tile_rows not in (2, 4, 8):
        raise ValueError("W8A16 multirow tile_rows must be 2,4,8")
    if type(k_lanes) is not int or k_lanes not in (8, 16, 32):
        raise ValueError("W8A16 multirow k_lanes must be 8,16,32")
    if type(simdgroups) is not int or simdgroups not in (2, 4, 8):
        raise ValueError("W8A16 multirow simdgroups must be 2,4,8")
    stream = gpu_stream(stream, operator="W8A16 multirow")
    if variant == "nax":
        if not is_w8a8_available():
            raise RuntimeError("W8A16 NAX requires Apple M5+ and macOS26.2+")
        grid = (32, math.ceil(n / (32 * simdgroups)) * simdgroups, 1)
    elif variant == "matrix":
        grid = (32, math.ceil(n / (8 * simdgroups)) * simdgroups, 1)
    else:
        rows_per_group = (32 // k_lanes) * simdgroups
        grid = (
            math.ceil(m / tile_rows) * 32,
            math.ceil(n / rows_per_group) * simdgroups,
            1,
        )
    return _kernel(n, k, m, variant, tile_rows, k_lanes, simdgroups)(
        inputs=[x, weight, scales.reshape(n)],
        template=[("OT", output_dtype)],
        output_shapes=[(*x.shape[:-1], n)],
        output_dtypes=[output_dtype],
        grid=grid,
        threadgroup=(32, simdgroups, 1),
        stream=stream,
    )[0]
