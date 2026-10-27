"""BF16 two-pass vector SDPA for Spark single-token GPU inference.

The online softmax and merge follow MLX 0.32.4 sdpa_vector.h (Copyright
Apple Inc., MIT). This port retains the verified
length-dependent split policy and TK=4 implementation. Q/K/V, partials and
output use BF16 storage; multiplication, accumulation, softmax and merge use
FP32. Input strides are read directly; no KV copy or synchronous evaluation
is performed. This inference operator has no autodiff rule.
"""

import math
from functools import lru_cache
from string import Template

import mlx.core as mx

from ._bf16_sdpa_sources import _HEADER, _PASS1_BASE, _PASS2_BASE
from ._types import StreamOrDevice
from ._validation import gpu_stream

__all__ = ["bf16_sdpa"]


def _split_count(length):
    """Select the verified policy within the supported KV-length range."""
    if type(length) is not int or not 2048 <= length <= 131072:
        raise ValueError("BF16 SDPA requires KV length in [2048, 131072]")
    return 64 if length < 32768 else (128 if length < 131072 else 256)


@lru_cache(maxsize=3)
def _kernels(splits):
    # Cache compiled kernel callables only. Every call retains its actual
    # Q/K/V graph dependencies and creates fresh scalar and output arrays.
    first_source = Template(_PASS1_BASE).substitute(SPLITS=splits)
    second_source = Template(_PASS2_BASE).substitute(SPLITS=splits)
    first = mx.fast.metal_kernel(
        name=f"spark_bf16_sdpa_2pass_1_s{splits}",
        input_names=["queries", "keys", "values", "scale", "length"],
        output_names=["partials", "sums", "maxs"],
        source=first_source,
        header=_HEADER,
        ensure_row_contiguous=False,
        compile_options={"math_mode": "fast"},
    )
    second = mx.fast.metal_kernel(
        name=f"spark_bf16_sdpa_2pass_2_s{splits}",
        input_names=["partials", "sums", "maxs"],
        output_names=["out"],
        source=second_source,
        header=_HEADER,
        compile_options={"math_mode": "fast"},
    )
    return first, second


def bf16_sdpa(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    *,
    scale: float | int,
    mask: str | None = None,
    stream: StreamOrDevice = None,
) -> mx.array:
    """Return lazy two-pass attention for the supported Spark decode shape.

    Q must be BF16 [1, 16, 1, 256]; K and V must both be BF16
    [1, 4, N, 256], with 2048 <= N <= 131072. Arbitrary valid MLX input
    strides, including sliced cache capacity and the final dimension, are
    supported without materializing KV copies. Inputs are never modified.

    ``scale`` is a finite Python int/float representable as FP32. ``mask``
    may be None or "causal": for a single query aligned to the end of the
    cache, every supplied KV token is visible. Other masks/shapes/dtypes
    raise ValueError; callers may select native SDPA as their fallback.
    ``stream`` follows MLX's stream/device convention and must select GPU.

    Splits are 64 below 32768 tokens, 128 below 131072, and 256 at 131072.
    This is an inference operator. It returns an MLX graph without eval or
    synchronize and does not provide an autodiff rule. Split-dependent
    BF16 partial rounding can differ from native SDPA's output.
    """
    if any(not isinstance(x, mx.array) for x in (queries, keys, values)):
        raise TypeError("Q/K/V must be MLX arrays")
    if any(x.dtype != mx.bfloat16 or x.ndim != 4 for x in (queries, keys, values)):
        raise ValueError("BF16 SDPA requires four-dimensional BF16 Q/K/V")
    if (
        queries.shape != (1, 16, 1, 256)
        or keys.shape != values.shape
        or keys.shape[:2] != (1, 4)
        or keys.shape[-1] != 256
    ):
        raise ValueError("BF16 SDPA requires Q[1,16,1,256] and KV[1,4,N,256]")
    splits = _split_count(keys.shape[2])
    if not (mask is None or isinstance(mask, str) and mask == "causal"):
        raise ValueError("BF16 SDPA supports only mask=None or 'causal'")
    if type(scale) not in (float, int):
        raise ValueError("scale must be a finite FP32-range Python int or float")
    try:
        scale = float(scale)
    except OverflowError as exc:
        raise ValueError(
            "scale must be a finite FP32-range Python int or float"
        ) from exc
    if not math.isfinite(scale) or abs(scale) > 3.4028234663852886e38:
        raise ValueError("scale must be a finite FP32-range Python int or float")
    stream = gpu_stream(stream, operator="BF16 SDPA")
    first, second = _kernels(splits)
    partials, sums, maxs = first(
        inputs=[
            queries,
            keys,
            values,
            mx.array(scale, mx.float32),
            mx.array(keys.shape[2], mx.int32),
        ],
        template=[("KH", 4)],
        grid=(128, 4, splits),
        threadgroup=(32, 4, 1),
        output_shapes=[(16, splits, 256), (16, splits), (16, splits)],
        output_dtypes=[mx.bfloat16, mx.float32, mx.float32],
        stream=stream,
    )
    return second(
        inputs=[partials, sums, maxs],
        grid=(16384, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[queries.shape],
        output_dtypes=[mx.bfloat16],
        stream=stream,
    )[0]
