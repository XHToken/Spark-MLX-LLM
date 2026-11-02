"""Precise BF16 attention for the fixed Spark M8/N519 sliding geometry.

QK scores, softmax and accumulation are FP32. Each probability is represented
as a high plus residual BF16 term for two MPP PV products. Output rounds to
BF16 once. Other valid geometries use native SDPA. The optimized route requires
M5 MPP and the default GPU stream; it never changes input values or masks.
"""

import math
from functools import lru_cache

import mlx.core as mx

from ._bf16_multiquery_sources import HEADER, MERGE, PV, QK, SOFTMAX
from ._ext import is_w8a8_available
from ._types import StreamOrDevice
from ._validation import gpu_stream

__all__ = ["bf16_sdpa_multiquery"]


def _validate_inputs(queries, keys, values, scale, mask):
    if any(not isinstance(x, mx.array) for x in (queries, keys, values)):
        raise TypeError("Q/K/V must be MLX arrays")
    if any(x.ndim != 4 or x.dtype != mx.bfloat16 for x in (queries, keys, values)):
        raise ValueError("BF16 multiquery SDPA requires four-dimensional BF16 Q/K/V")
    if any(d <= 0 for x in (queries, keys, values) for d in x.shape):
        raise ValueError("BF16 multiquery SDPA requires nonempty dimensions")
    if (
        keys.shape[:3] != values.shape[:3]
        or queries.shape[0] != keys.shape[0]
        or queries.shape[-1] != keys.shape[-1]
        or queries.shape[1] % keys.shape[1]
    ):
        raise ValueError("Q/K/V batch, head, key length or key dimension mismatch")
    if queries.shape[2] > keys.shape[2]:
        raise ValueError("Query length must not exceed KV length")
    if type(scale) not in (float, int):
        raise ValueError("scale must be a finite FP32-range Python int or float")
    try:
        scale = float(scale)
    except OverflowError as error:
        raise ValueError(
            "scale must be a finite FP32-range Python int or float"
        ) from error
    if not math.isfinite(scale) or abs(scale) > 3.4028234663852886e38:
        raise ValueError("scale must be a finite FP32-range Python int or float")
    if isinstance(mask, mx.array):
        if mask.dtype not in (mx.bool_, mx.bfloat16) or mask.ndim > 4:
            raise ValueError(
                "BF16 attention mask must be boolean or BF16 with <=4 dimensions"
            )
        scores_shape = (*queries.shape[:3], keys.shape[2])
        try:
            broadcast = mx.broadcast_shapes(mask.shape, scores_shape)
        except ValueError as error:
            raise ValueError("Attention mask does not broadcast to scores") from error
        if broadcast != scores_shape:
            raise ValueError("Attention mask must not enlarge scores shape")
    elif mask is not None and not (isinstance(mask, str) and mask == "causal"):
        raise ValueError("Attention mask must be None, 'causal' or an array")
    return scale


def _supports_precise(queries, keys, values, scale, mask):
    return (
        queries.shape == (1, 16, 8, 256)
        and keys.shape == values.shape == (1, 4, 519, 256)
        and scale == 0.0625
        and isinstance(mask, mx.array)
        and mask.dtype == mx.bool_
        and mask.shape == (8, 519)
    )


@lru_cache(maxsize=1)
def _kernels():
    options = {"math_mode": "safe"}
    qk = mx.fast.metal_kernel(
        name="spark_precise_bf16_m8_qk",
        input_names=["queries", "keys", "mask", "scale"],
        output_names=["scores"],
        source=QK,
        header=HEADER,
        ensure_row_contiguous=False,
        compile_options=options,
    )
    softmax = mx.fast.metal_kernel(
        name="spark_precise_bf16_m8_softmax",
        input_names=["scores"],
        output_names=["highp", "lowp"],
        source=SOFTMAX,
        header=HEADER,
        compile_options=options,
    )
    pv = mx.fast.metal_kernel(
        name="spark_precise_bf16_m8_pv",
        input_names=["highp", "lowp", "values"],
        output_names=["partials"],
        source=PV,
        header=HEADER,
        ensure_row_contiguous=False,
        compile_options=options,
    )
    merge = mx.fast.metal_kernel(
        name="spark_precise_bf16_m8_merge",
        input_names=["partials"],
        output_names=["out"],
        source=MERGE,
        header=HEADER,
        compile_options=options,
    )
    return qk, softmax, pv, merge


@mx.compile
def _compiled_precise(queries, keys, values, mask):
    qk, softmax, pv, merge = _kernels()
    canonical_mask = mx.broadcast_to(mask, (1, 16, 8, 519))
    scores = qk(
        inputs=[queries, keys, canonical_mask, mx.array(0.0625, mx.float32)],
        template=[("TM", 16), ("SG", 1), ("MASK_MODE", 2)],
        grid=(544, 8, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(16, 8, 519)],
        output_dtypes=[mx.float32],
    )[0]
    high, low = softmax(
        inputs=[scores],
        grid=(4096, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(16, 8, 519), (16, 8, 519)],
        output_dtypes=[mx.bfloat16, mx.bfloat16],
    )
    partials = pv(
        inputs=[high, low, values],
        template=[("TM", 16), ("SG", 1), ("S", 4), ("OUT_TYPE", mx.float32)],
        grid=(256, 8, 4),
        threadgroup=(32, 1, 1),
        output_shapes=[(4, 16, 8, 256)],
        output_dtypes=[mx.float32],
    )[0]
    return merge(
        inputs=[partials],
        template=[("S", 4)],
        grid=(32768, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(1, 16, 8, 256)],
        output_dtypes=[mx.bfloat16],
    )[0]


def bf16_sdpa_multiquery(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    *,
    scale: float | int,
    mask: mx.array | str | None = None,
    stream: StreamOrDevice = None,
) -> mx.array:
    """Return BF16 SDPA, using precise MPP for supported fixed sliding inputs.

    Q/K/V are BF16 [B,H,M,D] arrays with ordinary SDPA geometry. Boolean and
    BF16 masks broadcast to [B,Hq,M,N]; None and bottom-right 'causal' masks
    are accepted. Output has Q's first three dimensions and V's final size.

    The optimized case is Q[1,16,8,256], KV[1,4,519,256], scale1/16 and a
    boolean [8,519] mask, on M5+ and the default GPU stream. Its mask values
    remain explicit compiled-function inputs. Arbitrary input strides and
    nonzero cache offsets are supported without mutation. Scores/softmax
    and partial sums remain FP32; probability hi+lo BF16 representation
    introduces a small documented arithmetic error checked against FP64.

    All other valid geometries, dtypes of masks, scales, hardware capabilities
    and explicit streams retain native BF16 SDPA. CPU streams are rejected.
    This inference operator constructs lazy results without synchronization.
    """
    scale = _validate_inputs(queries, keys, values, scale, mask)
    stream = gpu_stream(stream, operator="BF16 multiquery SDPA")
    if (
        stream is None
        and _supports_precise(queries, keys, values, scale, mask)
        and is_w8a8_available()
    ):
        return _compiled_precise(queries, keys, values, mask)
    return mx.fast.scaled_dot_product_attention(
        queries, keys, values, scale=scale, mask=mask, stream=stream
    )
