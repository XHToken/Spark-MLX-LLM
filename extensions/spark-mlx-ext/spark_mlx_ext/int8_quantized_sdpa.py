"""Fused affine UINT8/G32 KV attention for decode and chunked prefill.

The online attention structure follows MLX 0.32.4 sdpa_vector.h
(Copyright Apple Inc., MIT). Packed loads, GQA
reuse, checked 32-bit offsets, stride handling and FP32 partials are
implemented here. Short decode contexts use a single pass with threadgroup-local
reduction and contiguous pitches; longer contexts retain the two-pass path.
KV storage is quantized, arithmetic is FP32.
"""

import math
import struct
from functools import lru_cache

import mlx.core as mx

from ._int8_sdpa_sources import _ATTENTION, _MERGE, _SHORT_CONTIGUOUS_ATTENTION
from ._types import AttentionMask, QuantizedKV
from .int8_quantized_sdpa_prefill import _prefill_attention


@lru_cache(maxsize=1)
def _kernels():
    attention = mx.fast.metal_kernel(
        name="spark_affine_u8_sdpa_decode",
        input_names=[
            "queries",
            "keys",
            "key_scales",
            "key_biases",
            "values",
            "value_scales",
            "value_biases",
            "scale",
            "length",
        ],
        output_names=["partials", "sums", "maxs"],
        source=_ATTENTION,
        ensure_row_contiguous=False,
        compile_options={"math_mode": "fast"},
    )
    merge = mx.fast.metal_kernel(
        name="spark_affine_u8_sdpa_merge",
        input_names=["partials", "sums", "maxs"],
        output_names=["out"],
        source=_MERGE,
        compile_options={"math_mode": "fast"},
    )
    return attention, merge


@lru_cache(maxsize=1)
def _short_kernel():
    return mx.fast.metal_kernel(
        name="spark_affine_u8_sdpa_short",
        input_names=[
            "queries",
            "keys",
            "key_scales",
            "key_biases",
            "values",
            "value_scales",
            "value_biases",
        ],
        output_names=["out"],
        source=_SHORT_CONTIGUOUS_ATTENTION,
        ensure_row_contiguous=True,
        compile_options={"math_mode": "fast"},
    )


@lru_cache(maxsize=64)
def _short_scale_bits(scale):
    # A signed template integer retains every float32 bit pattern, including
    # negative scales, without exceeding a signed 32-bit template argument.
    return struct.unpack("<i", struct.pack("<f", float(scale)))[0]


def _short_attention(
    queries,
    keys,
    values,
    *,
    scale,
    simds=32,
    heads_per_simd=1,
    tokens_per_iter=1,
    merge_width=2,
):
    """Single-pass compact pitches; MLX materializes noncontiguous views."""
    batch, heads, _, _ = queries.shape
    kv_heads = keys[0].shape[1]
    gqa = heads // kv_heads
    if (
        simds not in (8, 16, 32)
        or heads_per_simd not in (1, 2, 4, 8)
        or gqa % heads_per_simd
        or simds * heads_per_simd > 32
        or tokens_per_iter not in (1, 2, 4)
        or merge_width not in (1, 2, 4)
    ):
        raise ValueError("Invalid short SDPA launch tuning")
    return _short_kernel()(
        inputs=[queries, *keys, *values],
        template=[
            ("T", queries.dtype),
            ("G", gqa),
            ("KH", kv_heads),
            ("HPT", heads_per_simd),
            ("NS", simds),
            ("TK", tokens_per_iter),
            ("MW", merge_width),
            ("SCALE_BITS", _short_scale_bits(scale)),
        ],
        grid=(32 * simds, batch * heads // heads_per_simd, 1),
        threadgroup=(32 * simds, 1, 1),
        output_shapes=[queries.shape],
        output_dtypes=[queries.dtype],
    )[0]


def _fallback(queries, keys, values, scale, group_size, bits, mask):
    """Same staged arithmetic as MLX-LM, without mutating the query descriptor."""
    batch, heads, length, dim = queries.shape
    kv_heads = keys[0].shape[1]
    repeats = heads // kv_heads
    q = queries * scale
    if repeats > 1:
        q = q.reshape(batch, kv_heads, repeats, length, dim)
        keys = tuple(mx.expand_dims(x, -3) for x in keys)
        values = tuple(mx.expand_dims(x, -3) for x in values)
    scores = mx.quantized_matmul(
        q, *keys, transpose=True, group_size=group_size, bits=bits
    )
    if mask is not None:
        if isinstance(mask, str):
            if mask != "causal":
                raise ValueError("mask string must be 'causal'")
            n = scores.shape[-1]
            mask = mx.arange(n - length, n)[:, None] >= mx.arange(n)[None, :]
        if repeats > 1 and mask.ndim >= 3:
            mask = mx.broadcast_to(mask, (batch, heads, length, scores.shape[-1]))
            mask = mask.reshape(batch, kv_heads, repeats, length, scores.shape[-1])
        scores = (
            mx.where(mask, scores, mx.finfo(scores.dtype).min)
            if mask.dtype == mx.bool_
            else scores + mask
        )
    probs = mx.softmax(scores, axis=-1, precise=True)
    out = mx.quantized_matmul(
        probs, *values, transpose=False, group_size=group_size, bits=bits
    )
    return out.reshape(batch, heads, length, values[0].shape[-1] * (32 // bits))


def int8_quantized_sdpa(
    queries: mx.array,
    keys: QuantizedKV,
    values: QuantizedKV,
    *,
    scale: float | int,
    group_size: int = 32,
    bits: int = 8,
    mask: AttentionMask = None,
    single_pass_max_length: int = 1023,
) -> mx.array:
    """Attention over MLX affine packed KV tuples (weights, scales, biases).

    Fuses D256 attention, GQA ratios 1/2/4/8 on Metal. Multi-token inputs use
    tiled QK/online-softmax/PV, with no mask or bottom-right causal masking;
    single-token inputs retain the existing one-/two-pass decode policy.
    Other supported MLX quantization/attention shapes use a staged fallback.
    Sliced cache capacity, arbitrary array strides and batch>1 are supported.
    Short contexts materialize noncontiguous inputs before their single-pass
    kernel; longer contexts read original strides directly. Short length is
    read from the input shape, so a new length needs no Metal specialization.
    Inputs are never modified. Inference only; no autodiff rule is provided.
    The GPU kernels use FP32 accumulation (and FP32 decode partials), and cast only the
    final output to the query dtype; it need not match staged BF16 bitwise.

    single_pass_max_length is an inclusive, nonnegative Python int policy.
    This policy applies only to single-query decode. Zero selects two passes
    for eligible decode inputs. The default is 1023:
    N < 1024 uses a single pass and N >= 1024 uses two passes, subject to the
    existing index/scale guards. A larger limit permits longer single-pass
    inputs without promising optimal performance.
    """
    if type(single_pass_max_length) is not int or single_pass_max_length < 0:
        raise ValueError("single_pass_max_length must be a nonnegative Python int")
    if queries.ndim != 4 or len(keys) != 3 or len(values) != 3:
        raise ValueError(
            "Expected Q[B,H,L,D] and affine (packed, scales, biases) KV tuples"
        )
    batch, heads, length, dim = queries.shape
    if any(x.ndim != 4 for x in (*keys, *values)):
        raise ValueError("Every packed KV and metadata array must have four dimensions")
    kv_heads, n = keys[0].shape[1:3]
    if min(batch, heads, length, dim, kv_heads, n) <= 0 or heads % kv_heads:
        raise ValueError(
            "Nonempty inputs and query heads divisible by KV heads required"
        )
    if not isinstance(scale, (float, int)) or not math.isfinite(scale):
        raise ValueError("scale must be a finite Python number")
    if bits not in (2, 4, 8) or group_size not in (32, 64, 128):
        raise ValueError("Supported affine formats: bits 2/4/8, group size 32/64/128")
    pack = 32 // bits
    for name, tensors in (("key", keys), ("value", values)):
        packed, scales, biases = tensors
        width = dim if name == "key" else packed.shape[-1] * pack
        if (
            width <= 0
            or width % group_size
            or packed.shape != (batch, kv_heads, n, width // pack)
            or scales.shape != (batch, kv_heads, n, width // group_size)
            or biases.shape != scales.shape
            or packed.dtype != mx.uint32
            or scales.dtype != queries.dtype
            or biases.dtype != queries.dtype
        ):
            raise ValueError(
                f"Incompatible {name} packed shape, affine metadata or dtype"
            )
    if queries.dtype not in (mx.float16, mx.bfloat16, mx.float32):
        raise ValueError("Q and metadata must be float16, bfloat16 or float32")
    simple_mask = mask is None or (isinstance(mask, str) and mask == "causal")
    fused = (
        bits == 8
        and group_size == 32
        and dim == 256
        and values[0].shape[-1] == 64
        and heads // kv_heads in (1, 2, 4, 8)
        and simple_mask
        and mx.default_device() == mx.gpu
        and mx.metal.is_available()
    )
    if not fused:
        return _fallback(queries, keys, values, scale, group_size, bits, mask)
    if length > 1:
        causal = isinstance(mask, str)
        if (
            (causal and length > n)
            or max(length * (heads // kv_heads), n) >= 2147483647
            or abs(scale) > 3.4028234663852886e38
        ):
            return _fallback(queries, keys, values, scale, group_size, bits, mask)
        return _prefill_attention(queries, keys, values, scale=scale, causal=causal)
    if (
        n <= single_pass_max_length
        and batch * kv_heads * n * 64 + 32 * 64 + 63 < 2147483647
        and batch * heads * 256 < 2147483647
        and abs(scale) <= 3.4028234663852886e38
    ):
        return _short_attention(queries, keys, values, scale=scale)
    splits = 64 if n <= 2048 else (128 if n < 16384 else 256)
    heads_per_simd = min(heads // kv_heads, 2)
    attention, merge = _kernels()
    partials, sums, maxs = attention(
        inputs=[
            queries,
            *keys,
            *values,
            mx.array(scale, mx.float32),
            mx.array(n, mx.int32),
        ],
        template=[
            ("G", heads // kv_heads),
            ("KH", kv_heads),
            ("HPT", heads_per_simd),
            ("S", splits),
        ],
        grid=(splits * 32, batch * heads // heads_per_simd, 1),
        threadgroup=(128, heads // kv_heads // heads_per_simd, 1),
        output_shapes=[
            (batch * heads, splits, 256),
            (batch * heads, splits),
            (batch * heads, splits),
        ],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    return merge(
        inputs=[partials, sums, maxs],
        template=[("T", queries.dtype), ("S", splits)],
        grid=(256, batch * heads, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[queries.shape],
        output_dtypes=[queries.dtype],
    )[0]
