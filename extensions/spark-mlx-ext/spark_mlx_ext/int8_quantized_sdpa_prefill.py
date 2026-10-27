"""Tiled multi-query attention over affine UINT8/G32 KV, D=256.

Internal implementation of the public ``int8_quantized_sdpa`` API. Thirty-two
query rows (interleaved across the GQA heads) share each decoded 16-token KV
tile. QK, online softmax and PV execute in one Metal dispatch without global
scores/probabilities or a dense copy of the cache. Q/K/V and probabilities use
the query dtype, with explicit casts for BF16 stores. Scores, normalizers and
output accumulators use FP32.

The SIMD fragment coordinate mapping follows MLX 0.32.4 steel/attn/mma.h
(Copyright Apple Inc., MIT). The tiled attention,
affine decoding, causal bounds and stride indexing are local implementations.
"""

from functools import lru_cache

import mlx.core as mx

from ._int8_prefill_sources import _HEADER, _PREFILL


@lru_cache(maxsize=1)
def _prefill_kernel():
    return mx.fast.metal_kernel(
        name="spark_affine_u8_sdpa_prefill",
        input_names=[
            "queries",
            "keys",
            "key_scales",
            "key_biases",
            "values",
            "value_scales",
            "value_biases",
            "scale",
        ],
        output_names=["out"],
        source=_PREFILL,
        header=_HEADER,
        ensure_row_contiguous=False,
        compile_options={"math_mode": "fast"},
    )


def _prefill_attention(queries, keys, values, *, scale, causal):
    """Launch after validation by the public wrapper; no evaluation or copies."""
    batch, heads, length, _ = queries.shape
    kv_heads = keys[0].shape[1]
    gqa = heads // kv_heads
    tiles = (length * gqa + 31) // 32
    return _prefill_kernel()(
        inputs=[queries, *keys, *values, mx.array(scale, mx.float32)],
        template=[("T", queries.dtype), ("G", gqa), ("CAUSAL", causal)],
        grid=(tiles * 128, kv_heads, batch),
        threadgroup=(128, 1, 1),
        output_shapes=[queries.shape],
        output_dtypes=[queries.dtype],
    )[0]
