"""Immutable native Markov bias cache with a device-conditional BF16 fallback.

The caller supplies cache entries computed from the same embedding/projection.
Cache hits copy their stored native result; misses compute a FP32 dot followed
by one BF16 output rounding. No weights or cache arrays are modified.
"""

import math
from functools import lru_cache
from string import Template

import mlx.core as mx

from ._cached_markov_sources import HEADER, SOURCE
from ._types import StreamOrDevice
from ._validation import gpu_stream

__all__ = ["cached_markov_bias"]


@lru_cache(maxsize=32)
def _kernel(v, n, r, c):
    return mx.fast.metal_kernel(
        name=f"spark_cached_markov_v{v}_n{n}_r{r}_c{c}",
        input_names=["ids", "embedding", "projection", "cache", "lookup"],
        output_names=["out"],
        header=HEADER,
        source=Template(SOURCE).substitute(V=v, N=n, R=r, C=c),
        ensure_row_contiguous=True,
        compile_options={"math_mode": "safe"},
    )


def cached_markov_bias(
    ids, embedding, projection, cache, lookup, *, stream: StreamOrDevice = None
):
    """Return Markov biases for valid integer IDs, using immutable cache rows.

    Embedding is BF16[V,R], projection BF16[N,R], cache BF16[C,N], lookup
    int32[V]. Valid lookup entries select cache rows; negative or >=C entries
    fall back to a dot product. R must be positive and divisible by32; V,N,C
    are positive. IDs have nonempty shape and int32 dtype; the output has shape
    ``(*ids.shape,N)`` and BF16 dtype. Out-of-vocabulary IDs produce NaNs rather
    than out-of-bounds reads; the caller is responsible for valid token IDs and
    for keeping cache entries consistent with its supplied weights.

    Cache hits execute no projection. The fallback changes FP32 reduction
    ordering relative to native GEMV and requires numerical/quality validation.
    No eval, synchronization, activation quantization, automatic cache creation
    or CPU fallback is performed. Noncontiguous inputs are copied lazily.
    """
    values = (ids, embedding, projection, cache, lookup)
    if any(not isinstance(x, mx.array) for x in values):
        raise TypeError("Cached Markov inputs must be MLX arrays")
    if ids.dtype != mx.int32 or lookup.dtype != mx.int32:
        raise ValueError("Cached Markov IDs and lookup must be int32")
    if any(x.dtype != mx.bfloat16 for x in [embedding, projection, cache]):
        raise ValueError("Cached Markov weights and cache must be BF16")
    if ids.ndim < 1 or math.prod(ids.shape) == 0:
        raise ValueError("Cached Markov IDs must have nonempty shape")
    if embedding.ndim != 2 or projection.ndim != 2 or cache.ndim != 2:
        raise ValueError("Cached Markov weights/cache must be matrices")
    v, r = embedding.shape
    n, rank = projection.shape
    c, columns = cache.shape
    if (
        min(v, r, n, c) <= 0
        or r % 32
        or rank != r
        or columns != n
        or lookup.shape != (v,)
    ):
        raise ValueError(
            "Cached Markov requires embedding[V,R],projection[N,R],cache[C,N],lookup[V],R%32=0"
        )
    selected = gpu_stream(stream, operator="Cached Markov")
    return _kernel(v, n, r, c)(
        inputs=list(values),
        output_shapes=[(*ids.shape, n)],
        output_dtypes=[mx.bfloat16],
        grid=(math.ceil(n * 8 / 128) * 128, math.prod(ids.shape), 1),
        threadgroup=(128, 1, 1),
        stream=selected,
    )[0]
