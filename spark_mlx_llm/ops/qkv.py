"""QKV/RoPE dispatch and implementations, sharing LinearConfig's W4 policy."""

from functools import cache

import mlx.core as mx
from mlx import nn

from .linear import AffineParameters, LinearConfig

__all__ = ["dispatch_qkv"]

try:
    from spark_mlx_ext import spark_qkv_rope
except ImportError:
    spark_qkv_rope = None


def dispatch_qkv(
    x,
    projection,
    rope,
    *,
    q_size,
    kv_size,
    n_heads,
    n_kv_heads,
    head_dim,
    offset,
    config: LinearConfig | None = None,
):
    config = config if config is not None else LinearConfig()
    batch_size, sequence_length, _ = x.shape
    quantized_qkv = mx.metal.is_available() and AffineParameters.accepts(projection)
    splitk_qkv = (
        quantized_qkv
        and config.w4_qmv_backend == "splitk"
        and AffineParameters.supports_splitk(
            x.shape, x.dtype, *AffineParameters.from_module(projection)
        )
    )
    use_native_qkv = (
        quantized_qkv
        and not splitk_qkv
        and callable(spark_qkv_rope)
        and batch_size == 1
        and sequence_length == 1
        and projection.bits == 4
        and projection.group_size in (32, 64, 128)
        and type(rope) is nn.RoPE
        and not rope.traditional
        and rope.scale == 1.0
        and isinstance(offset, int)
    )
    if use_native_qkv:
        return _fused_rope_projection(
            x,
            projection,
            rope,
            q_size=q_size,
            kv_size=kv_size,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            offset=offset,
        )
    if quantized_qkv:
        fused_qkv = _make_compiled_projection(
            q_size,
            kv_size,
            n_heads,
            n_kv_heads,
            head_dim,
            AffineParameters.splitk if splitk_qkv else AffineParameters.native,
        )
        queries, keys, values = fused_qkv(x, AffineParameters.from_module(projection))
    else:
        queries, keys, values = _module_projection(
            x,
            projection,
            q_size=q_size,
            kv_size=kv_size,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
        )
    return _apply_rope(queries, keys, values, rope, offset)


@cache
def _make_compiled_projection(
    q_size, kv_size, n_heads, n_kv_heads, head_dim, project=AffineParameters.native
):
    """Cache a graph for an already selected linear implementation."""

    @mx.compile
    def compiled_projection(x, projection_params):
        return _split_heads(
            project(x, *projection_params),
            q_size=q_size,
            kv_size=kv_size,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
        )

    return compiled_projection


def _fused_rope_projection(
    x,
    projection,
    rope,
    *,
    q_size,
    kv_size,
    n_heads,
    n_kv_heads,
    head_dim,
    offset,
):
    return spark_qkv_rope(
        x,
        projection.weight,
        projection.scales,
        projection.biases,
        q_size,
        kv_size,
        n_heads,
        n_kv_heads,
        head_dim,
        rope.dims,
        rope.base,
        offset,
        projection.group_size,
        projection.bits,
    )


def _split_heads(qkv, *, q_size, kv_size, n_heads, n_kv_heads, head_dim):
    """Materialize Q/K/V views before RoPE, including on CUDA."""
    batch_size, sequence_length, _ = qkv.shape
    queries, keys, values = mx.split(
        qkv,
        [q_size, q_size + kv_size],
        axis=-1,
    )
    queries = queries.reshape(batch_size, sequence_length, n_heads, head_dim).transpose(
        0, 2, 1, 3
    )
    keys = keys.reshape(batch_size, sequence_length, n_kv_heads, head_dim).transpose(
        0, 2, 1, 3
    )
    values = values.reshape(
        batch_size, sequence_length, n_kv_heads, head_dim
    ).transpose(0, 2, 1, 3)
    queries = mx.contiguous(queries)
    keys = mx.contiguous(keys)
    values = mx.contiguous(values)
    return queries, keys, values


def _module_projection(
    x, projection, *, q_size, kv_size, n_heads, n_kv_heads, head_dim
):
    return _split_heads(
        projection(x),
        q_size=q_size,
        kv_size=kv_size,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
    )


def _apply_rope(queries, keys, values, rope, offset):
    return rope(queries, offset=offset), rope(keys, offset=offset), values
