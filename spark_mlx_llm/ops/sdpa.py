"""SDPA configuration, dispatch and native/INT8/BF16 implementations."""

from dataclasses import dataclass

import mlx.core as mx
from mlx_lm.models.base import scaled_dot_product_attention

from .base import OperatorConfig

__all__ = ["SDPAConfig", "dispatch_sdpa"]


@dataclass(kw_only=True)
class SDPAConfig(OperatorConfig):
    """Prefer available kernels; KV cache quantization is configured separately."""

    fused_quantized_sdpa: bool = True
    fused_quantized_sdpa_prefill: bool = False
    tuned_bf16_sdpa: bool = False


try:
    from spark_mlx_ext import int8_quantized_sdpa
except ImportError:
    int8_quantized_sdpa = None

try:
    from spark_mlx_ext import bf16_sdpa
except ImportError:
    bf16_sdpa = None


def dispatch_sdpa(
    queries,
    keys,
    values,
    *,
    cache,
    scale,
    mask,
    config: SDPAConfig | None = None,
    is_sliding=False,
):
    config = config if config is not None else SDPAConfig()
    if config.fused_quantized_sdpa and _supports_int8(
        queries,
        keys,
        values,
        cache=cache,
        mask=mask,
        prefill=config.fused_quantized_sdpa_prefill,
    ):
        return _int8_attention(
            queries, keys, values, cache=cache, scale=scale, mask=mask
        )
    if (
        config.tuned_bf16_sdpa
        and not is_sliding
        and _supports_bf16(queries, keys, values, mask=mask)
    ):
        return _bf16_attention(queries, keys, values, scale=scale, mask=mask)
    return _native_attention(queries, keys, values, cache=cache, scale=scale, mask=mask)


def _supports_int8(queries, keys, values, *, cache, mask, prefill=False):
    return (
        callable(int8_quantized_sdpa)
        and getattr(cache, "bits", None) == 8
        and getattr(cache, "group_size", None) == 32
        and queries.ndim == 4
        and queries.shape[-2] > 0
        and (queries.shape[-2] == 1 or prefill)
        and queries.shape[-1] == 256
        and isinstance(keys, (tuple, list))
        and isinstance(values, (tuple, list))
        and len(keys) == len(values) == 3
        and keys[0].ndim == values[0].ndim == 4
        and keys[0].shape[-1] == values[0].shape[-1] == 64
        and keys[0].shape[1] > 0
        and queries.shape[1] % keys[0].shape[1] == 0
        and queries.shape[1] // keys[0].shape[1] in (1, 2, 4, 8)
        and queries.dtype in (mx.float16, mx.bfloat16, mx.float32)
        and all(x.dtype == queries.dtype for x in (*keys[1:], *values[1:]))
        and (mask is None or (isinstance(mask, str) and mask == "causal"))
        and mx.default_device() == mx.gpu
        and mx.metal.is_available()
    )


def _supports_bf16(queries, keys, values, *, mask):
    return (
        callable(bf16_sdpa)
        and isinstance(keys, mx.array)
        and isinstance(values, mx.array)
        and queries.shape == (1, 16, 1, 256)
        and keys.ndim == values.ndim == 4
        and keys.shape == values.shape
        and keys.shape[:2] == (1, 4)
        and keys.shape[-1] == 256
        and 2048 <= keys.shape[2] <= 131072
        and queries.dtype == keys.dtype == values.dtype == mx.bfloat16
        and (mask is None or (isinstance(mask, str) and mask == "causal"))
        and mx.default_device() == mx.gpu
        and mx.metal.is_available()
    )


def _native_attention(queries, keys, values, *, cache, scale, mask):
    return scaled_dot_product_attention(
        queries, keys, values, cache=cache, scale=scale, mask=mask
    )


def _int8_attention(queries, keys, values, *, cache, scale, mask):
    if not callable(int8_quantized_sdpa):
        raise RuntimeError(  # noqa: TRY004 -- missing optional extension
            "fused_quantized_sdpa requires spark-mlx-ext with the "
            "int8_quantized_sdpa API; install or rebuild the extension and restart Python"
        )
    return int8_quantized_sdpa(
        queries,
        keys,
        values,
        scale=scale,
        group_size=cache.group_size,
        bits=cache.bits,
        mask=mask,
    )


def _bf16_attention(queries, keys, values, *, scale, mask):
    if not callable(bf16_sdpa):
        raise RuntimeError(  # noqa: TRY004 -- missing optional extension
            "tuned_bf16_sdpa requires spark-mlx-ext with the bf16_sdpa API; "
            "install the updated extension package and restart Python"
        )
    return bf16_sdpa(queries, keys, values, scale=scale, mask=mask)
