# Copyright © 2026 Apple Inc.

from importlib import import_module as _import_module
from importlib.metadata import version as _package_version

if _package_version("mlx") != "0.32.4":
    raise ImportError("spark-mlx-ext requires official mlx==0.32.4")

# Load libmlx before the native module, including for editable installations.
_import_module("mlx.core")

from ._ext import (
    int8_matmul_int32,
    is_w8a8_available,
    spark_qkv_rope,
    w8a8_linear,
    w8a16_per_channel,
)
from .bf16_sdpa import bf16_sdpa
from .int8_quantized_sdpa import int8_quantized_sdpa
from .linear import W8A8Linear
from .w4a16_qmv import w4a16_qmv

__all__ = [
    "W8A8Linear",
    "bf16_sdpa",
    "int8_matmul_int32",
    "int8_quantized_sdpa",
    "is_w8a8_available",
    "spark_qkv_rope",
    "w4a16_qmv",
    "w8a8_linear",
    "w8a16_per_channel",
]
