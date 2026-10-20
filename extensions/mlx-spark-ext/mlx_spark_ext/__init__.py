# Copyright © 2026 Apple Inc.

from importlib.metadata import version as _package_version

if _package_version("mlx") != "0.32.4":
    raise ImportError("mlx-spark-ext requires official mlx==0.32.4")

from ._ext import (
    int8_matmul_int32,
    is_w8a8_available,
    spark_qkv_rope,
    w8a8_linear,
)
from .linear import W8A8Linear

__all__ = [
    "W8A8Linear",
    "int8_matmul_int32",
    "is_w8a8_available",
    "spark_qkv_rope",
    "w8a8_linear",
]
