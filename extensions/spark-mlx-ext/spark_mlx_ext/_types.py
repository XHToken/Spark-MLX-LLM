"""Internal type aliases used by the public Python operators."""

from typing import TypeAlias

import mlx.core as mx

StreamOrDevice: TypeAlias = (
    mx.Stream | mx.ThreadLocalStream | mx.Device | mx.DeviceType | None
)
QuantizedKV: TypeAlias = tuple[mx.array, mx.array, mx.array]
AttentionMask: TypeAlias = mx.array | str | None
