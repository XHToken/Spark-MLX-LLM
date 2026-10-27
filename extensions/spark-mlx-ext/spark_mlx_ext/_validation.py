"""Shared device validation without evaluation or synchronization."""

import mlx.core as mx

from ._types import StreamOrDevice


def gpu_stream(stream: StreamOrDevice, *, operator: str) -> StreamOrDevice:
    if stream is None:
        device = mx.default_device()
    elif isinstance(stream, (mx.Stream, mx.ThreadLocalStream)):
        device = stream.device
    elif isinstance(stream, (mx.Device, mx.DeviceType)):
        device = stream
    else:
        raise TypeError("stream must be an MLX stream, device or None")
    if device != mx.gpu or not mx.metal.is_available():
        raise ValueError(f"{operator} requires Metal GPU execution")
    return stream
