from mlx.core import array

from ._types import StreamOrDevice

def is_w8a8_available() -> bool: ...
def w8a16_per_channel(
    x: array,
    weight: array,
    scales: array,
    *,
    stream: StreamOrDevice = None,
) -> array: ...
def int8_matmul_int32(
    x: array,
    weight: array,
    *,
    stream: StreamOrDevice = None,
) -> array: ...
def w8a8_linear(
    x: array,
    weight: array,
    scales: array,
    bias: array | None = None,
    *,
    stream: StreamOrDevice = None,
) -> array: ...
def spark_qkv_rope(
    x: array,
    weight: array,
    scales: array,
    biases: array,
    q_size: int,
    kv_size: int,
    num_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    rope_dims: int,
    rope_base: float,
    offset: int,
    group_size: int = 128,
    bits: int = 4,
    *,
    stream: StreamOrDevice = None,
) -> list[array]: ...
