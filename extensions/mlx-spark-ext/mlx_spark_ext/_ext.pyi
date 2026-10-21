from mlx.core import Device, Stream, array

def is_w8a8_available() -> bool: ...
def int8_matmul_int32(
    x: array,
    weight: array,
    *,
    stream: Stream | Device | None = None,
) -> array: ...
def w8a8_linear(
    x: array,
    weight: array,
    scales: array,
    bias: array | None = None,
    *,
    stream: Stream | Device | None = None,
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
    stream: Stream | Device | None = None,
) -> list[array]: ...
