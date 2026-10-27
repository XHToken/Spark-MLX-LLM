// Copyright © 2026 Apple Inc.

#include <cmath>
#include <cstdint>

#include "spark_qkv_rope.h"
#include "mlx/fast.h"
#include "mlx/ops.h"

namespace spark_mlx_ext {
using namespace mlx::core;
using mlx::core::fast::rope;

std::vector<array> spark_qkv_rope(
    const array& x,
    const array& weight,
    const array& scales,
    const array& biases,
    int q_size,
    int kv_size,
    int num_heads,
    int num_key_value_heads,
    int head_dim,
    int rope_dims,
    float rope_base,
    int offset,
    int group_size,
    int bits,
    StreamOrDevice s) {
  if (x.ndim() != 3 || x.shape(0) != 1 || x.shape(1) != 1 ||
      x.shape(2) <= 0) {
    throw std::invalid_argument(
        "[spark_qkv_rope] x must have shape [1, 1, hidden] with hidden > 0.");
  }
  if (!issubdtype(x.dtype(), floating)) {
    throw std::invalid_argument(
        "[spark_qkv_rope] x must have a real floating dtype.");
  }
  if (num_heads <= 0 || num_key_value_heads <= 0 || head_dim <= 0 ||
      num_heads % num_key_value_heads != 0 ||
      q_size != int64_t{num_heads} * head_dim ||
      kv_size != int64_t{num_key_value_heads} * head_dim) {
    throw std::invalid_argument(
        "[spark_qkv_rope] Invalid head counts, head_dim, q_size, or kv_size.");
  }
  if (rope_dims <= 0 || rope_dims % 2 != 0 || rope_dims > head_dim ||
      !std::isfinite(rope_base) || rope_base <= 0) {
    throw std::invalid_argument(
        "[spark_qkv_rope] rope_dims must be positive, even, and no larger than "
        "head_dim; rope_base must be positive and finite.");
  }
  if (group_size != 32 && group_size != 64 && group_size != 128) {
    throw std::invalid_argument(
        "[spark_qkv_rope] group_size must be 32, 64, or 128.");
  }
  if (bits != 1 && bits != 2 && bits != 3 && bits != 4 && bits != 5 &&
      bits != 6 && bits != 8) {
    throw std::invalid_argument(
        "[spark_qkv_rope] bits must be 1, 2, 3, 4, 5, 6, or 8.");
  }
  if (weight.ndim() != 2 || scales.ndim() != 2 || biases.ndim() != 2) {
    throw std::invalid_argument(
        "[spark_qkv_rope] weight, scales, and biases must be matrices.");
  }
  if (weight.dtype() != uint32 || !issubdtype(scales.dtype(), floating) ||
      !issubdtype(biases.dtype(), floating)) {
    throw std::invalid_argument(
        "[spark_qkv_rope] weight must be uint32; scales and biases must have "
        "real floating dtypes.");
  }
  if (x.shape(2) % group_size != 0 ||
      int64_t{weight.shape(0)} != int64_t{q_size} + 2 * int64_t{kv_size} ||
      int64_t{weight.shape(1)} * 32 != int64_t{x.shape(2)} * bits ||
      scales.shape() != biases.shape() ||
      scales.shape(0) != weight.shape(0) ||
      scales.shape(1) != x.shape(2) / group_size) {
    throw std::invalid_argument(
        "[spark_qkv_rope] Packed weight and quantization metadata shapes do "
        "not match x, q_size, kv_size, group_size, and bits.");
  }

  auto stream = to_stream(s);
  auto qkv = quantized_matmul(
      x, weight, scales, biases, true, group_size, bits, "affine", stream);
  auto parts = split(qkv, {q_size, q_size + kv_size}, -1, stream);
  auto layout = [&](const array& part, int heads) {
    return contiguous(
        transpose(
            reshape(part, {1, 1, heads, head_dim}, stream),
            {0, 2, 1, 3},
            stream),
        false,
        stream);
  };
  auto q = layout(parts[0], num_heads);
  auto k = layout(parts[1], num_key_value_heads);
  auto v = layout(parts[2], num_key_value_heads);
  return {
      rope(q, rope_dims, false, rope_base, 1.0f, offset, std::nullopt, stream),
      rope(k, rope_dims, false, rope_base, 1.0f, offset, std::nullopt, stream),
      v};
}

} // namespace spark_mlx_ext
