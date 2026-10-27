// Copyright © 2026 Apple Inc.

#pragma once

#include <vector>
#include "mlx/array.h"
#include "mlx/utils.h"

namespace spark_mlx_ext {

/** Affine quantized QKV projection and non-traditional Q/K RoPE for one token. */
std::vector<mlx::core::array> spark_qkv_rope(
    const mlx::core::array& x,
    const mlx::core::array& weight,
    const mlx::core::array& scales,
    const mlx::core::array& biases,
    int q_size,
    int kv_size,
    int num_heads,
    int num_key_value_heads,
    int head_dim,
    int rope_dims,
    float rope_base,
    int offset,
    int group_size = 128,
    int bits = 4,
    mlx::core::StreamOrDevice s = {});

} // namespace spark_mlx_ext
