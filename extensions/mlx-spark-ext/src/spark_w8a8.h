// Copyright (c) 2026 Spark-MLX-LLM contributors
#pragma once

#include <optional>
#include "mlx/array.h"
#include "mlx/utils.h"

namespace mlx_spark_ext {

bool is_w8a8_available();

mlx::core::array int8_matmul_int32(
    const mlx::core::array& x,
    const mlx::core::array& weight,
    mlx::core::StreamOrDevice s = {});

mlx::core::array w8a8_linear(
    const mlx::core::array& x,
    const mlx::core::array& weight,
    const mlx::core::array& scales,
    const std::optional<mlx::core::array>& bias = std::nullopt,
    mlx::core::StreamOrDevice s = {});

} // namespace mlx_spark_ext
