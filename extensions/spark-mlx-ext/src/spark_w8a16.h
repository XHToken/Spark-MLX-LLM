// Copyright (c) 2026 Spark-MLX-LLM contributors
#pragma once

#include "mlx/array.h"
#include "mlx/utils.h"

namespace spark_mlx_ext {

// Single-token signed INT8 matvec with one FP32 scale per output channel:
// y[n] = T(sum_k(float(x[k]) * float(weight[n, k])) * scales[n]).
// x is BF16/FP16, weight is INT8 [N, K], and scales is FP32 [N] or [N, 1].
// x may have any number of leading singleton dimensions; the output preserves
// them and replaces its final dimension with N. This operation is GPU-only.
mlx::core::array w8a16_per_channel(
    const mlx::core::array& x,
    const mlx::core::array& weight,
    const mlx::core::array& scales,
    mlx::core::StreamOrDevice s = {});

} // namespace spark_mlx_ext
