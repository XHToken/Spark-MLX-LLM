"""Metal sources for w4a16_qmv."""

_HEADER = r"""
// Copyright © 2023-2024 Apple Inc.
// W4/P2 expressions adapted from MLX 0.32.4 quantized.h under the MIT license.
// Split-K scheduling and merging are local changes.
#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;

inline float spark_load_vector_w4_p2(
    const device bfloat16_t* x, thread float* x_thread) {
  float sum = 0;
  for (int i = 0; i < 16; i += 4) {
    sum += float(x[i]) + float(x[i + 1]) + float(x[i + 2]) + float(x[i + 3]);
    x_thread[i] = x[i];
    x_thread[i + 1] = x[i + 1] / 16.0f;
    x_thread[i + 2] = x[i + 2] / 256.0f;
    x_thread[i + 3] = x[i + 3] / 4096.0f;
  }
  return sum;
}

inline float spark_qdot_w4_p2(
    const device uint8_t* w,
    const thread float* x_thread,
    float scale,
    float bias,
    float sum) {
  float accum = 0;
  const device uint16_t* ws = (const device uint16_t*)w;
  for (int i = 0; i < 4; i++) {
    accum +=
        (x_thread[4 * i] * (ws[i] & 0x000f) +
         x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
         x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
         x_thread[4 * i + 3] * (ws[i] & 0xf000));
  }
  return scale * accum + sum * bias;
}
"""

_BODY = r"""
constexpr int K = $K;
constexpr int Rows = 4;
constexpr int Splits = 8;
constexpr int Values = 16;
const uint split = simdgroup_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
const int out_row = int(threadgroup_position_in_grid.y) * Rows;
thread float result[Rows] = {0.0f};
thread float x_thread[Values];
threadgroup float partials[Splits * Rows];

for (int tile = int(split); tile < K / 512; tile += Splits) {
  const int col = tile * 512 + int(lane) * Values;
  const float sum = spark_load_vector_w4_p2(x + col, x_thread);
  for (int row = 0; row < Rows; ++row) {
    const size_t metadata_offset = size_t(out_row + row) * (K / 32) + col / 32;
    const device uint8_t* wl = reinterpret_cast<const device uint8_t*>(weight)
        + size_t(out_row + row) * (K / 2) + col / 2;
    const float scale = scales[metadata_offset];
    const float bias = biases[metadata_offset];
    result[row] += spark_qdot_w4_p2(wl, x_thread, scale, bias, sum);
  }
}

for (int row = 0; row < Rows; ++row) {
  const float value = simd_sum(result[row]);
  if (lane == 0) partials[int(split) * Rows + row] = value;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

if (split == 0 && lane < Rows) {
  float value = partials[lane];
  for (int part = 1; part < Splits; ++part) {
    value += partials[part * Rows + lane];
  }
  out[out_row + lane] = bfloat16_t(value);
}
"""
