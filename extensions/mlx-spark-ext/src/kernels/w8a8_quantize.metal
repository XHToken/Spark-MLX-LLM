// Copyright (c) 2026 Spark-MLX-LLM contributors

#include <metal_stdlib>
using namespace metal;

template <typename T>
inline void quantize_token(
    const device T* x, device int8_t* q, device float* scales,
    uint K, uint row, uint tid, uint lane, uint sg, uint width,
    threadgroup float* maxima) {
  float amax = 0.0f;
  for (uint k = tid; k < K; k += width) {
    amax = max(amax, abs(float(x[row * K + k])));
  }
  amax = simd_max(amax);
  if (lane == 0) maxima[sg] = amax;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    amax = simd_max(lane < width / 32 ? maxima[lane] : 0.0f);
    if (lane == 0) maxima[0] = amax;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float scale = maxima[0] > 0.0f ? maxima[0] / 127.0f : 1.0f;
  if (tid == 0) scales[row] = scale;
  for (uint k = tid; k < K; k += width) {
    q[row * K + k] = int8_t(clamp(rint(float(x[row * K + k]) / scale), -127.0f, 127.0f));
  }
}

#define QUANTIZE_KERNEL(T, SUFFIX) \
kernel void quantize_per_token_##SUFFIX( \
    const device T* x [[buffer(0)]], device int8_t* q [[buffer(1)]], \
    device float* scales [[buffer(2)]], constant uint& K [[buffer(3)]], \
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]], \
    uint width [[threads_per_threadgroup]]) { \
  threadgroup float maxima[8]; \
  quantize_token(x, q, scales, K, row, tid, lane, sg, width, maxima); \
}

QUANTIZE_KERNEL(half, f16)
QUANTIZE_KERNEL(bfloat, bf16)
QUANTIZE_KERNEL(float, f32)
