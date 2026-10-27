// Copyright (c) 2026 Spark-MLX-LLM contributors
// Signed INT8 weights, BF16/FP16 activation, FP32 per-channel postscale.
#include <metal_stdlib>
using namespace metal;

constant uint W8A16_N [[function_constant(0)]];
constant uint W8A16_K [[function_constant(1)]];
constant bool W8A16_ALIGNED_WEIGHT [[function_constant(2)]];

// The two production shapes are complete R4/G2/P2 tiles. In particular K is
// divisible by 256, so both packs of all 32 SIMD lanes are always in bounds.
// Their N*K also fits uint32; generic shapes below retain size_t addressing.
template <typename T, uint N, uint K>
inline void w8a16_per_channel_full_tiles(
    const device T* x, const device char* weight,
    const device float* scales, device T* out,
    uint group, uint simd, uint lane) {
  static_assert(N % 8 == 0 && K % 256 == 0);
  static_assert(ulong(N) * K <= 0xfffffffful);
  const uint n0 = (group * 2 + simd) * 4;
  float sums[4];
  for (uint r = 0; r < 4; ++r) sums[r] = 0.0f;
  for (uint base = lane * 8; base < K; base += 256) {
    for (uint p = 0; p < 2; ++p) {
      const uint k = base + p * 4;
      const float4 a = float4(x[k], x[k + 1], x[k + 2], x[k + 3]);
      for (uint r = 0; r < 4; ++r) {
        const char4 w = *reinterpret_cast<const device char4*>(
            weight + (n0 + r) * K + k);
        sums[r] += dot(a, float4(w));
      }
    }
  }
  for (uint r = 0; r < 4; ++r) {
    const float sum = simd_sum(sums[r]);
    if (lane == 0) out[n0 + r] = T(sum * scales[n0 + r]);
  }
}

template <typename T, uint Rows, uint Groups, uint Packs>
inline void w8a16_per_channel_matvec(
    const device T* x, const device char* weight,
    const device float* scales, device T* out,
    uint group, uint simd, uint lane) {
  const uint n0 = (group * Groups + simd) * Rows;
  float sums[Rows];
  #pragma unroll
  for (uint r = 0; r < Rows; ++r) sums[r] = 0.0f;
  for (uint base = lane * 4 * Packs; base < W8A16_K; base += 128 * Packs) {
    #pragma unroll
    for (uint p = 0; p < Packs; ++p) {
      const uint k = base + p * 4;
      if (k + 3 < W8A16_K) {
        const float4 a = float4(x[k], x[k + 1], x[k + 2], x[k + 3]);
        #pragma unroll
        for (uint r = 0; r < Rows; ++r) {
          if (n0 + r < W8A16_N) {
            const size_t offset = size_t(n0 + r) * W8A16_K + k;
            if (W8A16_K % 4 == 0 && W8A16_ALIGNED_WEIGHT) {
              const char4 w = *reinterpret_cast<const device char4*>(weight + offset);
              sums[r] += dot(a, float4(w));
            } else {
              // Odd row strides or sliced storage may make char4 unaligned.
              const float4 w = float4(weight[offset], weight[offset + 1],
                  weight[offset + 2], weight[offset + 3]);
              sums[r] += dot(a, w);
            }
          }
        }
      } else if (k < W8A16_K) {
        // General positive K is supported without reading beyond either input.
        #pragma unroll
        for (uint j = 0; j < 4; ++j) {
          if (k + j < W8A16_K) {
            const float a = float(x[k + j]);
            #pragma unroll
            for (uint r = 0; r < Rows; ++r) {
              if (n0 + r < W8A16_N) {
                sums[r] += a * float(weight[size_t(n0 + r) * W8A16_K + k + j]);
              }
            }
          }
        }
      }
    }
  }
  #pragma unroll
  for (uint r = 0; r < Rows; ++r) {
    const float sum = simd_sum(sums[r]);
    if (lane == 0 && n0 + r < W8A16_N) {
      out[n0 + r] = T(sum * scales[n0 + r]);
    }
  }
}

#define W8A16_KERNEL(T, SUFFIX) \
kernel void w8a16_per_channel_##SUFFIX( \
    const device T* x [[buffer(0)]], const device char* weight [[buffer(1)]], \
    const device float* scales [[buffer(2)]], device T* out [[buffer(3)]], \
    uint group [[threadgroup_position_in_grid]], \
    uint simd [[simdgroup_index_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]]) { \
  w8a16_per_channel_matvec<T, 4, 2, 2>(x, weight, scales, out, group, simd, lane); \
}

W8A16_KERNEL(half, f16)
W8A16_KERNEL(bfloat, bf16)

#define W8A16_FULL_TILE_KERNEL(T, SUFFIX, N, K) \
kernel void w8a16_per_channel_n##N##_k##K##_##SUFFIX( \
    const device T* x [[buffer(0)]], const device char* weight [[buffer(1)]], \
    const device float* scales [[buffer(2)]], device T* out [[buffer(3)]], \
    uint group [[threadgroup_position_in_grid]], \
    uint simd [[simdgroup_index_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]]) { \
  w8a16_per_channel_full_tiles<T, N, K>(x, weight, scales, out, group, simd, lane); \
}

W8A16_FULL_TILE_KERNEL(half, f16, 2560, 4096)
W8A16_FULL_TILE_KERNEL(bfloat, bf16, 2560, 4096)
W8A16_FULL_TILE_KERNEL(half, f16, 131072, 2560)
W8A16_FULL_TILE_KERNEL(bfloat, bf16, 131072, 2560)
