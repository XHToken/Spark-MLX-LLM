"""Conditional immutable bias cache lookup and BF16 Markov GEMV fallback."""

HEADER = r"""
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;
"""

SOURCE = r"""
constexpr uint V = $V, N = $N, R = $R, C = $C;
constexpr uint KL = 8;
const uint batch = threadgroup_position_in_grid.y;
const int token = int(ids[batch]);
const uint tid = thread_position_in_grid.x;
if (token < 0 || token >= int(V)) {
  if (tid < N) out[size_t(batch) * N + tid] = bfloat16_t(NAN);
  return;
}
const int entry = lookup[token];
if (entry >= 0 && entry < int(C)) {
  // All lanes in the batch take the same branch. No fallback projection is
  // evaluated on a hit; the extra GEMV-layout threads simply return.
  if (tid < N)
    out[size_t(batch) * N + tid] = cache[size_t(entry) * N + tid];
  return;
}
const uint row = tid / KL;
if (row >= N) return;
const uint klane = thread_index_in_simdgroup % KL;
float result = 0.0f;
for (uint col = klane * 4; col < R; col += KL * 4) {
  const size_t a = size_t(token) * R + col;
  const size_t w = size_t(row) * R + col;
  result += dot(float4(embedding[a],embedding[a+1],embedding[a+2],embedding[a+3]),
      float4(projection[w],projection[w+1],projection[w+2],projection[w+3]));
}
for (uint shift=KL/2; shift>0; shift/=2)
  result += simd_shuffle_down(result,shift);
if (klane==0) out[size_t(batch)*N+row]=bfloat16_t(result);
"""
