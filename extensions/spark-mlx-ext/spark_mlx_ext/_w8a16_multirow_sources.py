"""Metal sources for signed-INT8 per-channel multirow projection.

The fragment ownership formulas follow Metal's simdgroup 8x8 layout. No
activation quantization or scale rounding is performed by these kernels.
"""

HEADER = r"""
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;
"""


WIDE = r"""
constexpr int K = $K;
constexpr int N = $N;
constexpr int M = $M;
constexpr int TM = $TM;
constexpr int KL = $KL;
constexpr int SG = $SG;
constexpr int RowsPerSG = 32 / KL;
const uint lane = thread_index_in_simdgroup;
const uint group = simdgroup_index_in_threadgroup;
const int klane = int(lane) % KL;
const int row = int(threadgroup_position_in_grid.y) * (RowsPerSG * SG)
    + int(group) * RowsPerSG + int(lane) / KL;
const int first_m = int(threadgroup_position_in_grid.x) * TM;
if (row >= N) return;
thread float result[TM] = {0.0f};
for (int col = klane * 4; col < K; col += KL * 4) {
  const float4 w = float4(weight[size_t(row) * K + col],
      weight[size_t(row) * K + col + 1],
      weight[size_t(row) * K + col + 2],
      weight[size_t(row) * K + col + 3]);
  for (int m = 0; m < TM; ++m) {
    if (first_m + m < M) {
      const size_t ix = size_t(first_m + m) * K + col;
      const float4 a = float4(x[ix], x[ix + 1], x[ix + 2], x[ix + 3]);
      result[m] += dot(a, w);
    }
  }
}
for (int m = 0; m < TM; ++m) {
  float value = result[m];
  for (int offset = KL / 2; offset > 0; offset /= 2)
    value += simd_shuffle_down(value, offset);
  if (klane == 0 && first_m + m < M)
    out[size_t(first_m + m) * N + row] = OT(value * scales[row]);
}
"""


MATRIX = r"""
constexpr int K = $K;
constexpr int N = $N;
constexpr int M = $M;
constexpr int SG = $SG;
const uint lane = thread_index_in_simdgroup;
const int output_col = (int(threadgroup_position_in_grid.y) * SG
    + int(simdgroup_index_in_threadgroup)) * 8;
const int fragment_row = ((int(lane) / 4) & 4) + ((int(lane) / 2) % 4);
const int fragment_col = ((int(lane) / 4) & 2) * 2 + (int(lane) % 2) * 2;
simdgroup_float8x8 accum = make_filled_simdgroup_matrix<float, 8>(0.0f);
for (int col = 0; col < K; col += 8) {
  simdgroup_float8x8 a;
  simdgroup_float8x8 b;
  a.thread_elements()[0] = fragment_row < M
      ? float(x[size_t(fragment_row) * K + col + fragment_col]) : 0.0f;
  a.thread_elements()[1] = fragment_row < M
      ? float(x[size_t(fragment_row) * K + col + fragment_col + 1]) : 0.0f;
  b.thread_elements()[0] = output_col + fragment_col < N
      ? float(weight[size_t(output_col + fragment_col) * K + col + fragment_row]) : 0.0f;
  b.thread_elements()[1] = output_col + fragment_col + 1 < N
      ? float(weight[size_t(output_col + fragment_col + 1) * K + col + fragment_row]) : 0.0f;
  simdgroup_multiply_accumulate(accum, a, b, accum);
}
if (fragment_row < M && output_col + fragment_col < N)
  out[size_t(fragment_row) * N + output_col + fragment_col] =
      OT(accum.thread_elements()[0] * scales[output_col + fragment_col]);
if (fragment_row < M && output_col + fragment_col + 1 < N)
  out[size_t(fragment_row) * N + output_col + fragment_col + 1] =
      OT(accum.thread_elements()[1] * scales[output_col + fragment_col + 1]);
"""

NAX_HEADER = HEADER + r"""
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
"""

NAX = r"""
constexpr int K = $K, N = $N, M = $M, SG = $SG;
const uint sg = simdgroup_index_in_threadgroup;
const uint n0 = (threadgroup_position_in_grid.y * SG + sg) * 32;
if (n0 >= N) return;
constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(8, 32, 64, false, true, false,
    mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
auto a = op.get_left_input_cooperative_tensor<bfloat16_t, int8_t, float>();
using BType = metal::tensor<device int8_t, metal::dextents<int, 2>, metal::tensor_inline>;
const BType all_weights(const_cast<device int8_t*>(weight), metal::dextents<int, 2>(K, N));
auto c = op.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(a)>, BType, float>();
uint a_index[32], out_index[16], scale_index[16];
const uint ac = a.get_capacity(), cc = c.get_capacity();
#pragma unroll
for (uint i = 0; i < ac; ++i) {
    const auto ix = a.get_multidimensional_index(i);
    a_index[i] = ix[1] < M ? uint(ix[1]) * K + uint(ix[0]) : 0xffffffffu;
}
#pragma unroll
for (uint i = 0; i < cc; ++i) {
    const auto ix = c.get_multidimensional_index(i);
    out_index[i] = ix[1] < M ? uint(ix[1]) * N + n0 + uint(ix[0]) : 0xffffffffu;
    scale_index[i] = n0 + uint(ix[0]);
    c[i] = 0.0f;
}
for (uint offset = 0; offset < K; offset += 64) {
    #pragma unroll
    for (uint i = 0; i < ac; ++i)
        a[i] = a_index[i] != 0xffffffffu ? x[a_index[i] + offset] : bfloat16_t(0);
    const auto b = all_weights.slice(offset, n0);
    op.run(a, b, c);
}
#pragma unroll
for (uint i = 0; i < cc; ++i)
    if (out_index[i] != 0xffffffffu)
        out[out_index[i]] = OT(c[i] * scales[scale_index[i]]);
"""
