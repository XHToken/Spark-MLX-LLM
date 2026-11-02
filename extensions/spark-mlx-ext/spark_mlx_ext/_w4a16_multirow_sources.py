"""Metal sources for affine W4/G32 multirow verification projections.

The wide kernel follows the lane grouping and affine decode in MLX 0.32.4
quantized.h (Apple Inc., MIT); the eight-row schedule is local. The matrix
kernel reconstructs FP32 weights directly in matrix fragments, preserving
the checkpoint's BF16 scale/bias values without rounding decoded weights.
"""

HEADER = r"""
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>
using namespace metal;
"""

WIDE = r"""
constexpr int M = $M, N = $N, K = $K;
constexpr int KL = $KL, SG = $SG;
const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;
const int klane = lane % KL;
const int row = int(threadgroup_position_in_grid.x) * (32 / KL * SG)
    + int(sg) * (32 / KL) + int(lane) / KL;
if (row >= N) return;
const auto wrow = weight + size_t(row) * (K / 8);
const auto srow = scales + size_t(row) * (K / 32);
const auto brow = biases + size_t(row) * (K / 32);
float result[M] = {0.0f};
for (int g = klane; g < K / 32; g += KL) {
    const float scale = float(srow[g]);
    const float bias = float(brow[g]);
    #pragma unroll
    for (int sub = 0; sub < 4; ++sub) {
        const int k0 = g * 32 + sub * 8;
        const uint code = wrow[g * 4 + sub];
        float decoded[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j)
            decoded[j] = float((code >> (4 * j)) & 15) * scale + bias;
        #pragma unroll
        for (int m = 0; m < M; ++m) {
            float value = 0.0f;
            #pragma unroll
            for (int j = 0; j < 8; ++j)
                value += float(x[m * K + k0 + j]) * decoded[j];
            result[m] += value;
        }
    }
}
#pragma unroll
for (int m = 0; m < M; ++m) {
    #pragma unroll
    for (int shift = KL / 2; shift >= 1; shift /= 2)
        result[m] += simd_shuffle_down(result[m], shift);
    if (klane == 0) out[m * N + row] = bfloat16_t(result[m]);
}
"""

MMA = r"""
constexpr int M = $M, N = $N, K = $K, SG = $SG, SPLITS = $SPLITS;
using OT = $OT;
constexpr bool INTERLEAVED = $INTERLEAVED;
const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;
const uint tile = threadgroup_position_in_grid.x;
const uint split = threadgroup_position_in_grid.y;
const uint fm = ((lane / 4) & 4) + ((lane / 2) % 4);
const uint fn = ((lane / 4) & 2) * 2 + (lane % 2) * 2;
const uint n0 = (tile * SG + sg) * 8;
simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8>(0.0f);
// Split only at affine group boundaries. K is divisible by 32.
for (uint group = split; group < K / 32; group += SPLITS) {
    #pragma unroll
    for (uint sub = 0; sub < 4; ++sub) {
        const uint k0 = group * 32 + sub * 8;
        simdgroup_float8x8 a, b;
        a.thread_elements()[0] = fm < M ? float(x[fm * K + k0 + fn]) : 0.0f;
        a.thread_elements()[1] = fm < M ? float(x[fm * K + k0 + fn + 1]) : 0.0f;
        const uint nr0 = min(n0 + fn, uint(N - 1));
        const uint nr1 = min(n0 + fn + 1, uint(N - 1));
        const size_t w0 = INTERLEAVED ? (size_t(nr0 / 8) * (K / 8) + k0 / 8) * 8 + nr0 % 8 : size_t(nr0) * (K / 8) + k0 / 8;
        const size_t w1 = INTERLEAVED ? (size_t(nr1 / 8) * (K / 8) + k0 / 8) * 8 + nr1 % 8 : size_t(nr1) * (K / 8) + k0 / 8;
        const size_t sidx0 = INTERLEAVED ? (size_t(nr0 / 8) * (K / 32) + group) * 8 + nr0 % 8 : size_t(nr0) * (K / 32) + group;
        const size_t sidx1 = INTERLEAVED ? (size_t(nr1 / 8) * (K / 32) + group) * 8 + nr1 % 8 : size_t(nr1) * (K / 32) + group;
        const uint code0 = weight[w0];
        const uint code1 = weight[w1];
        const float s0 = float(scales[sidx0]);
        const float s1 = float(scales[sidx1]);
        const float b0 = float(biases[sidx0]);
        const float b1 = float(biases[sidx1]);
        b.thread_elements()[0] = float((code0 >> (4 * fm)) & 15) * s0 + b0;
        b.thread_elements()[1] = float((code1 >> (4 * fm)) & 15) * s1 + b1;
        simdgroup_multiply_accumulate(acc, a, b, acc);
    }
}
if (fm < M) {
    if (n0 + fn < N)
        partials[(size_t(split) * M + fm) * N + n0 + fn] = OT(acc.thread_elements()[0]);
    if (n0 + fn + 1 < N)
        partials[(size_t(split) * M + fm) * N + n0 + fn + 1] = OT(acc.thread_elements()[1]);
}
"""

MERGE = r"""
constexpr uint SIZE = $SIZE, SPLITS = $SPLITS;
const uint i = thread_position_in_grid.x;
if (i >= SIZE) return;
float value = partials[i];
#pragma unroll
for (uint s = 1; s < SPLITS; ++s) value += partials[s * SIZE + i];
out[i] = bfloat16_t(value);
"""

P2 = r"""
constexpr int M = $M, N = $N, K = $K, KL = $KL, SG = $SG;
const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;
const int klane = lane % KL;
const int row = int(threadgroup_position_in_grid.x) * (32 / KL * SG)
    + int(sg) * (32 / KL) + int(lane) / KL;
if (row >= N) return;
float result[M] = {0.0f};
for (int col = klane * 8; col < K; col += KL * 8) {
    const uint code = weight[size_t(row) * (K / 8) + col / 8];
    const float scale = float(scales[size_t(row) * (K / 32) + col / 32]);
    const float bias = float(biases[size_t(row) * (K / 32) + col / 32]);
    #pragma unroll
    for (int m = 0; m < M; ++m) {
        float dot_value = 0.0f, sum_value = 0.0f;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float value = float(x[m * K + col + j]);
            sum_value += value;
            dot_value += (value / float(1u << (4 * j))) * float(code & (15u << (4 * j)));
        }
        result[m] += scale * dot_value + bias * sum_value;
    }
}
#pragma unroll
for (int m = 0; m < M; ++m) {
    #pragma unroll
    for (int shift = KL / 2; shift >= 1; shift /= 2)
        result[m] += simd_shuffle_down(result[m], shift);
    if (klane == 0) out[m * N + row] = bfloat16_t(result[m]);
}
"""

MMA2 = r"""
constexpr int M = $M, N = $N, K = $K, SG = $SG, SPLITS = $SPLITS;
using OT = $OT;
constexpr bool INTERLEAVED = $INTERLEAVED;
const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;
const uint tile = threadgroup_position_in_grid.x;
const uint split = threadgroup_position_in_grid.y;
const uint fm = ((lane / 4) & 4) + ((lane / 2) % 4);
const uint fn = ((lane / 4) & 2) * 2 + (lane % 2) * 2;
const uint n0 = (tile * SG + sg) * 8;
simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8>(0.0f);
for (uint group = split; group < K / 32; group += SPLITS) {
    #pragma unroll
    for (uint sub = 0; sub < 4; ++sub) {
        const uint k0 = group * 32 + sub * 8;
        simdgroup_matrix<bfloat16_t, 8, 8> a, high, low;
        a.thread_elements()[0] = fm < M ? x[fm * K + k0 + fn] : bfloat16_t(0);
        a.thread_elements()[1] = fm < M ? x[fm * K + k0 + fn + 1] : bfloat16_t(0);
        const uint nr0 = min(n0 + fn, uint(N - 1));
        const uint nr1 = min(n0 + fn + 1, uint(N - 1));
        const size_t w0 = INTERLEAVED ? (size_t(nr0 / 8) * (K / 8) + k0 / 8) * 8 + nr0 % 8 : size_t(nr0) * (K / 8) + k0 / 8;
        const size_t w1 = INTERLEAVED ? (size_t(nr1 / 8) * (K / 8) + k0 / 8) * 8 + nr1 % 8 : size_t(nr1) * (K / 8) + k0 / 8;
        const size_t sidx0 = INTERLEAVED ? (size_t(nr0 / 8) * (K / 32) + group) * 8 + nr0 % 8 : size_t(nr0) * (K / 32) + group;
        const size_t sidx1 = INTERLEAVED ? (size_t(nr1 / 8) * (K / 32) + group) * 8 + nr1 % 8 : size_t(nr1) * (K / 32) + group;
        const uint code0 = weight[w0];
        const uint code1 = weight[w1];
        const float s0 = float(scales[sidx0]);
        const float s1 = float(scales[sidx1]);
        const float b0 = float(biases[sidx0]);
        const float b1 = float(biases[sidx1]);
        const float v0 = float((code0 >> (4 * fm)) & 15) * s0 + b0;
        const float v1 = float((code1 >> (4 * fm)) & 15) * s1 + b1;
        high.thread_elements()[0] = bfloat16_t(v0);
        high.thread_elements()[1] = bfloat16_t(v1);
        low.thread_elements()[0] = bfloat16_t(v0 - float(high.thread_elements()[0]));
        low.thread_elements()[1] = bfloat16_t(v1 - float(high.thread_elements()[1]));
        simdgroup_multiply_accumulate(acc, a, high, acc);
        simdgroup_multiply_accumulate(acc, a, low, acc);
    }
}
if (fm < M) {
    if (n0 + fn < N)
        partials[(size_t(split) * M + fm) * N + n0 + fn] = OT(acc.thread_elements()[0]);
    if (n0 + fn + 1 < N)
        partials[(size_t(split) * M + fm) * N + n0 + fn + 1] = OT(acc.thread_elements()[1]);
}
"""

NAX_HEADER = (
    HEADER
    + r"""
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
"""
)

NAX = r"""
constexpr int M = $M, N = $N, K = $K, SG = $SG, SPLITS = $SPLITS;
constexpr bool INTERLEAVED = $INTERLEAVED, TWO_PART = $TWO_PART;
using BType = $BTYPE;
using OT = $OT;
const uint sg = simdgroup_index_in_threadgroup;
const uint tile = threadgroup_position_in_grid.x;
const uint split = threadgroup_position_in_grid.y;
const uint n0 = (tile * SG + sg) * 32;
if (n0 >= N) return;
constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 16, false, false, false,
    mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
auto a = op.get_left_input_cooperative_tensor<bfloat16_t, BType, float>();
auto b = op.get_right_input_cooperative_tensor<bfloat16_t, BType, float>();
auto low = op.get_right_input_cooperative_tensor<bfloat16_t, BType, float>();
auto c = op.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(a)>,
    metal::remove_addrspace_t<decltype(b)>, float>();
#pragma unroll
for (uint i = 0; i < c.get_capacity(); ++i) c[i] = 0.0f;
for (uint group = split; group < K / 32; group += SPLITS) {
    #pragma unroll
    for (uint sub = 0; sub < 2; ++sub) {
        const uint k0 = group * 32 + sub * 16;
        #pragma unroll
        for (uint i = 0; i < a.get_capacity(); ++i) {
            {
                const auto ix = a.get_multidimensional_index(i);
                a[i] = ix[1] < M ? x[uint(ix[1]) * K + k0 + uint(ix[0])] : bfloat16_t(0);
            }
        }
        #pragma unroll
        for (uint i = 0; i < b.get_capacity(); ++i) {
            {
                const auto ix = b.get_multidimensional_index(i);
                const uint nr = min(n0 + uint(ix[0]), uint(N - 1));
                const uint kk = k0 + uint(ix[1]);
                const size_t wi = INTERLEAVED ? (size_t(nr / 8) * (K / 8) + kk / 8) * 8 + nr % 8 : size_t(nr) * (K / 8) + kk / 8;
                const size_t si = INTERLEAVED ? (size_t(nr / 8) * (K / 32) + group) * 8 + nr % 8 : size_t(nr) * (K / 32) + group;
                const float value = float((weight[wi] >> ((kk % 8) * 4)) & 15) * float(scales[si]) + float(biases[si]);
                b[i] = BType(value);
                if constexpr (TWO_PART) low[i] = BType(value - float(b[i]));
            }
        }
        op.run(a, b, c);
        if constexpr (TWO_PART) op.run(a, low, c);
    }
}
#pragma unroll
for (uint i = 0; i < c.get_capacity(); ++i) {
    {
        const auto ix = c.get_multidimensional_index(i);
        if (ix[1] < M && n0 + uint(ix[0]) < N)
            partials[(size_t(split) * M + uint(ix[1])) * N + n0 + uint(ix[0])] = OT(c[i]);
    }
}
"""

NAX4 = r"""
constexpr int M = $M, N = $N, K = $K, SG = $SG, SPLITS = $SPLITS;
constexpr bool TILED = $TILED;
constexpr bool STATIC_LAYOUT = $STATIC_LAYOUT;
constexpr bool MULTIPLY_MODE = $MULTIPLY_MODE;
using OT = $OT;
const uint sg = simdgroup_index_in_threadgroup;
const uint tile = threadgroup_position_in_grid.x;
const uint split = threadgroup_position_in_grid.y;
const uint n0 = (tile * SG + sg) * 32;
if (n0 >= N) return;
const uint lane = thread_index_in_simdgroup;
const uint fm = ((lane / 4) & 4) + ((lane / 2) % 4);
const uint fn = (((lane / 4) & 2) | (lane & 1)) * 4;
constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(8, 32, 32, false, !TILED, false,
    mpp::tensor_ops::matmul2d_descriptor::mode::$MODE);
mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
auto a = op.get_left_input_cooperative_tensor<bfloat16_t, metal::uint4b_format, float>();
using BType = metal::tensor<device metal::uint4b_format, metal::dextents<int, 2>, metal::tensor_inline>;
const BType all_weights(reinterpret_cast<device uchar*>(const_cast<device uint*>(weight)),
    metal::dextents<int, 2>(K, N));
auto c = op.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(a)>, BType, float>();
float accumulated[16] = {0.0f};
// Layout queries take mutable cooperative-storage pointers. Query only once,
// before the reduction loop, so the compiler need not prove them invariant.
uint a_index[16], out_index[16], metadata_base[16], sum_base[16];
const uint a_count = STATIC_LAYOUT ? 16 : a.get_capacity(), c_count = STATIC_LAYOUT ? 16 : c.get_capacity();
#pragma unroll
for (uint i = 0; i < a_count; ++i) {
    const auto ix = STATIC_LAYOUT ? metal::array<int,2>{int(fn + i % 4 + (i / 8) * 16),int(fm + ((i % 8) / 4) * 8)} : a.get_multidimensional_index(i);
    a_index[i] = ix[1] < M ? uint(ix[1]) * K + uint(ix[0]) : 0xffffffffu;
}
#pragma unroll
for (uint i = 0; i < c_count; ++i) {
    const auto ix = STATIC_LAYOUT ? metal::array<int,2>{int(fn + i % 4 + (i / 8) * 16),int(fm + ((i % 8) / 4) * 8)} : c.get_multidimensional_index(i);
    out_index[i] = ix[1] < M ? uint(ix[1]) * N + n0 + uint(ix[0]) : 0xffffffffu;
    metadata_base[i] = TILED ? (n0 / 256) * (K / 32) * 256 + n0 % 256 + uint(ix[0]) : (n0 + uint(ix[0])) * (K / 32);
    sum_base[i] = uint(ix[1]) * (K / 32);
}
for (uint group = $GROUP_BEGIN; group < $GROUP_END; group += $GROUP_STEP) {
    #pragma unroll
    for (uint i = 0; i < a_count; ++i) {
        a[i] = a_index[i] != 0xffffffffu ? x[a_index[i] + group * 32] : bfloat16_t(0);
    }
    if constexpr (!MULTIPLY_MODE) {
        #pragma unroll
        for (uint i = 0; i < c_count; ++i) c[i] = 0.0f;
    }
    if constexpr (TILED) {
        using TileType = metal::tensor<device metal::uint4b_format,
            metal::extents<int, 256, 32>, metal::tensor_inline>;
        const TileType block_weights(reinterpret_cast<device uchar*>(const_cast<device uint*>(weight))
            + (size_t(n0 / 256) * (K / 32) + group) * 4096,
            metal::extents<int, 256, 32>());
        const auto block_slice = block_weights.$SLICE(n0 % 256, 0);
        op.run(a, block_slice, c);
    } else {
        const auto b = all_weights.$SLICE(group * 32, n0);
        op.run(a, b, c);
    }
    #pragma unroll
    for (uint i = 0; i < c_count; ++i) {
        if (out_index[i] != 0xffffffffu) {
            const size_t si = metadata_base[i] + group * (TILED ? 256 : 1);
            accumulated[i] += float(scales[si]) * c[i]
                + float(biases[si]) * group_sums[sum_base[i] + group];
        }
    }
}
#pragma unroll
for (uint i = 0; i < c_count; ++i) {
    if (out_index[i] != 0xffffffffu)
        partials[size_t(split) * M * N + out_index[i]] = OT(accumulated[i]);
}
"""

NAX4_PACK = r"""
constexpr uint N = $N, K = $K, WORDS = N * (K / 8);
const uint i = thread_position_in_grid.x;
if (i >= WORDS) return;
const uint tile = i / 1024;
const uint group = tile % (K / 32);
const uint nblock = tile / (K / 32);
const uint within = i % 1024;
const uint klocal = within / 32;
const uint nword = within % 32;
uint result = 0;
#pragma unroll
for (uint j = 0; j < 8; ++j) {
    const uint nr = nblock * 256 + nword * 8 + j;
    const uint code = weight[size_t(nr) * (K / 8) + group * 4 + klocal / 8];
    result |= ((code >> ((klocal % 8) * 4)) & 15) << (j * 4);
}
out[i] = result;
"""

SPLIT_P2 = r"""
constexpr int M = $M, N = $N, K = $K, SPLITS = $SPLITS, ROWS = 4;
const uint split = simdgroup_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
const uint out_row = threadgroup_position_in_grid.x * ROWS;
float results[M * ROWS] = {0.0f};
threadgroup float partials[SPLITS * M * ROWS];
for (uint tile = split; tile < K / 512; tile += SPLITS) {
    const uint col = tile * 512 + lane * 16;
    ushort words[ROWS * 4];
    float scales_r[ROWS], biases_r[ROWS];
    #pragma unroll
    for (uint r = 0; r < ROWS; ++r) {
        const uint row = out_row + r;
        const auto ptr = reinterpret_cast<const device ushort*>(weight)
            + size_t(row) * (K / 4) + col / 4;
        #pragma unroll
        for (uint j = 0; j < 4; ++j) words[r * 4 + j] = ptr[j];
        scales_r[r] = float(scales[size_t(row) * (K / 32) + col / 32]);
        biases_r[r] = float(biases[size_t(row) * (K / 32) + col / 32]);
    }
    #pragma unroll
    for (uint m = 0; m < M; ++m) {
        float xv[16], sum = 0.0f;
        #pragma unroll
        for (uint i = 0; i < 16; i += 4) {
            const float a0 = float(x[m * K + col + i]);
            const float a1 = float(x[m * K + col + i + 1]);
            const float a2 = float(x[m * K + col + i + 2]);
            const float a3 = float(x[m * K + col + i + 3]);
            sum += a0 + a1 + a2 + a3;
            xv[i] = a0; xv[i + 1] = a1 / 16.0f;
            xv[i + 2] = a2 / 256.0f; xv[i + 3] = a3 / 4096.0f;
        }
        #pragma unroll
        for (uint r = 0; r < ROWS; ++r) {
            float dot = 0.0f;
            #pragma unroll
            for (uint j = 0; j < 4; ++j) {
                const ushort word = words[r * 4 + j];
                dot += xv[j * 4] * float(word & 0x000f)
                    + xv[j * 4 + 1] * float(word & 0x00f0)
                    + xv[j * 4 + 2] * float(word & 0x0f00)
                    + xv[j * 4 + 3] * float(word & 0xf000);
            }
            results[m * ROWS + r] += scales_r[r] * dot + biases_r[r] * sum;
        }
    }
}
#pragma unroll
for (uint i = 0; i < M * ROWS; ++i) {
    const float value = simd_sum(results[i]);
    if (lane == 0) partials[split * M * ROWS + i] = value;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (split == 0 && lane < M * ROWS) {
    float value = partials[lane];
    #pragma unroll
    for (uint i = 1; i < SPLITS; ++i) value += partials[i * M * ROWS + lane];
    out[(lane / ROWS) * N + out_row + lane % ROWS] = bfloat16_t(value);
}
"""
