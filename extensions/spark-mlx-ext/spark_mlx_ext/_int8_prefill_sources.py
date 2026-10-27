"""Metal sources for int8_quantized_sdpa_prefill."""

_HEADER = r"""
#include <metal_simdgroup_matrix>
"""

_PREFILL = r"""
constexpr uint D = 256, BQ = 32;
constexpr uint BK = sizeof(T) == 4 ? 16 : 32;
const uint tid = thread_position_in_threadgroup.x;
const uint sg = tid / 32;
const uint lane = thread_index_in_simdgroup;
const uint tile = threadgroup_position_in_grid.x;
const uint kh = threadgroup_position_in_grid.y;
const uint batch = threadgroup_position_in_grid.z;
const uint L = queries_shape[2];
const uint N = keys_shape[2];
const uint HQ = queries_shape[1];
const uint row_start = tile * BQ;

// Q stays in matrix registers; this buffer alternates between decoded K/V.
threadgroup T kv[BK * D];
threadgroup float scores[BQ * BK];
threadgroup T probabilities[BQ * BK];
threadgroup float maxima[BQ], normalizers[BQ], factors[BQ];

// The 8x8 fragment lane mapping follows MLX steel/attn/mma.h.
const uint fm = ((lane / 4) & 4) + ((lane / 2) % 4);
const uint fn = ((lane / 4) & 2) * 2 + (lane % 2) * 2;
const uint row = sg * 8 + fm;
const uint q = (row_start + row) / G;
const uint h = kh * G + (row_start + row) % G;
simdgroup_matrix<T, 8, 8> q_frag[D / 8];
#pragma unroll
for (uint d = 0; d < D / 8; ++d) {
    const long offset = long(batch) * queries_strides[0] + long(h) * queries_strides[1] +
        long(q) * queries_strides[2] + long(d * 8 + fn) * queries_strides[3];
    q_frag[d].thread_elements()[0] = q < L ? T(float(queries[offset]) * float(scale)) : T(0);
    q_frag[d].thread_elements()[1] = q < L ? T(float(queries[offset + queries_strides[3]]) * float(scale)) : T(0);
}
if (tid < BQ) {
    maxima[tid] = -MAXFLOAT;
    normalizers[tid] = 0.f;
}
// Each of four SIMD groups owns one eight-query, full-width output tile.
simdgroup_float8x8 output_frag[D / 8];
#pragma unroll
for (uint d = 0; d < D / 8; ++d)
    output_frag[d] = make_filled_simdgroup_matrix<float, 8>(0.f);

const uint last_q = min(L - 1, (row_start + BQ - 1) / G);
const uint end = CAUSAL ? N - L + last_q + 1 : N;
for (uint start = 0; start < end; start += BK) {
    // Each packed word is fetched once and decoded for all thirty-two Q rows.
    for (uint i = tid; i < BK * (D / 4); i += 128) {
        const uint t = start + i / (D / 4), w = i % (D / 4);
        uint code = 0;
        float s = 0.f, b = 0.f;
        if (t < end) {
            code = keys[long(batch) * keys_strides[0] + long(kh) * keys_strides[1] +
                long(t) * keys_strides[2] + long(w) * keys_strides[3]];
            s = float(key_scales[long(batch) * key_scales_strides[0] +
                long(kh) * key_scales_strides[1] + long(t) * key_scales_strides[2] +
                long(w / 8) * key_scales_strides[3]]);
            b = float(key_biases[long(batch) * key_biases_strides[0] +
                long(kh) * key_biases_strides[1] + long(t) * key_biases_strides[2] +
                long(w / 8) * key_biases_strides[3]]);
        }
        for (uint j = 0; j < 4; ++j)
            kv[(i / (D / 4)) * D + w * 4 + j] = T(float((code >> (8 * j)) & 255) * s + b);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma unroll
    for (uint k = 0; k < BK / 8; ++k) {
        simdgroup_float8x8 score = make_filled_simdgroup_matrix<float, 8>(0.f);
        #pragma unroll
        for (uint d = 0; d < D / 8; ++d) {
                simdgroup_matrix<T, 8, 8> key;
            simdgroup_load(key, kv + k * 8 * D + d * 8, D, ulong2(0), true);
            simdgroup_multiply_accumulate(score, q_frag[d], key, score);
        }
        simdgroup_store(score, scores + sg * 8 * BK + k * 8, BK);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < BQ) {
        const uint q = (row_start + tid) / G;
        const uint limit = CAUSAL ? N - L + q + 1 : N;
        float next_max = maxima[tid];
        for (uint k = 0; k < BK; ++k)
            if (q < L && start + k < limit && start + k < end)
                next_max = max(next_max, scores[tid * BK + k]);
        const float alpha = fast::exp(maxima[tid] - next_max);
        float total = 0.f;
        for (uint k = 0; k < BK; ++k) {
            const bool valid = q < L && start + k < limit && start + k < end;
            const float p = valid ? fast::exp(scores[tid * BK + k] - next_max) : 0.f;
            probabilities[tid * BK + k] = T(p);
            total += p;
        }
        normalizers[tid] = normalizers[tid] * alpha + total;
        factors[tid] = alpha;
        maxima[tid] = next_max;
    }
    // All QK users finish before replacing the shared K tile with V.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < BK * (D / 4); i += 128) {
        const uint t = start + i / (D / 4), w = i % (D / 4);
        uint code = 0;
        float s = 0.f, b = 0.f;
        if (t < end) {
            code = values[long(batch) * values_strides[0] + long(kh) * values_strides[1] +
                long(t) * values_strides[2] + long(w) * values_strides[3]];
            s = float(value_scales[long(batch) * value_scales_strides[0] +
                long(kh) * value_scales_strides[1] + long(t) * value_scales_strides[2] +
                long(w / 8) * value_scales_strides[3]]);
            b = float(value_biases[long(batch) * value_biases_strides[0] +
                long(kh) * value_biases_strides[1] + long(t) * value_biases_strides[2] +
                long(w / 8) * value_biases_strides[3]]);
        }
        for (uint j = 0; j < 4; ++j)
            kv[(i / (D / 4)) * D + w * 4 + j] = T(float((code >> (8 * j)) & 255) * s + b);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    #pragma unroll
    for (uint d = 0; d < D / 8; ++d) {
        output_frag[d].thread_elements()[0] *= factors[row];
        output_frag[d].thread_elements()[1] *= factors[row];
    }
    #pragma unroll
    for (uint k = 0; k < BK / 8; ++k) {
        simdgroup_matrix<T, 8, 8> probability;
        simdgroup_load(probability, probabilities + sg * 8 * BK + k * 8, BK);
        #pragma unroll
        for (uint d = 0; d < D / 8; ++d) {
            simdgroup_matrix<T, 8, 8> value;
            simdgroup_load(value, kv + k * 8 * D + d * 8, D);
            simdgroup_multiply_accumulate(output_frag[d], probability, value, output_frag[d]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

if (q < L) {
    #pragma unroll
    for (uint d = 0; d < D / 8; ++d) {
        const uint col = d * 8 + fn;
        const long index = ((long(batch) * HQ + h) * L + q) * D + col;
        out[index] = T(output_frag[d].thread_elements()[0] / normalizers[row]);
        out[index + 1] = T(output_frag[d].thread_elements()[1] / normalizers[row]);
    }
}
"""
