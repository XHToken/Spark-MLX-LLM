"""Metal sources for bf16_sdpa."""

_HEADER = r"""
// Derived from MLX v0.32.4 sdpa_vector.h, Copyright © 2024 Apple Inc., MIT.
// Specialized D=V=256, GQA=4, QLEN=1, TK=4.
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;
"""

_PASS1_BASE = r"""
constexpr int D = 256;
constexpr int BD = 32;
constexpr int TK = 4;
constexpr int blocks = $SPLITS;
constexpr int qk_per_thread = D / BD;
constexpr int v_per_thread = D / BD;
const uint3 tid = threadgroup_position_in_grid;
const uint3 tidtg = thread_position_in_threadgroup;
const uint simd_lid = thread_index_in_simdgroup;
const int kv_head_idx = tid.x;
const int batch_idx = tid.y;
const int block_idx = tid.z;
const int q_head_idx = 4 * kv_head_idx + tidtg.y;
const int num_q_heads = KH * 4;
const int o_offset = batch_idx * num_q_heads + q_head_idx;
const device bfloat* qp = queries + batch_idx * queries_strides[0] +
    q_head_idx * queries_strides[1] +
    long(simd_lid) * qk_per_thread * queries_strides[3];
const device bfloat* kp = keys + batch_idx * keys_strides[0] +
    kv_head_idx * keys_strides[1] + block_idx * keys_strides[2] +
    long(simd_lid) * qk_per_thread * keys_strides[3];
const device bfloat* vp = values + batch_idx * values_strides[0] +
    kv_head_idx * values_strides[1] + block_idx * values_strides[2] +
    long(simd_lid) * v_per_thread * values_strides[3];
device bfloat* op = partials + o_offset * blocks * D + block_idx * D +
    simd_lid * v_per_thread;
device float* sp = sums + o_offset * blocks + block_idx;
device float* mp = maxs + o_offset * blocks + block_idx;
thread float q[qk_per_thread];
thread float o[v_per_thread] = {0};
for (int j = 0; j < qk_per_thread; ++j) {
    q[j] = scale * float(qp[long(j) * queries_strides[3]]);
}
float max_score = -MAXFLOAT;
float sum_exp_score = 0;
const long stride_k_tok = long(blocks) * keys_strides[2];
const long stride_v_tok = long(blocks) * values_strides[2];
const int N = int(length);
int i = block_idx;
for (; i + (TK - 1) * blocks < N; i += blocks * TK) {
    bfloat k_reg[TK][qk_per_thread];
    bfloat v_reg[TK][v_per_thread];
    for (int t = 0; t < TK; ++t) {
        const device bfloat* kpt = kp + t * stride_k_tok;
        for (int j = 0; j < qk_per_thread; ++j) k_reg[t][j] = kpt[long(j) * keys_strides[3]];
    }
    for (int t = 0; t < TK; ++t) {
        const device bfloat* vpt = vp + t * stride_v_tok;
        for (int j = 0; j < v_per_thread; ++j) v_reg[t][j] = vpt[long(j) * values_strides[3]];
    }
    for (int t = 0; t < TK; ++t) {
        float score = 0;
        for (int j = 0; j < qk_per_thread; ++j) {
            score += q[j] * float(k_reg[t][j]);
        }
        score = simd_sum(score);
        const float new_max = max(max_score, score);
        const float factor = fast::exp(max_score - new_max);
        const float exp_score = fast::exp(score - new_max);
        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;

        for (int j = 0; j < v_per_thread; ++j) {
            o[j] = o[j] * factor + exp_score * float(v_reg[t][j]);
        }
    }
    kp += TK * stride_k_tok;
    vp += TK * stride_v_tok;
}
for (; i < N; i += blocks) {
    float score = 0;
    for (int j = 0; j < qk_per_thread; ++j) {
        score += q[j] * float(kp[long(j) * keys_strides[3]]);
    }
    score = simd_sum(score);
    const float new_max = max(max_score, score);
    const float factor = fast::exp(max_score - new_max);
    const float exp_score = fast::exp(score - new_max);
    max_score = new_max;
    sum_exp_score = sum_exp_score * factor + exp_score;

    for (int j = 0; j < v_per_thread; ++j) {
        o[j] = o[j] * factor + exp_score * float(vp[long(j) * values_strides[3]]);
    }
    kp += stride_k_tok;
    vp += stride_v_tok;
}
if (simd_lid == 0) { sp[0] = sum_exp_score; mp[0] = max_score; }
for (int j = 0; j < v_per_thread; ++j) op[j] = bfloat(o[j]);
"""

_PASS2_BASE = r"""
constexpr int D = 256;
constexpr int BN = 32;
constexpr int BD = 32;
constexpr int blocks = $SPLITS;
constexpr int elem_per_thread = D / BD;
const uint3 tid = threadgroup_position_in_grid;
const uint simd_gid = simdgroup_index_in_threadgroup;
const uint simd_lid = thread_index_in_simdgroup;
thread float o[elem_per_thread] = {0};
threadgroup float outputs[BN * BD];
const int q_offset = tid.x;
const device bfloat* pp = partials + q_offset * blocks * D +
    simd_gid * D + simd_lid * elem_per_thread;
const device float* sp = sums + q_offset * blocks;
const device float* mp = maxs + q_offset * blocks;
device bfloat* op = out + q_offset * D + simd_gid * elem_per_thread;
float sum_exp_score = 0;
float max_score = -MAXFLOAT;
for (int b = 0; b < blocks / BN; ++b) {
    max_score = max(max_score, mp[simd_lid + BN * b]);
}
max_score = simd_max(max_score);
for (int b = 0; b < blocks / BN; ++b) {
    const float factor = fast::exp(mp[simd_lid + BN * b] - max_score);
    sum_exp_score += factor * sp[simd_lid + BN * b];
}
sum_exp_score = simd_sum(sum_exp_score);
for (int b = 0; b < blocks / BN; ++b) {
    const float factor = fast::exp(mp[simd_gid] - max_score);
    for (int j = 0; j < elem_per_thread; ++j) o[j] += factor * float(pp[j]);
    mp += BN;
    sp += BN;
    pp += BN * D;
}
for (int j = 0; j < elem_per_thread; ++j) {
    outputs[simd_lid * BD + simd_gid] = o[j];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[j] = simd_sum(outputs[simd_gid * BD + simd_lid]);
    o[j] = sum_exp_score == 0 ? o[j] : o[j] / sum_exp_score;
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
if (simd_lid == 0) {
    for (int j = 0; j < elem_per_thread; ++j) op[j] = bfloat(o[j]);
}
"""
