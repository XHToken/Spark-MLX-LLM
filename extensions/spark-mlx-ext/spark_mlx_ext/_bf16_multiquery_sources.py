"""Experimental MPP attention with FP32 scores and split BF16 probabilities."""

HEADER = r"""
#include <metal_stdlib>
#include <metal_simdgroup>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
"""

QK = r"""
const uint3 tid = threadgroup_position_in_grid;
const uint sg = simdgroup_index_in_threadgroup;
const int N = keys_shape[2];
const int token0 = (tid.x * SG + sg) * 32;
const int kv_head = tid.y / (32 / TM);
const int query0 = (tid.y % (32 / TM)) * TM;
if (token0 >= N) return;
constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(TM, 32, 16, false, false, false,
    mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
auto a = op.template get_left_input_cooperative_tensor<bfloat16_t, bfloat16_t, float>();
auto b = op.template get_right_input_cooperative_tensor<bfloat16_t, bfloat16_t, float>();
auto c = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(a)>,
    metal::remove_addrspace_t<decltype(b)>, float>();
#pragma unroll
for (uint i=0; i<c.get_capacity(); ++i) c[i] = 0;
for (int dim0=0; dim0<256; dim0+=16) {
    #pragma unroll
    for (uint i=0; i<a.get_capacity(); ++i) {
        const auto ix = a.get_multidimensional_index(i);
        const int qr = query0 + int(ix[1]);
        a[i] = queries[(kv_head*4 + qr/8) * queries_strides[1] +
            (qr%8) * queries_strides[2] + (dim0+int(ix[0])) * queries_strides[3]];
    }
    #pragma unroll
    for (uint i=0; i<b.get_capacity(); ++i) {
        const auto ix = b.get_multidimensional_index(i);
        const int token = token0 + int(ix[0]);
        b[i] = token < N ? keys[kv_head * keys_strides[1] + token * keys_strides[2] +
            (dim0+int(ix[1])) * keys_strides[3]] : bfloat16_t(0);
    }
    op.run(a,b,c);
}
#pragma unroll
for (uint i=0; i<c.get_capacity(); ++i) {
    const auto ix = c.get_multidimensional_index(i);
    const int token = token0 + int(ix[0]);
    const int qr = query0 + int(ix[1]);
    const int head = kv_head*4 + qr/8;
    const int query = qr%8;
    if (token < N) {
        bool visible = true;
        float bias = 0;
        if (MASK_MODE==1) visible = token <= N-8+query;
        else if (MASK_MODE>=2) {
            bias = float(mask[head * mask_strides[1] + query * mask_strides[2] + token * mask_strides[3]]);
            visible = MASK_MODE==2 ? bias != 0 : bias >= -MAXFLOAT;
        }
        scores[(head*8+query)*N + token] = visible ? c[i]*scale + (MASK_MODE==3 ? bias : 0) : -INFINITY;
    }
}
"""

SOFTMAX = r"""
const uint lane = thread_index_in_simdgroup;
const uint row = threadgroup_position_in_grid.x;
const int N = scores_shape[2];
float m = -INFINITY;
for (int token=lane; token<N; token+=32) m=max(m,scores[row*N+token]);
m=simd_max(m);
float z=0;
if (m > -INFINITY) {
    for (int token=lane; token<N; token+=32) z += fast::exp(scores[row*N+token]-m);
}
z=simd_sum(z);
for (int token=lane; token<N; token+=32) {
    const float p = z>0 ? fast::exp(scores[row*N+token]-m)/z : 0;
    const bfloat16_t high = bfloat16_t(p);
    highp[row*N+token] = high;
    lowp[row*N+token] = bfloat16_t(p - float(high));
}
"""

PV = r"""
const uint3 tid=threadgroup_position_in_grid;
const uint sg=simdgroup_index_in_threadgroup;
const int dim0=(tid.x * SG + sg) * 32;
const int kv_head=tid.y / (32/TM);
const int query0=(tid.y % (32/TM))*TM;
const int split=tid.z;
const int N=values_shape[2];
if (dim0 >= 256) return;
constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(TM,32,16,false,false,false,
    mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
mpp::tensor_ops::matmul2d<desc,metal::execution_simdgroup> op;
auto a=op.template get_left_input_cooperative_tensor<bfloat16_t,bfloat16_t,float>();
auto low=op.template get_left_input_cooperative_tensor<bfloat16_t,bfloat16_t,float>();
auto b=op.template get_right_input_cooperative_tensor<bfloat16_t,bfloat16_t,float>();
auto c=op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(a)>,
    metal::remove_addrspace_t<decltype(b)>,float>();
#pragma unroll
for (uint i=0; i<c.get_capacity(); ++i)c[i]=0;
for (int token0=split*16; token0<N; token0+=S*16) {
    #pragma unroll
    for (uint i=0; i<a.get_capacity(); ++i) {
        const auto ix=a.get_multidimensional_index(i);
        const int token=token0+int(ix[0]);
        const int qr=query0+int(ix[1]);
        const int row=(kv_head*4+qr/8)*8+qr%8;
        a[i]=token<N ? highp[row*N+token] : bfloat16_t(0);
        low[i]=token<N ? lowp[row*N+token] : bfloat16_t(0);
    }
    #pragma unroll
    for (uint i=0; i<b.get_capacity(); ++i) {
        const auto ix=b.get_multidimensional_index(i);
        const int token=token0+int(ix[1]);
        b[i]=token<N ? values[kv_head*values_strides[1]+token*values_strides[2]+
            (dim0+int(ix[0]))*values_strides[3]] : bfloat16_t(0);
    }
    op.run(a,b,c);op.run(low,b,c);
}
#pragma unroll
for (uint i=0; i<c.get_capacity(); ++i) {
    const auto ix=c.get_multidimensional_index(i);
    const int qr=query0+int(ix[1]);
    const int row=(kv_head*4+qr/8)*8+qr%8;
    partials[(split*128+row)*256+dim0+int(ix[0])]=OUT_TYPE(c[i]);
}
"""

MERGE = r"""
const uint i=thread_position_in_grid.x;
if (i >= 128*256) return;
float value=0;
#pragma unroll
for (int split=0; split<S; ++split)value+=partials[split*128*256+i];
out[i]=bfloat16_t(value);
"""
