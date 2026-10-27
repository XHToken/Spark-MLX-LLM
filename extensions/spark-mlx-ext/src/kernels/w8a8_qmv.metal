// Copyright (c) 2026 Spark-MLX-LLM contributors.
// Based on the v9 direct tensor-view kernel, with independent M=1 ablations.
// Tensor view design adapted from Apple MLX.
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>
using namespace metal;

template<typename T, int BM, int BN, int BK, int KS, bool Compact, bool Static>
void qmv(const device int8_t* A, const device int8_t* B, device T* C,
    uint M, uint N, uint K, const device float* sa, const device float* sw,
    const device float* bias, uint2 tg, uint sg, threadgroup int32_t* partials) {
  using extents_t = metal::dextents<int32_t,2>;
  using strides_t = metal::array<int32_t,2>;
  using tensor_t = metal::tensor<device int8_t,extents_t,metal::tensor_inline>;
  const uint no=tg.x*BN;
  const int cols=min(uint(BN),N-no);
  B+=size_t(no)*K;
  tensor_t a((device int8_t*)A,extents_t(K,1),strides_t{1,int(K)});
  tensor_t b((device int8_t*)B,extents_t(K,cols),strides_t{1,int(K)});
  constexpr auto desc=mpp::tensor_ops::matmul2d_descriptor(
      BM,BN,BK,false,true,true,mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<desc,metal::execution_simdgroup> op;
  using at=decltype(a.template slice<BK,BM>(0,0));
  using bt=decltype(b.template slice<BK,BN>(0,0));
  auto acc=op.template get_destination_cooperative_tensor<at,bt,int32_t>();
  #pragma unroll
  for(uint i=0;i<acc.get_capacity();i++) acc[i]=0;
  for(uint k=sg*BK;k<K;k+=KS*BK) {
    if constexpr(Static) {
      if(cols==BN && k+BK<=K) {
        auto av=a.template slice<BK,1>(k,0);
        auto bv=b.template slice<BK,BN>(k,0);
        op.run(av,bv,acc);
      } else {
        const int kk=min(uint(BK),K-k);
        tensor_t av((device int8_t*)A+k,extents_t(kk,1),strides_t{1,int(K)});
        tensor_t bv((device int8_t*)B+k,extents_t(kk,cols),strides_t{1,int(K)});
        op.run(av,bv,acc);
      }
    } else {
      const int kk=min(uint(BK),K-k);
      tensor_t av((device int8_t*)A+k,extents_t(kk,1),strides_t{1,int(K)});
      tensor_t bv((device int8_t*)B+k,extents_t(kk,cols),strides_t{1,int(K)});
      op.run(av,bv,acc);
    }
  }
  if constexpr(KS>1) {
    if constexpr(Compact) {
      if(sg>0) {
        #pragma unroll
        for(uint i=0;i<acc.get_capacity();i++) {
          auto idx=acc.get_multidimensional_index(i);
          if(idx[1]==0) partials[(sg-1)*BN+idx[0]]=acc[i];
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if(sg==0) {
        #pragma unroll
        for(uint i=0;i<acc.get_capacity();i++) {
          auto idx=acc.get_multidimensional_index(i);
          if(idx[1]==0) {
            #pragma unroll
            for(uint s=1;s<KS;s++) acc[i]+=partials[(s-1)*BN+idx[0]];
          }
        }
      }
    } else {
      using partial_t=metal::tensor<threadgroup int32_t,extents_t,metal::tensor_inline>;
      if(sg>0) {
        partial_t p(partials+(sg-1)*BM*BN,extents_t(BN,BM),strides_t{1,BN});
        acc.store(p);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if(sg==0) {
        for(uint s=1;s<KS;s++) {
          partial_t p(partials+(s-1)*BM*BN,extents_t(BN,BM),strides_t{1,BN});
          auto tmp=op.template get_destination_cooperative_tensor<at,bt,int32_t>();
          tmp.load(p);
          #pragma unroll
          for(uint i=0;i<acc.get_capacity();i++) acc[i]+=tmp[i];
        }
      }
    }
  }
  if(sg!=0) return;
  #pragma unroll
  for(uint i=0;i<acc.get_capacity();i++) {
    auto idx=acc.get_multidimensional_index(i);
    const uint n=no+idx[0];
    if(idx[1]==0 && n<N) C[n]=T(float(acc[i])*sa[0]*sw[n]+bias[n]);
  }
}

#define KERNEL(T,DT,ID,BM,BN,BK,KS,C,S) \
kernel void w8a8_qmv_##ID##_##DT( \
    const device int8_t* A [[buffer(0)]],const device int8_t* B [[buffer(1)]], \
    device T* O [[buffer(2)]],constant uint& M [[buffer(3)]], \
    constant uint& N [[buffer(4)]],constant uint& K [[buffer(5)]], \
    const device float* sa [[buffer(6)]],const device float* sw [[buffer(7)]], \
    const device float* bias [[buffer(8)]],uint2 tg [[threadgroup_position_in_grid]], \
    uint sg [[simdgroup_index_in_threadgroup]]) { \
  threadgroup int32_t partials[KS>1?(KS-1)*(C?1:BM)*BN:1]; \
  qmv<T,BM,BN,BK,KS,C,S>(A,B,O,M,N,K,sa,sw,bias,tg,sg,partials); \
}
#define DTYPES(ID,BM,BN,BK,KS,C,S) \
KERNEL(half,f16,ID,BM,BN,BK,KS,C,S) \
KERNEL(bfloat,bf16,ID,BM,BN,BK,KS,C,S) \
KERNEL(float,f32,ID,BM,BN,BK,KS,C,S)

DTYPES(15,8,32,64,4,true,true)
