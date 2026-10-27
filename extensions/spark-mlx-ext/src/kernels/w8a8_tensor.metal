// Copyright (c) 2026 Spark-MLX-LLM contributors.
// Tensor-view and local split-K design adapted from MLX steel_gemm_thin_nax.h.
// Copyright (c) 2026 Apple Inc.
// Direct MPP tensor views and threadgroup-local INT32 split-K reduction.
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>
using namespace metal;

template<typename T, int BM, int BN, int BK, int KS>
void tensor_gemm(const device int8_t* A, const device int8_t* B, device T* C,
    uint M, uint N, uint K, const device float* sa, const device float* sw,
    const device float* bias, uint2 tg, uint sg, threadgroup int32_t* partials) {
  using extents_t = metal::dextents<int32_t,2>;
  using strides_t = metal::array<int32_t,2>;
  using tensor_t = metal::tensor<device int8_t,extents_t,metal::tensor_inline>;
  const uint mo=tg.y*BM, no=tg.x*BN;
  const int rows=min(uint(BM),M-mo), cols=min(uint(BN),N-no);
  A+=size_t(mo)*K; B+=size_t(no)*K;
  tensor_t a((device int8_t*)A,extents_t(K,rows),strides_t{1,int(K)});
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
    if(rows==BM && cols==BN && k+BK<=K) {
      auto av=a.template slice<BK,BM>(k,0);
      auto bv=b.template slice<BK,BN>(k,0);
      op.run(av,bv,acc);
    } else {
      const int kk=min(uint(BK),K-k);
      tensor_t av((device int8_t*)A+k,extents_t(kk,rows),strides_t{1,int(K)});
      tensor_t bv((device int8_t*)B+k,extents_t(kk,cols),strides_t{1,int(K)});
      op.run(av,bv,acc);
    }
  }
  if constexpr(KS>1) {
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
  if(sg!=0) return;
  #pragma unroll
  for(uint i=0;i<acc.get_capacity();i++) {
    auto idx=acc.get_multidimensional_index(i);
    const uint m=mo+idx[1], n=no+idx[0];
    if(m<M && n<N) C[size_t(m)*N+n]=T(float(acc[i])*sa[m]*sw[n]+bias[n]);
  }
}

#define KERNEL(T,DT,ID,BM,BN,BK,KS) \
kernel void w8a8_tensor_##ID##_##DT( \
    const device int8_t* A [[buffer(0)]],const device int8_t* B [[buffer(1)]], \
    device T* C [[buffer(2)]],constant uint& M [[buffer(3)]], \
    constant uint& N [[buffer(4)]],constant uint& K [[buffer(5)]], \
    const device float* sa [[buffer(6)]],const device float* sw [[buffer(7)]], \
    const device float* bias [[buffer(8)]],uint2 tg [[threadgroup_position_in_grid]], \
    uint sg [[simdgroup_index_in_threadgroup]]) { \
  threadgroup int32_t partials[KS>1?(KS-1)*BM*BN:1]; \
  tensor_gemm<T,BM,BN,BK,KS>(A,B,C,M,N,K,sa,sw,bias,tg,sg,partials); \
}
#define DTYPES(ID,BM,BN,BK,KS) \
KERNEL(half,f16,ID,BM,BN,BK,KS) \
KERNEL(bfloat,bf16,ID,BM,BN,BK,KS) \
KERNEL(float,f32,ID,BM,BN,BK,KS)
DTYPES(0,16,32,64,4)
DTYPES(5,32,64,128,1)
DTYPES(6,32,64,128,2)
