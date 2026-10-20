// Copyright (c) 2026 Spark-MLX-LLM contributors

#include <dlfcn.h>
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <string>

#include "spark_w8a8.h"
#include "mlx/backend/metal/device.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx_spark_ext {
using namespace mlx::core;

namespace {

std::string binary_directory() {
  static const auto path = [] {
    Dl_info info;
    if (!dladdr(reinterpret_cast<void*>(&binary_directory), &info)) {
      throw std::runtime_error("Cannot locate mlx-spark-ext Metal library.");
    }
    return std::filesystem::path(info.dli_fname).parent_path().string();
  }();
  return path;
}

MTL::ComputePipelineState* kernel(Stream s, const std::string& name) {
  auto& dev = metal::device(s.device);
  auto library = dev.get_library("spark_w8a8", binary_directory());
  return dev.get_kernel(name, library);
}

void require_gpu(Stream s) {
  if (s.device != Device::gpu || !is_w8a8_available()) {
    throw std::invalid_argument(
        "W8A8 requires an Apple M5+ GPU and macOS 26.2 or newer.");
  }
}

bool real_dtype(Dtype dtype) {
  return dtype == float16 || dtype == bfloat16 || dtype == float32;
}

void validate_dimensions(const array& x, const array& weight) {
  if (x.ndim() < 1 || weight.ndim() != 2 || x.shape(-1) <= 0 ||
      weight.shape(0) <= 0 || x.size() == 0 ||
      x.shape(-1) != weight.shape(1)) {
    throw std::invalid_argument(
        "Expected nonempty x[..., K] and weight[N, K] with matching K.");
  }
  // INT32 accumulation must not overflow for full-range signed INT8 inputs.
  if (x.shape(-1) % 32 != 0 || x.shape(-1) > 131040 ||
      x.size() / x.shape(-1) > std::numeric_limits<int>::max()) {
    throw std::invalid_argument("K must be a multiple of 32 in [32, 131040].");
  }
  if (weight.dtype() != int8) {
    throw std::invalid_argument("weight must contain signed INT8 values.");
  }
}

std::string dtype_suffix(Dtype dtype) {
  return dtype == float16 ? "f16" : dtype == bfloat16 ? "bf16" : "f32";
}

void dispatch_gemm(
    Stream s, const array& x, const array& weight, array& out,
    const array* activation_scales = nullptr,
    const array* weight_scales = nullptr, const array* bias = nullptr) {
  const uint32_t M = x.shape(0), N = weight.shape(0), K = x.shape(1);
  const bool small = M <= 64;
  const uint32_t bm = small ? 32 : 128;
  const uint32_t tiles_m = (M + bm - 1) / bm, tiles_n = (N + 127) / 128;
  const uint32_t swizzle = tiles_m <= 3 ? 0 : tiles_m <= 6 ? 1 : 2;
  std::string name = activation_scales ? "w8a8_matmul_fused_dequant" : "int8_matmul_int32";
  if (small) name += "_small";
  if (activation_scales) name += "_" + dtype_suffix(out.dtype());
  auto& enc = metal::get_command_encoder(s);
  enc.set_compute_pipeline_state(kernel(s, name));
  enc.set_input_array(x, 0);
  enc.set_input_array(weight, 1);
  enc.set_output_array(out, 2);
  enc.set_bytes(M, 3);
  enc.set_bytes(N, 4);
  enc.set_bytes(K, 5);
  int index = 6;
  if (activation_scales) {
    enc.set_input_array(*activation_scales, index++);
    enc.set_input_array(*weight_scales, index++);
  }
  enc.set_bytes(swizzle, index++);
  enc.set_bytes(tiles_m, index++);
  enc.set_bytes(tiles_n, index++);
  if (bias) enc.set_input_array(*bias, index);
  enc.dispatch_threadgroups(
      MTL::Size(tiles_n * (1u << swizzle),
                (tiles_m + (1u << swizzle) - 1) >> swizzle, 1),
      MTL::Size(small ? 128 : 512, 1, 1));
}

class Int8Matmul : public UnaryPrimitive {
 public:
  explicit Int8Matmul(Stream s) : UnaryPrimitive(s) {}
  void eval_cpu(const std::vector<array>&, array&) override {
    throw std::runtime_error("INT8 TensorOps matmul is GPU-only.");
  }
  void eval_gpu(const std::vector<array>& inputs, array& out) override {
    out.set_data(allocator::malloc(out.nbytes()));
    dispatch_gemm(stream(), inputs[0], inputs[1], out);
  }
  const char* name() const override { return "SparkInt8MatmulInt32"; }
  bool is_equivalent(const Primitive& other) const override {
    return dynamic_cast<const Int8Matmul*>(&other) != nullptr;
  }
};

class W8A8Linear : public UnaryPrimitive {
 public:
  explicit W8A8Linear(Stream s) : UnaryPrimitive(s) {}
  void eval_cpu(const std::vector<array>&, array&) override {
    throw std::runtime_error("W8A8 linear is GPU-only.");
  }
  void eval_gpu(const std::vector<array>& inputs, array& out) override {
    const auto& x = inputs[0];
    const uint32_t M = x.shape(0), K = x.shape(1);
    out.set_data(allocator::malloc(out.nbytes()));
    array quantized({static_cast<int>(M), static_cast<int>(K)}, int8, nullptr, {});
    array scales({static_cast<int>(M)}, float32, nullptr, {});
    quantized.set_data(allocator::malloc(quantized.nbytes()));
    scales.set_data(allocator::malloc(scales.nbytes()));
    auto& enc = metal::get_command_encoder(stream());
    enc.set_compute_pipeline_state(kernel(stream(), "quantize_per_token_" + dtype_suffix(x.dtype())));
    enc.set_input_array(x, 0);
    enc.set_output_array(quantized, 1);
    enc.set_output_array(scales, 2);
    enc.set_bytes(K, 3);
    enc.dispatch_threadgroups(MTL::Size(M, 1, 1), MTL::Size(std::min(256u, K), 1, 1));
    enc.barrier();
    dispatch_gemm(stream(), quantized, inputs[1], out, &scales, &inputs[2], &inputs[3]);
    enc.add_temporary(quantized);
    enc.add_temporary(scales);
  }
  const char* name() const override { return "SparkW8A8Linear"; }
  bool is_equivalent(const Primitive& other) const override {
    return dynamic_cast<const W8A8Linear*>(&other) != nullptr;
  }
};

} // namespace

bool is_w8a8_available() {
  if (__builtin_available(macOS 26.2, *)) {
    try {
      const auto& dev = metal::device(Device::gpu);
      const auto& arch = dev.get_architecture();
      return !arch.empty() && dev.get_architecture_gen() >= (arch.back() == 'p' ? 18 : 17);
    } catch (const std::exception&) {
      return false;
    }
  }
  return false;
}

array int8_matmul_int32(const array& x, const array& weight, StreamOrDevice s) {
  validate_dimensions(x, weight);
  if (x.ndim() != 2 || x.dtype() != int8) {
    throw std::invalid_argument("int8_matmul_int32 requires x[M, K] with dtype int8.");
  }
  auto stream = to_stream(s);
  require_gpu(stream);
  return array({x.shape(0), weight.shape(0)}, int32,
      std::make_shared<Int8Matmul>(stream),
      {contiguous(x, false, stream), contiguous(weight, false, stream)});
}

array w8a8_linear(const array& x, const array& weight, const array& scales,
                  const std::optional<array>& bias, StreamOrDevice s) {
  validate_dimensions(x, weight);
  const int N = weight.shape(0), K = weight.shape(1);
  if (!real_dtype(x.dtype())) {
    throw std::invalid_argument("x must have dtype float16, bfloat16, or float32.");
  }
  if (scales.dtype() != float32 ||
      !(scales.shape() == Shape{N} || scales.shape() == Shape{N, 1})) {
    throw std::invalid_argument("scales must have shape [N] or [N, 1] and dtype float32.");
  }
  if (bias && (bias->shape() != Shape{N} || !real_dtype(bias->dtype()))) {
    throw std::invalid_argument("bias must have shape [N] and a real floating dtype.");
  }
  auto stream = to_stream(s);
  require_gpu(stream);
  const int M = static_cast<int>(x.size() / K);
  auto out = array({M, N}, x.dtype(), std::make_shared<W8A8Linear>(stream),
      {contiguous(reshape(x, {M, K}, stream), false, stream),
       contiguous(weight, false, stream),
       contiguous(reshape(scales, {N}, stream), false, stream),
       bias ? contiguous(astype(*bias, float32, stream), false, stream)
            : zeros({N}, float32, stream)});
  auto shape = x.shape();
  shape.back() = N;
  return reshape(out, shape, stream);
}

} // namespace mlx_spark_ext
