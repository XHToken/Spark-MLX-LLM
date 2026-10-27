// Copyright (c) 2026 Spark-MLX-LLM contributors

#include <cstdint>
#include <stdexcept>
#include <string>

#include "metal_utils.h"
#include "mlx/backend/metal/device.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "spark_w8a16.h"

namespace spark_mlx_ext {
using namespace mlx::core;

namespace {

MTL::ComputePipelineState* matvec_kernel(Stream stream, Dtype dtype, uint32_t n,
                                         uint32_t k, bool aligned_weight) {
  auto& dev = metal::device(stream.device);
  auto library = dev.get_library("spark_w8a16", detail::binary_directory());
  const std::string suffix = dtype == bfloat16 ? "bf16" : "f16";
  // These two production matrices have full eight-row/256-column tiles and
  // fit in 32-bit byte offsets. Compile-time dimensions let Metal optimize the
  // same compact dot loop used by the original per-channel experiment.
  if (aligned_weight && ((n == 2560 && k == 4096) || (n == 131072 && k == 2560))) {
    return dev.get_kernel("w8a16_per_channel_n" + std::to_string(n) + "_k" +
                              std::to_string(k) + "_" + suffix,
                          library);
  }
  const std::string name = "w8a16_per_channel_" + suffix;
  // Specializing N and K keeps the aligned-load and tail predicates constant,
  // as in the original compact per-channel decode experiment.
  const std::string hash_name = name + "_n" + std::to_string(n) + "_k" +
                                std::to_string(k) +
                                (aligned_weight ? "_aligned" : "_unaligned");
  const metal::MTLFCList constants = {
      {&n, MTL::DataType::DataTypeUInt, 0},
      {&k, MTL::DataType::DataTypeUInt, 1},
      {&aligned_weight, MTL::DataType::DataTypeBool, 2}};
  return dev.get_kernel(name, library, hash_name, constants);
}

class W8A16PerChannel : public UnaryPrimitive {
public:
  explicit W8A16PerChannel(Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<array>&, array&) override {
    throw std::runtime_error("Per-channel W8A16 decode is GPU-only.");
  }

  void eval_gpu(const std::vector<array>& inputs, array& out) override {
    const uint32_t n = inputs[1].shape(0);
    const uint32_t k = inputs[1].shape(1);
    out.set_data(allocator::malloc(out.nbytes()));
    auto& encoder = metal::get_command_encoder(stream());
    // A row-contiguous view can retain a nonzero byte offset. Only aligned
    // INT8 storage may be loaded through char4, even when K is divisible by 4.
    encoder.set_compute_pipeline_state(
        matvec_kernel(stream(), out.dtype(), n, k, inputs[1].offset() % 4 == 0));
    encoder.set_input_array(inputs[0], 0);
    encoder.set_input_array(inputs[1], 1);
    encoder.set_input_array(inputs[2], 2);
    encoder.set_output_array(out, 3);
    // Two SIMD groups each compute four output rows. No shared memory or
    // activation quantization is required for a single decode token.
    encoder.dispatch_threadgroups(MTL::Size((uint64_t{n} + 7) / 8, 1, 1),
                                  MTL::Size(64, 1, 1));
  }

  const char* name() const override { return "SparkW8A16PerChannel"; }

  bool is_equivalent(const Primitive& other) const override {
    return dynamic_cast<const W8A16PerChannel*>(&other) != nullptr;
  }
};

} // namespace

array w8a16_per_channel(const array& x, const array& weight, const array& scales,
                        StreamOrDevice s) {
  if (x.ndim() < 1 || weight.ndim() != 2 || x.shape(-1) <= 0 || weight.shape(0) <= 0 ||
      weight.shape(1) != x.shape(-1) || x.size() != static_cast<size_t>(x.shape(-1))) {
    throw std::invalid_argument(
        "w8a16_per_channel requires one activation row x[..., K] with "
        "singleton leading dimensions and nonempty weight[N, K].");
  }
  if (x.dtype() != bfloat16 && x.dtype() != float16) {
    throw std::invalid_argument(
        "w8a16_per_channel x must have dtype bfloat16 or float16.");
  }
  if (weight.dtype() != int8) {
    throw std::invalid_argument("w8a16_per_channel weight must have dtype int8.");
  }
  const int n = weight.shape(0);
  if (scales.dtype() != float32 ||
      !(scales.shape() == Shape{n} || scales.shape() == Shape{n, 1})) {
    throw std::invalid_argument(
        "w8a16_per_channel scales must have dtype float32 and shape [N] or [N, 1].");
  }
  auto stream = to_stream(s);
  if (stream.device != Device::gpu) {
    throw std::invalid_argument("w8a16_per_channel requires a GPU stream.");
  }
  auto shape = x.shape();
  shape.back() = n;
  return array(shape, x.dtype(), std::make_shared<W8A16PerChannel>(stream),
               {contiguous(x, false, stream), contiguous(weight, false, stream),
                contiguous(reshape(scales, {n}, stream), false, stream)});
}

} // namespace spark_mlx_ext
