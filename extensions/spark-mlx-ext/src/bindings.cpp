// Copyright © 2026 Apple Inc.

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include "spark_qkv_rope.h"
#include "spark_w8a8.h"
#include "spark_w8a16.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_ext, m) {
  nb::module_::import_("mlx.core");
  m.def("w8a16_per_channel", &spark_mlx_ext::w8a16_per_channel,
        "x"_a, "weight"_a, "scales"_a,
        nb::kw_only(), "stream"_a = nb::none(),
        R"pbdoc(
          Compute a single-token per-channel INT8 projection without bias.

          Accepts signed INT8 weight[N,K], FP32 scales[N] or scales[N,1],
          and FP16/BF16 x[...,K] with exactly one activation row. Accumulates
          signed dot products in FP32, multiplies each result by its FP32
          channel scale, then casts to the input dtype. Output is [...,N].
          Activations are not quantized. Noncontiguous inputs are copied.
          Positive N and K, including tails, are supported. This is an
          inference-only GPU primitive; no CPU or autodiff fallback.
        )pbdoc");
  m.def("is_w8a8_available", &spark_mlx_ext::is_w8a8_available,
        "Return whether this device supports the W8A8 TensorOps kernels.");
  m.def("int8_matmul_int32", &spark_mlx_ext::int8_matmul_int32,
        "x"_a, "weight"_a, nb::kw_only(), "stream"_a = nb::none(),
        "Compute x[M,K] @ weight[N,K].T using INT8 TensorOps and INT32 accumulation.");
  m.def("w8a8_linear", &spark_mlx_ext::w8a8_linear,
        "x"_a, "weight"_a, "scales"_a, "bias"_a = nb::none(),
        nb::kw_only(), "stream"_a = nb::none(),
        R"pbdoc(
          Apply per-token symmetric INT8 activation quantization and a per-channel
          INT8 linear projection. Accepts existing INT8 weight[N,K] and FP32
          scales[N] or scales[N,1]. The input and output have matching FP16,
          BF16, or FP32 dtypes. Single-token calls also quantize activations.
          K must be a multiple of 32. Requires Apple M5+ and macOS 26.2+.
          This is an inference-only GPU primitive; no CPU or autodiff fallback.
        )pbdoc");
  m.def(
      "spark_qkv_rope",
      &spark_mlx_ext::spark_qkv_rope,
      "x"_a,
      "weight"_a,
      "scales"_a,
      "biases"_a,
      "q_size"_a,
      "kv_size"_a,
      "num_heads"_a,
      "num_key_value_heads"_a,
      "head_dim"_a,
      "rope_dims"_a,
      "rope_base"_a,
      "offset"_a,
      "group_size"_a = 128,
      "bits"_a = 4,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def spark_qkv_rope(x: array, weight: array, scales: array, biases: array, q_size: int, kv_size: int, num_heads: int, num_key_value_heads: int, head_dim: int, rope_dims: int, rope_base: float, offset: int, group_size: int = 128, bits: int = 4, *, stream: StreamOrDevice = None) -> list[array]"),
      R"pbdoc(
        Compute an affine quantized QKV projection and Q/K RoPE for one token.

        The input must have shape ``(1, 1, hidden)``. Weights and metadata use
        the layout returned by :func:`quantize` in affine mode. This function
        composes standard MLX operations in C++; it does not use a new kernel.
        RoPE uses non-traditional layout with scale 1 and an integer offset.
        Supported group sizes are 32, 64, and 128.

        Returns:
            list[array]: Query, key, and value arrays with shapes
            ``(1, num_heads, 1, head_dim)``,
            ``(1, num_key_value_heads, 1, head_dim)``, and
            ``(1, num_key_value_heads, 1, head_dim)``.
      )pbdoc");
}
