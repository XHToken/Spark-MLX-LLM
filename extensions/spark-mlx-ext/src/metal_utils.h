// Copyright (c) 2026 Spark-MLX-LLM contributors
#pragma once

#include <dlfcn.h>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace spark_mlx_ext::detail {

// Resolve resources beside this extension binary, independently of cwd.
inline const std::string& binary_directory() {
  static const auto path = [] {
    Dl_info info;
    if (!dladdr(reinterpret_cast<void*>(&binary_directory), &info)) {
      throw std::runtime_error("Cannot locate spark-mlx-ext Metal library.");
    }
    return std::filesystem::path(info.dli_fname).parent_path().string();
  }();
  return path;
}

} // namespace spark_mlx_ext::detail
