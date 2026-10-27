# spark-mlx-ext

[English](README.md) | 简体中文

Spark-MLX-LLM 的投影与注意力算子扩展，基于官方 MLX 0.32.4 构建。
Python 包名为 `spark_mlx_ext`。

## 环境要求

- Apple Silicon、macOS 26.2 或更新版本、原生 ARM64 Python；推荐 Python 3.12。
- 包含 Metal 4 编译器和 MetalPerformancePrimitives 头文件的 Xcode。
- uv。安装时固定 `mlx==0.32.4`，保证原生扩展的 ABI 兼容性。
- W8A8 和原始 INT8 矩阵乘法需要 Apple M5 或更新芯片。

## 安装

在 **Spark-MLX-LLM 仓库根目录**执行：

```sh
uv python install 3.12
xcrun --find metal
DEBUG=0 MACOSX_DEPLOYMENT_TARGET=26.2 \
CMAKE_ARGS='-DCMAKE_OSX_DEPLOYMENT_TARGET=26.2' \
uv sync --locked
```

在受支持的 Apple Silicon 环境中，本扩展是主项目的必需依赖。
uv 会自动从 `extensions/spark-mlx-ext` 编译并安装扩展。
修改原生代码后，在上述同步命令中追加 `--reinstall-package spark-mlx-ext`
重新构建，然后重启 Python。

## 快速使用

安装完成后，在同一仓库根目录运行：

```sh
uv run --no-sync python - <<'PY'
import mlx.core as mx
from spark_mlx_ext import w8a16_per_channel

x = mx.ones((32,), dtype=mx.bfloat16)
weight = mx.ones((4, 32), dtype=mx.int8)
scales = mx.ones((4,), dtype=mx.float32)
y = w8a16_per_channel(x, weight, scales)
mx.eval(y)
print(y)  # 输出四个 32，数据类型为 bfloat16。
PY
```

## 算子

| API | 用途 / 主要约束 |
| --- | --- |
| `spark_qkv_rope` | 量化 QKV 投影与 RoPE；batch size 为 1，单 token |
| `w4a16_qmv` | W4/G32 投影；BF16、单行、K 为 512 的倍数、N 为 4 的倍数 |
| `w8a16_per_channel` | 有符号 INT8 权重与 FP32 通道缩放；FP16/BF16、单行 |
| `w8a8_linear`、`W8A8Linear` | INT8 激活与权重投影；FP16/BF16/FP32 输入，M5+ |
| `int8_matmul_int32` | 有符号 INT8 矩阵乘法，输出 INT32；M5+ |
| `int8_quantized_sdpa` | 量化 KV decode 与 prefill；融合 D256 INT8/G32 路径及分阶段回退 |
| `bf16_sdpa` | BF16 decode；Q `[1,16,1,256]`，KV `[1,4,N,256]`，N 为 2048–131072 |

这些 API 用于推理。安装扩展不会让模型自动启用全部优化算子，
具体算子分派由模型配置决定。

采用 MIT 许可证。第三方许可证文本和来源说明随包提供，
分别位于 `LICENSES/` 和 `THIRD_PARTY.md`。
