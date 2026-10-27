#!/usr/bin/env bash
# 模型转换示例：编辑下面两个路径，或通过同名环境变量设置。
# SPARK_SOURCE=/path/to/source SPARK_OUTPUT=/path/to/new-mlx bash example/cli_convert.sh
set -euo pipefail

SPARK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SPARK_PYTHON="${SPARK_PYTHON:-$SPARK_ROOT/.venv/bin/python}"
SPARK_SOURCE="${SPARK_SOURCE:-/path/to/source-model}"
SPARK_OUTPUT="${SPARK_OUTPUT:-/path/to/new-mlx-model}"

# --source：转换前的本地 BF16 或 compressed-tensors 模型目录。
# --output：新的 MLX 输出目录，必须不存在，且与 source 互不包含。
# --quantization：mixed 为 W4/G32 + 逐通道 W8；none 保留 BF16，仅限 BF16 输入。
# --w8-backend：native 使用 MLX 原生算子；也可选 w8a16 / w8a8，依赖相应扩展。
# --chunk-rows：每次在 CPU 上处理的行数，调小可减少转换时的临时内存。
# 等价命令为 spark-mlx-convert；末尾 "$@" 允许覆盖上面的示例参数。
exec "$SPARK_PYTHON" -m spark_mlx_llm.commands.convert \
    --source "$SPARK_SOURCE" \
    --output "$SPARK_OUTPUT" \
    --quantization mixed \
    --w8-backend native \
    --chunk-rows 256 \
    "$@"
