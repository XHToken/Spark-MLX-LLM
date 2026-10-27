#!/usr/bin/env bash
# 单次生成示例：SPARK_MODEL=/path/to/model bash example/cli_generate.sh
set -euo pipefail

SPARK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SPARK_PYTHON="${SPARK_PYTHON:-$SPARK_ROOT/.venv/bin/python}"
SPARK_MODEL="${SPARK_MODEL:-/path/to/converted-mlx-model}"

# --model：模型目录或 HF 仓库 ID；压缩整数权重需要先转换为 MLX 格式。
# --prompt：本次问题，入口会自动套用模型的聊天模板。
# --max-tokens / --temp：输出长度上限和采样温度；温度 0 使用贪心采样。
# --no-thinking：关闭思考；可替换为 --thinking --reasoning-effort low。
# --w8-qmv-backend：已有 W8 层使用 native 实现，不改变权重量化位数。
# --kv-bits none：保留模型精度（BF16 模型使用 BF16 KV）；改为 8 可启用全局 KV 量化。
# --prefill-step-size：输入按 512 tokens 分块。
# 等价命令为 spark-mlx-generate；末尾 "$@" 可覆盖示例值，如 --max-tokens 1024。
exec "$SPARK_PYTHON" -m spark_mlx_llm.commands.generate \
    --model "$SPARK_MODEL" \
    --config "$SPARK_ROOT/example/inference.json" \
    --prompt '请用一句话介绍你自己。' \
    --max-tokens 256 \
    --temp 0 \
    --no-thinking \
    --w8-qmv-backend native \
    --kv-bits none \
    --prefill-step-size 512 \
    "$@"
