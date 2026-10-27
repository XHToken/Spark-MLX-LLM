#!/usr/bin/env bash
# 终端聊天示例：SPARK_MODEL=/path/to/model bash example/cli_chat.sh
set -euo pipefail

SPARK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SPARK_PYTHON="${SPARK_PYTHON:-$SPARK_ROOT/.venv/bin/python}"
SPARK_MODEL="${SPARK_MODEL:-/path/to/converted-mlx-model}"

# --system-prompt：每轮使用的系统提示；--max-tokens：每轮的输出上限。
# --temp 0：贪心采样；--no-thinking：关闭思考，直接回答。
# 启动后输入问题，q 退出、r 清空会话缓存、h 显示帮助。
# 等价命令为 spark-mlx-chat；末尾 "$@" 可覆盖示例参数。
exec "$SPARK_PYTHON" -m spark_mlx_llm.commands.chat \
    --model "$SPARK_MODEL" \
    --config "$SPARK_ROOT/example/inference.json" \
    --system-prompt '请使用中文简洁回答。' \
    --max-tokens 512 \
    --temp 0 \
    --no-thinking \
    "$@"
