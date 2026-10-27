#!/usr/bin/env bash
# HTTP 服务示例：SPARK_MODEL=/path/to/model bash example/cli_server.sh
set -euo pipefail

SPARK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SPARK_PYTHON="${SPARK_PYTHON:-$SPARK_ROOT/.venv/bin/python}"
SPARK_MODEL="${SPARK_MODEL:-/path/to/converted-mlx-model}"

# --model：启动时加载的模型；HTTP 客户端用 model="default_model" 调用它。
# --host / --port：监听地址和端口，此示例仅监听本机的 8080 端口。
# --max-tokens / --temp / --no-thinking：服务默认值，可被单次 HTTP 请求覆盖。
# 启动后在另一终端运行 python example/http_chat.py；Ctrl-C 停止服务。
# 等价命令为 spark-mlx-server；末尾 "$@" 可覆盖示例参数，如 --port 8090。
exec "$SPARK_PYTHON" -m spark_mlx_llm.commands.server \
    --model "$SPARK_MODEL" \
    --config "$SPARK_ROOT/example/inference.json" \
    --host 127.0.0.1 \
    --port 8080 \
    --max-tokens 2048 \
    --temp 0 \
    --no-thinking \
    "$@"
