"""用 Python 标准库调用 /v1/chat/completions，支持普通响应和 SSE 流。"""

import json
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# 先启动 cli_server.sh，然后在这里直接修改请求地址、消息和推理参数。
BASE_URL = "http://127.0.0.1:8080"
TIMEOUT = 300  # 网络超时秒数。
PAYLOAD = {
    "model": "default_model",  # 使用服务启动时指定的模型。
    "messages": [
        {"role": "system", "content": "请使用中文简洁回答。"},
        {"role": "user", "content": "请用一句话介绍你自己。"},
    ],
    "max_tokens": 2048,
    "temperature": 0.0,
    "stream": False,  # 改成 True 即可逐段读取 SSE 响应。
    "chat_template_kwargs": {
        "enable_thinking": False,  # 改成 True 开启思考。
        "reasoning_effort": "low",  # 开启思考时使用 low 或 high。
    },
}
# 多轮对话：在 messages 中补入历史 user/assistant 消息。
# 若需继承服务端思考配置，删除 chat_template_kwargs 字段。


def print_stream(response):
    # 当前 MLX LM 服务每个事件发送一行 data: JSON，空行分隔事件。
    for line in response:
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if not data:
            continue
        if data == b"[DONE]":
            print()
            return
        event = json.loads(data)
        if "error" in event:
            raise RuntimeError(f"服务端错误：{event['error']}")
        for choice in event.get("choices", []):
            delta = choice.get("delta", {})
            # 思考文本输出到 stderr，最终回答输出到 stdout，方便分别重定向。
            print(delta.get("reasoning") or "", end="", file=sys.stderr, flush=True)
            print(delta.get("content") or "", end="", flush=True)
            if choice.get("finish_reason"):
                print(f"\nfinish_reason={choice['finish_reason']}", file=sys.stderr)
    raise RuntimeError("流式响应提前结束，未收到 data: [DONE]")


def main():
    request = Request(
        BASE_URL.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(PAYLOAD, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=TIMEOUT) as response:
            if PAYLOAD["stream"]:
                print_stream(response)
            else:
                result = json.load(response)
                choice = result["choices"][0]
                message = choice["message"]
                if message.get("reasoning"):
                    print(message["reasoning"], file=sys.stderr)
                print(message.get("content") or "")
                print(f"finish_reason={choice.get('finish_reason')}", file=sys.stderr)
                print(f"usage={result.get('usage', {})}", file=sys.stderr)
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {error.code}: {detail}") from error
    except (URLError, OSError, ValueError, RuntimeError) as error:
        raise SystemExit(f"请求失败：{error}") from error


if __name__ == "__main__":
    main()
