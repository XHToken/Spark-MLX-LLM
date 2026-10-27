"""用 Python 标准库调用 /v1/completions，直接续写原始 prompt。"""

import json
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# 直接编辑请求参数；此接口续写原始 prompt，不套用聊天模板。
BASE_URL = "http://127.0.0.1:8080"
TIMEOUT = 300
PAYLOAD = {
    "model": "default_model",
    "prompt": "春天来了，",
    "max_tokens": 128,
    "temperature": 0.0,
    "stream": False,
}


def main():
    request = Request(
        BASE_URL.rstrip("/") + "/v1/completions",
        data=json.dumps(PAYLOAD, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=TIMEOUT) as response:
            result = json.load(response)
        choice = result["choices"][0]
        print(choice["text"])
        print(f"finish_reason={choice.get('finish_reason')}", file=sys.stderr)
        print(f"usage={result.get('usage', {})}", file=sys.stderr)
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {error.code}: {detail}") from error
    except (URLError, OSError, ValueError) as error:
        raise SystemExit(f"请求失败：{error}") from error


if __name__ == "__main__":
    main()
