"""Python API 示例：直接调用 convert_to_mlx()，得到转换报告。"""

import json
from pathlib import Path

from spark_mlx_llm import convert_to_mlx

# 修改为实际路径。输出目录必须不存在，且不能与源目录互相包含。
SOURCE = Path("/path/to/source-model")
OUTPUT = Path("/path/to/new-mlx-model")


def main():
    report = convert_to_mlx(
        source=SOURCE,
        output=OUTPUT,
        quantization="mixed",  # W4/G32 + 逐通道 W8；BF16 输入可用 none 保留精度。
        w8_backend="native",  # 输出模型的 W8 计算策略，也支持 w8a16 / w8a8。
        chunk_rows=256,  # 每次在 CPU 上处理的行数。
    )
    # 如果只检查结构、不写文件，将上面的调用替换为：
    # report = convert_to_mlx(source=SOURCE, inspect_only=True)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
