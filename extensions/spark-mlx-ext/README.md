# spark-mlx-ext

English | [简体中文](README.zh.md)

Projection and attention operators for Spark-MLX-LLM, built against official
MLX 0.32.4. The Python package is `spark_mlx_ext`.

## Requirements

- Apple Silicon, macOS 26.2 or newer, and native ARM64 Python; Python 3.12 is recommended.
- Xcode with the Metal 4 compiler and MetalPerformancePrimitives headers.
- uv. Installation pins `mlx==0.32.4` for native ABI compatibility.
- W8A8 and raw INT8 matmul require Apple M5 or newer.

## Install

Run from the **Spark-MLX-LLM repository root**:

```sh
uv python install 3.12
xcrun --find metal
DEBUG=0 MACOSX_DEPLOYMENT_TARGET=26.2 \
CMAKE_ARGS='-DCMAKE_OSX_DEPLOYMENT_TARGET=26.2' \
uv sync --locked
```

On supported Apple Silicon environments, the main project requires this extension.
uv builds and installs it automatically from `extensions/spark-mlx-ext`.
After changing native sources, repeat the sync command with
`--reinstall-package spark-mlx-ext`, then restart Python.

## Quick start

Run this example from the same repository root after installation:

```sh
uv run --no-sync python - <<'PY'
import mlx.core as mx
from spark_mlx_ext import w8a16_per_channel

x = mx.ones((32,), dtype=mx.bfloat16)
weight = mx.ones((4, 32), dtype=mx.int8)
scales = mx.ones((4,), dtype=mx.float32)
y = w8a16_per_channel(x, weight, scales)
mx.eval(y)
print(y)  # Four values of 32, dtype bfloat16.
PY
```

## Operators

| API | Purpose / main constraints |
| --- | --- |
| `spark_qkv_rope` | Quantized QKV projection and RoPE; batch size 1, single token |
| `w4a16_qmv` | W4/G32 projection; BF16, one row, K divisible by 512, N divisible by 4 |
| `w8a16_per_channel` | Signed INT8 weights and FP32 channel scales; FP16/BF16, one row |
| `w8a8_linear`, `W8A8Linear` | INT8 activation/weight projection; FP16/BF16/FP32 input, M5+ |
| `int8_matmul_int32` | Signed INT8 matmul with INT32 output; M5+ |
| `int8_quantized_sdpa` | Quantized KV decode and prefill; fused D256 INT8/G32 paths with staged fallbacks |
| `bf16_sdpa` | BF16 decode; Q `[1,16,1,256]`, KV `[1,4,N,256]`, N 2048–131072 |

These are inference APIs. Installing the extension does not select every
optimized operator in a model; model configuration controls dispatch.

MIT licensed. Third-party license texts and source attribution are included
in `LICENSES/` and `THIRD_PARTY.md`.
