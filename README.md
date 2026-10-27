<p align="center">
  <img src="docs/assets/spark-mlx-logo.png" alt="Spark and MLX" width="480">
</p>

<h1 align="center">Spark-MLX-LLM</h1>

<p align="center">
  <strong>English</strong> | <a href="README.zh.md">简体中文</a>
</p>

Spark-MLX-LLM is a local inference project for **Spark 2.6 and Spark 2.5**, built on
[MLX](https://github.com/ml-explore/mlx) and [MLX LM](https://github.com/ml-explore/mlx-lm).
It adapts Spark's model architecture and mixed-precision quantization to MLX,
with dedicated inference optimizations for Apple Silicon. The project provides
model conversion, text generation, interactive chat, and HTTP serving to make
Spark models easier to deploy and use locally.

## Features

- **Spark model adaptation and conversion:** convert BF16 and compressed-tensors
  INT4 checkpoints to MLX format.
- **Mixed-precision quantization:** use INT4, INT8, and BF16 weights for different
  modules, with configurable W8 execution backends.
- **Apple Silicon optimizations:** dedicated QKV/RoPE, linear, and INT8 KV
  attention operators, selected according to configuration, device, and input shape.
- **Multiple interfaces:** command-line generation and chat, Python generation
  and streaming APIs, and an HTTP service.

## Environment setup

The Apple Silicon setup below requires:

| Component | Requirement |
| --- | --- |
| Hardware and OS | Apple Silicon, macOS >= 26.2 |
| Python | Native `arm64` CPython; 3.12 recommended |
| Build tools | Xcode with the Metal 4 compiler and MetalPerformancePrimitives headers |

W8A8 acceleration additionally requires **Apple M5 or newer** and a compatible
`spark-mlx-ext`. When that backend is unavailable, W8 projections use MLX affine
operators with floating-point activations. See [quantization and compatibility](#quantization-and-compatibility).

Use [uv](https://docs.astral.sh/uv/getting-started/installation/) to manage Python
and dependencies. After preparing Xcode and installing uv:

```sh
git clone https://github.com/XHToken/Spark-MLX-LLM.git
cd Spark-MLX-LLM
uv python install 3.12
uv sync --locked
```

This creates `.venv`, installs the project dependencies, and builds the bundled
Apple Silicon extension. Run subsequent commands from the repository root.
Inference requires a Metal or CUDA GPU; model conversion runs on CPU.
For Linux/CUDA installation, environment checks, and troubleshooting, see the
[environment guide](docs/en/environment.md).

## Quick start

### 1. Download a model

**Spark-X2.6-4B-INT4 is recommended.** Download either format to a local
directory for conversion in step 2.

| Model | Format | Hugging Face | ModelScope |
| --- | --- | --- | --- |
| Spark-X2.6-4B | BF16 | [Download](https://huggingface.co/XHToken/Spark-X2.6-4B) | [Download](https://www.modelscope.cn/models/XHToken/Spark-X2.6-4B) |
| Spark-X2.6-4B-INT4 | INT4 | [Download](https://huggingface.co/XHToken/Spark-X2.6-4B-INT4) | [Download](https://www.modelscope.cn/models/XHToken/Spark-X2.6-4B-INT4) |

### 2. Quantize and convert

Convert the downloaded model to MLX format:

```sh
.venv/bin/spark-mlx-convert \
    --source /path/to/Spark-X2.6-4B-INT4 \
    --output /path/to/Spark-X2.6-4B-MLX
```

`--source` is the local directory from step 1; `--output` must be a new directory.
The default conversion uses mixed INT4/INT8/BF16 weights; see the
[quantization scheme below](#quantization-and-compatibility) and the
[conversion guide](docs/en/conversion.md) for details.

### 3. Run inference

**Load the converted output directory from step 2** with `--model`.

Generate a response:

```sh
.venv/bin/spark-mlx-generate \
    --model /path/to/Spark-X2.6-4B-MLX \
    --prompt '你好，请介绍一下自己。' \
    --max-tokens 256 \
    --no-thinking
```

Start an interactive chat:

```sh
.venv/bin/spark-mlx-chat \
    --model /path/to/Spark-X2.6-4B-MLX \
    --no-thinking
```

These examples disable thinking with `--no-thinking`. For a model template that
supports thinking, replace it with `--thinking --reasoning-effort low` (or `high`).
`--max-tokens` limits the combined thinking and final-answer length; increase it
as needed. See [thinking modes](docs/en/examples.md#thinking-modes).

### 4. Start an HTTP service

Serve the converted model on localhost:

```sh
.venv/bin/spark-mlx-server \
    --model /path/to/Spark-X2.6-4B-MLX \
    --host 127.0.0.1 \
    --port 8080 \
    --no-thinking
```

Keep this terminal running and send requests to `http://127.0.0.1:8080` from
another terminal or application. Stop the service with Ctrl-C. See the
[HTTP examples](docs/en/examples.md#4-http-chat-streaming-chat-and-text-completion)
for chat, streaming responses, and text completion.

See the [complete interface guide](docs/en/examples.md) for `generate`, `chat`,
`server`, Python generation and streaming APIs, and HTTP requests.

## Quantization and compatibility

The default conversion applies the following scheme to the main model. Decoder
modules use the same scheme in every layer; embedding and `lm_head` are model-wide.
The activation column describes execution with the W8A8 backend available and
enabled, while the weight and scale columns describe the converted checkpoint.

| Layer / module | Weight quantization | Activations (W8A8 enabled) | Weight scale type |
| --- | --- | --- | --- |
| `self_attn.q_k_v_proj` | Asymmetric INT4, group size 32 | BF16 | BF16 |
| `mlp.gate_proj` | Asymmetric INT4, group size 32 | BF16 | BF16 |
| `mlp.up_proj` | Asymmetric INT4, group size 32 | BF16 | BF16 |
| `mlp.down_proj` | Asymmetric INT4, group size 32 | BF16 | BF16 |
| `self_attn.out_proj` | Symmetric INT8, per output channel | Dynamic per-token INT8 | BF16 |
| `lm_head` | Symmetric INT8, per vocabulary row | Dynamic per-token INT8 | FP32 |
| `self_attn.g_proj` | BF16, unquantized | BF16 | — |
| `model.embedding` | BF16, unquantized | — | — |
| RMSNorm | BF16, unquantized | Unquantized | — |

The converter defaults to `--w8-backend w8a8`. If W8A8 is unavailable at load
time, the model retains MLX affine W8 operators with floating-point activations.
To select that path explicitly, add `--w8-backend native` to the conversion
command. The W8 weights remain unchanged; floating-point and INT8 activation
paths can produce different numerical results. See the
[conversion guide](docs/en/conversion.md) for conversion options and precision details.

All generation entry points default to unquantized KV caches that retain model
precision: **BF16 models use BF16 KV**, including full-attention and sliding-window
layers. To opt into INT8/G32 KV for full-attention layers, use `--kv-bits 8` with
`spark-mlx-generate` or `kv_bits=8` with this package's Python generation functions.
Generate and the Python wrappers use 512-token prompt chunks; chat and server
retain upstream MLX LM chunking behavior. These settings are separate from weight
quantization; see
[KV caching and prompt chunking](docs/en/examples.md#kv-caching-and-prompt-chunking).

## Documentation

See the [documentation index](docs/README.md) for all guides.

| Topic | Guide |
| --- | --- |
| Installation, platform requirements, and troubleshooting | [Environment setup](docs/en/environment.md) |
| CLI, Python, HTTP, and thinking settings | [Interface guide](docs/en/examples.md) |
| Model conversion and quantization | [Conversion guide](docs/en/conversion.md) |
| Benchmark commands and metric definitions | [Benchmark guide](docs/en/benchmark.md) |

## License

See [LICENSE](LICENSE).
