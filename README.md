# Spark-MLX-LLM

Spark2_5 inference support for [MLX](https://github.com/ml-explore/mlx) and
[MLX LM](https://github.com/ml-explore/mlx-lm).

The package loads the original Hugging Face `safetensors` checkpoint directly.
It does not require GGUF conversion and does not modify the installed `mlx-lm`
package.

## Features

- Direct loading from a local Hugging Face checkpoint or repository ID
- Strict checkpoint weight-name and shape validation
- CPU, Apple Silicon GPU, and optional Linux CUDA execution
- BF16 and FP32 weight overrides
- Sliding-window and full-attention KV caches
- Automatic Spark2_5 function-call parsing through MLX LM
- Spark2_5 wrappers for MLX LM conversion, chat, and server commands
- Command-line and Python inference interfaces

## Supported model

The implementation supports `Spark2_5ForCausalLM` checkpoints with
`model_type: spark2_5`. The reference 1.7B checkpoint uses:

- 28 decoder layers
- 8 query heads and 2 key/value heads
- Fused QKV projection
- Head-wise sigmoid attention output gates
- Three sliding-window layers followed by one full-attention layer
- A 512-token sliding window
- 256 rotary dimensions with theta 10,000 in sliding layers
- 64 rotary dimensions with theta 1,000,000 in full-attention layers
- Parallel GELU feed-forward layers
- Tied input and output embeddings

The checkpoint directory must contain at least:

```text
config.json
model*.safetensors
tokenizer.json
tokenizer_config.json
```

If the tokenizer is stored in a nested directory, copy the tokenizer files to
the checkpoint root before loading the model.

## Installation

Clone the repository and create an isolated environment:

```sh
git clone https://github.com/XHToken/Spark-MLX-LLM.git
cd Spark-MLX-LLM
python3 -m venv .venv
```

Install the backend required by the host.

Apple Silicon:

```sh
.venv/bin/python -m pip install -e '.[test]'
```

Linux CPU:

```sh
.venv/bin/python -m pip install -e '.[cpu,test]'
```

Linux CUDA 12 or CUDA 13:

```sh
.venv/bin/python -m pip install -e '.[cuda12,test]'
# or
.venv/bin/python -m pip install -e '.[cuda13,test]'
```

Confirm the installation:

```sh
.venv/bin/python -c 'import mlx.core as mx; print(mx.__version__)'
```

## Quick start

Run deterministic CPU generation from a local checkpoint:

```sh
.venv/bin/spark-mlx-generate \
    --device cpu \
    --dtype bfloat16 \
    --model /path/to/spark2_5 \
    --prompt '只输出53乘以42的结果，不要解释。' \
    --max-tokens 64 \
    --temp 0 \
    --seed 1
```

Expected response:

```text
53乘以42的结果是2226。
```

Run on a selected CUDA GPU:

```sh
CUDA_VISIBLE_DEVICES=0 .venv/bin/spark-mlx-generate \
    --device gpu \
    --dtype bfloat16 \
    --model /path/to/spark2_5 \
    --prompt '只输出53乘以42的结果，不要解释。' \
    --max-tokens 64 \
    --temp 0 \
    --seed 1
```

Confirm that MLX sees the CUDA device before inference:

```sh
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -c \
    'import mlx.core as mx; mx.set_default_device(mx.gpu); print(mx.default_device()); print(mx.device_info())'
```

Hugging Face repository IDs are also accepted:

```sh
.venv/bin/spark-mlx-generate \
    --model XHToken/your-spark2_5-repository \
    --prompt '你好' \
    --max-tokens 128
```

The repository must expose the model and tokenizer files required above.

## MLX LM tool wrappers

The package provides wrappers for MLX LM tools that need to resolve the
Spark2_5 model architecture. Registration is limited to the command process;
the wrappers do not modify the installed `mlx_lm` package.

Convert or quantize a Hugging Face checkpoint:

```sh
.venv/bin/spark-mlx-convert \
    --hf-path XHToken/Spark-X2.5-1.7B \
    --mlx-path ./Spark-X2.5-1.7B-8bit \
    --quantize \
    --q-bits 8
```

The converter keeps the head-wise attention `g_proj` gates in BF16. These
small sigmoid gates scale every attention head and are sensitive to low-bit
quantization. Use 8-bit or BF16 weights for tool calling; 4-bit weights use
less memory but can reduce the accuracy of schema-constrained arguments.

Start an interactive chat session:

```sh
.venv/bin/spark-mlx-chat \
    --model ./Spark-X2.5-1.7B-8bit
```

Start the OpenAI-compatible server:

```sh
.venv/bin/spark-mlx-server \
    --model ./Spark-X2.5-1.7B-8bit \
    --host 127.0.0.1 \
    --port 8080
```

Use these `spark-mlx-*` commands until Spark2_5 is available in an official
MLX LM release. If the installed MLX LM already provides a native Spark2_5
module, the wrappers use it instead of the bundled implementation.

## Python API

```python
from mlx_lm import generate

from spark_mlx_llm import load


model, tokenizer = load(
    "/path/to/spark2_5",
    dtype="bfloat16",
)
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "你好"}],
    tokenize=False,
    add_generation_prompt=True,
)
response = generate(
    model,
    tokenizer,
    prompt=prompt,
    max_tokens=128,
)
print(response)
```

When the checkpoint provides the Spark2_5 tool-aware chat template, the returned
tokenizer automatically exposes MLX LM's tool parser:

```python
assert tokenizer.has_tool_calling
tool_call = tokenizer.tool_parser(
    "set_state"
    "<arg_key>name</arg_key><arg_value>上海</arg_value>"
    "<arg_key>count</arg_key><arg_value>42</arg_value>",
    [
        {
            "type": "function",
            "function": {
                "name": "set_state",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "count": {"type": "integer"},
                    },
                },
            },
        }
    ],
)
```

Select an execution device before loading the model when using the Python API:

```python
import mlx.core as mx

mx.set_default_device(mx.gpu)  # or mx.cpu
```

## CUDA implementation note

The checkpoint stores Q, K, and V in one fused projection. After the projection,
the implementation splits, reshapes, and transposes the result into attention
heads. These tensors are non-contiguous views into the fused projection.

Passing those views directly to the CUDA RoPE path caused the K view to be
evaluated with incorrect storage aliasing. The error first appeared at layer 0
K-RoPE and later propagated through attention and the remaining layers. In the
full lazy graph it could produce all-NaN logits or an illegal CUDA memory access.

The model materializes Q, K, and V before RoPE and attention:

```python
queries = mx.contiguous(queries)
keys = mx.contiguous(keys)
values = mx.contiguous(values)
```

This is a compatibility workaround for the affected non-contiguous CUDA path;
it does not change the Spark2_5 computation.

## Verification

Run the unit tests and lint checks:

```sh
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
```

Run the real-checkpoint integration test on CPU:

```sh
SPARK25_MODEL=/path/to/spark2_5 \
SPARK25_DTYPE=bfloat16 \
    .venv/bin/python -m pytest tests/test_integration.py -q -m integration
```

Run the complete test suite on CUDA:

```sh
CUDA_VISIBLE_DEVICES=0 \
SPARK25_MODEL=/path/to/spark2_5 \
SPARK25_DTYPE=bfloat16 \
    .venv/bin/python -m pytest -q
```

The integration test performs strict weight loading, validates complex
function-call parsing, and runs a real checkpoint forward pass. Always run an
end-to-end generation command as well, because a successful shape test alone
does not establish generation correctness.

The CUDA fix was validated with MLX 0.32.2, MLX LM 0.31.3, BF16 weights, and an
NVIDIA H100 PCIe GPU:

- CPU and CUDA prefill logits were finite
- CPU and CUDA selected the same first token
- Full-logit relative RMSE was approximately 0.528%
- The CUDA test suite reported `6 passed`
- Three consecutive deterministic generations produced the expected answer
- No NaN, illegal memory access, or non-deterministic text was observed

Other CUDA architectures should be validated with the same integration and
end-to-end generation commands before deployment.

## Project layout

```text
spark_mlx_llm/model.py   Spark2_5 computation graph and cache selection
spark_mlx_llm/loader.py  Local and Hugging Face checkpoint loading
spark_mlx_llm/cli.py     Text generation command
tests/                   Unit and real-checkpoint tests
```

## License

See [LICENSE](LICENSE).
