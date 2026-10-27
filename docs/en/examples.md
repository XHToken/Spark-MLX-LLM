# API usage examples

[Project overview](../../README.md) · [简体中文](../zh/examples.md)

These scripts demonstrate the project's Python APIs, four command-line entry
points, and HTTP endpoints. Python API examples set parameters and call functions
directly in code; they do not parse command-line arguments. Copy the relevant
calls into your application, or use the shell examples to pass arguments from
a terminal. See [shared inference settings](#5-shared-inference-settings) and
[upstream integration](#6-how-the-entry-points-relate-to-upstream-mlx-lm) below
for configuration details and differences between interfaces.

| Script | Interface | Purpose |
| --- | --- | --- |
| [python_convert.py](../../example/python_convert.py) | `convert_to_mlx()` | Inspect or convert a model and print a JSON report |
| [python_generate.py](../../example/python_generate.py) | `load()` + `generate()` | Return the complete generated text |
| [python_stream_generate.py](../../example/python_stream_generate.py) | `load()` + `stream_generate()` | Process generated text incrementally |
| [benchmark.py](../../example/benchmark.py) | `load()` + `stream_generate()` | Measure prefill/decode throughput for multiple input lengths from the command line |
| [cli_convert.sh](../../example/cli_convert.sh) | Module entry point for `spark-mlx-convert` | Inspect or convert a model from the command line |
| [cli_generate.sh](../../example/cli_generate.sh) | Module entry point for `spark-mlx-generate` | Generate a single response from the command line |
| [cli_chat.sh](../../example/cli_chat.sh) | Module entry point for `spark-mlx-chat` | Run an interactive terminal chat |
| [cli_server.sh](../../example/cli_server.sh) | Module entry point for `spark-mlx-server` | Start a local HTTP server |
| [http_chat.py](../../example/http_chat.py) | `POST /v1/chat/completions` | Request a complete chat response or an SSE stream |
| [http_completions.py](../../example/http_completions.py) | `POST /v1/completions` | Continue a raw text prompt |

## Prerequisites

Run the commands below from the repository root, using a `.venv` with this
project installed as described in the [environment guide](environment.md).
Python API calls and command-line inference do not require an HTTP server.
The HTTP clients use only the Python standard library; you can copy them to
another machine and run them with `python3`.

For Python API and HTTP examples, first edit the model paths or request parameters
at the top of the files. The benchmark accepts command-line arguments instead.
Inference accepts unquantized Hugging Face models or models already
converted to MLX format. Original compressed-tensors INT4/INT8 models must be
converted first.

The Python examples do not automatically read a configuration file. The three
inference shell examples read [inference.json](../../example/inference.json)
and override its values with the parameters shown in each script, as explained
in the command-line section below. Your Python code can also read the file
with `GenerationConfig.from_file()`; see the
[Python configuration example](#reusing-configuration-in-python).

## 1. Model conversion in Python

Edit `SOURCE` and `OUTPUT` in
[python_convert.py](../../example/python_convert.py), then run it directly.
`SOURCE` points to the original model. `OUTPUT` must be a new directory that
does not exist, and neither directory may contain the other.

```sh
.venv/bin/python example/python_convert.py
```

The script explicitly passes `quantization="mixed"`, `w8_backend="native"`,
and `chunk_rows=256`. To retain BF16 precision in the main model when the
input is BF16, change `quantization` to `"none"`. To inspect the structure
without writing output, replace the conversion call with:

```python
report = convert_to_mlx(source=SOURCE, inspect_only=True)
```

The returned `report` is a dictionary you can inspect or write to application
logs. Run `spark-mlx-convert --help` for all conversion options.
This release covers conversion and inference for models without draft weights.

## 2. Complete and streaming generation in Python

Edit `MODEL_PATH` and `MESSAGES` at the top of both files. They use the same
input, a 256-token output limit, temperature 0, thinking disabled, and unquantized
KV caches retaining model precision (BF16 KV for BF16 models). They differ only
in how your code receives the result.

| Python API | Return value | Use in application code |
| --- | --- | --- |
| `generate()` | A complete `str` | Wait for generation to finish, then save, process, or return the text |
| `stream_generate()` | An iterator yielding response objects | Use `response.text` in a loop to update a UI or send incremental output |

Both functions are exported by
[spark_mlx_llm/__init__.py](../../spark_mlx_llm/__init__.py) and implemented in
[generation.py](../../spark_mlx_llm/generation.py). There is one generation
command, `spark-mlx-generate`; there is no separate
`spark-mlx-stream-generate` command. `generate(verbose=True)` can print text
to the terminal during generation, but it still returns a complete string.
Use `stream_generate()` when application code needs to consume text as it
is generated.

```sh
.venv/bin/python example/python_generate.py
.venv/bin/python example/python_stream_generate.py
```

Parameters appear directly in their respective calls: model and operator
settings go to `load()`, thinking settings to
`tokenizer.apply_chat_template()`, sampling settings to `make_sampler()`,
and output length, KV, and prompt chunking settings to the generation function.
To enable thinking, pass `enable_thinking=True, reasoning_effort="low"` to
the chat template. Generated text may contain thinking markers.

### Benchmark prefill/decode at different input lengths

[`benchmark.py`](../../example/benchmark.py) is a standalone script using the
project's inference APIs. It does not depend on `benchmark/` or
`spark_mlx_llm/benchmark/`.

```sh
.venv/bin/python example/benchmark.py --model /path/to/converted-mlx-model \
  --lengths 128 512 2048 8192 --decode-tokens 256 --runs 3
# Explicitly enable INT8 KV quantization and change the prefill chunk size:
.venv/bin/python example/benchmark.py --model /path/to/converted-mlx-model \
  --lengths 512 2048 --kv-bits 8 --prefill-step-size 256
```

Defaults are batch size 1, input lengths of 128/512/2048/8192 tokens, and 256
output tokens. Each length gets one full warmup followed by three measured runs;
the table reports median prefill and decode tok/s for each length. The model
loads once, and each generation uses a fresh KV cache. Inputs are seeded random
tokens without a chat template, intended for throughput measurement. Greedy
sampling with EOS stopping disabled keeps output lengths fixed.

Results use MLX-LM's `prompt_tps` / `generation_tps` directly: prefill timing ends
at the first output token, and decode retains upstream token counting and
asynchronous pipeline timing. Model loading, input preparation, and warmup are
excluded. W8 uses `native`, matching the other Python examples; KV caches default
to model precision without quantization, with 512-token prefill chunks.
Use `--help` for all KV options.

See [Benchmark usage and metrics](benchmark.md) for the full argument table,
measurement procedure, timing formulas, and guidance on interpreting results.

## 3. The four command-line entry points

The shell examples accept model paths through environment variables or
command-line arguments, and each supports `--help`. They use the project's
`.venv/bin/python`; set `SPARK_PYTHON` to select another interpreter with
this package installed. After reading the shared JSON file, the three
inference scripts explicitly disable thinking and set output limits of
256, 512, and 2048 tokens for generation, chat, and server respectively.
Append arguments to override the example values.

```sh
export SPARK_MODEL="/path/to/converted-mlx-model"
export SPARK_SOURCE="/path/to/source-model"

bash example/cli_convert.sh --source "$SPARK_SOURCE" --inspect
# Convert with explicit source, output, quantization, w8-backend, and chunk-rows.
# Use a new output directory, separate from the earlier Python conversion.
bash example/cli_convert.sh \
  --source "$SPARK_SOURCE" --output /path/to/another-new-mlx-model \
  --quantization mixed --w8-backend native --chunk-rows 256

# The script supplies a prompt, output limit, sampling settings, and KV settings.
bash example/cli_generate.sh
# Override the example parameters as needed.
bash example/cli_generate.sh --prompt 'Hello, please introduce yourself.' --max-tokens 128

bash example/cli_chat.sh

bash example/cli_server.sh
```

In terminal chat, enter `q` to quit, `r` to reset the conversation cache, or
`h` for help. The server listens on `127.0.0.1:8080` by default; append
`--host` or `--port` to change it, and press Ctrl-C to stop it. While the
server is running, use another terminal for the HTTP examples below.

Chat and server retain the upstream KV and prompt chunking behavior and do
not accept all generation options. For example, do not pass
`--w8-qmv-backend` or a configuration file containing Spark KV fields to them.

## 4. HTTP chat, streaming chat, and text completion

Start `cli_server.sh` first. Both HTTP examples define `BASE_URL`, `TIMEOUT`,
and the `PAYLOAD` request dictionary at the top of the file. Edit these values
and run the script; it does not accept command-line arguments.

```sh
# Check server health and list models.
curl --fail-with-body http://127.0.0.1:8080/health
curl --fail-with-body http://127.0.0.1:8080/v1/models

# Chat; set stream to True in PAYLOAD to try an SSE response.
.venv/bin/python example/http_chat.py

# Continue a raw prompt without applying a chat template.
.venv/bin/python example/http_completions.py
```

`BASE_URL` must not include `/v1`. The request's `model="default_model"`
selects the model supplied when starting the server; the client does not
need a local model directory. `http_chat.py` requests up to 2048 tokens by
default, and `http_completions.py` requests 128. Both set temperature to 0.
These request fields override server startup values. The chat example
disables thinking in `chat_template_kwargs`; edit `enable_thinking` and
`reasoning_effort` there to enable it, or remove the entire
`chat_template_kwargs` field to inherit the server defaults.

HTTP answer text is written to stdout. Thinking text, the finish reason,
and token usage for non-streaming responses are written to stderr.
`finish_reason=length` means generation reached the output limit and the
answer may be incomplete. `http_chat.py` places thinking parameters in
`chat_template_kwargs` and handles SSE `delta.content`, `delta.reasoning`,
and `[DONE]` events. For multiple turns, include previous messages in the
`messages` list; the server does not maintain the full conversation for
the client.

## 5. Shared inference settings

Generate, chat, and server share
[GenerationConfig](../../spark_mlx_llm/generation.py) for sampling, output
length, and thinking settings. These are the interface defaults; the example
scripts can explicitly override them.

| Configuration field | Command-line option | Default |
| --- | --- | --- |
| `max_tokens` | `--max-tokens` / `-m` | `512` |
| `temperature` | `--temp` / `--temperature` | `0.0` |
| `top_p` | `--top-p` | `1.0` |
| `top_k` | `--top-k` | `0` |
| `min_p` | `--min-p` | `0.0` |
| `enable_thinking` | `--thinking` / `--no-thinking` | Preserve the model template default |
| `reasoning_effort` | `--reasoning-effort low/high` | Preserve the model template default |
| `chat_template_kwargs` | `--chat-template-args '{...}'` | Empty object; forward other template arguments |

All three commands accept `--config example/inference.json`, but the commands
do not load that file automatically. File paths are relative to the working
directory; the example shell scripts explicitly pass a configuration path
resolved from their own location. Python `generate()` / `stream_generate()`
also default to at most `512` output tokens. The two Python examples explicitly
set `256`; the shared JSON selects `2048` and low-effort thinking.

Precedence is **code defaults → configuration file → explicit command-line
arguments → individual HTTP request**. Template arguments merge by field.
Within a layer, dedicated thinking fields take precedence over the same fields
in `chat_template_kwargs`. Setting thinking fields to JSON `null` in a
configuration file restores the model template defaults.

Unknown configuration fields and invalid types raise errors. `max_tokens`
must be a positive integer and `top_k` a nonnegative integer. Sampling values
must be finite, with `temperature >= 0` and `top_p/min_p` in `[0, 1]`.
A value of `0` disables the corresponding `top_k/top_p/min_p` filter.
`messages/tools/tokenize/add_generation_prompt` are managed by the caller
and cannot appear in `chat_template_kwargs`.

### KV caching and prompt chunking

These fields apply to this package's Python generation functions and to the
`spark-mlx-generate` configuration file:

| Configuration field | Generate command-line option | Default and accepted values |
| --- | --- | --- |
| `prefill_step_size` | `--prefill-step-size` | `512`; a positive integer |
| `kv_bits` | `--kv-bits` | JSON `null` / CLI `none` by default, retaining model precision; explicitly set `4/8` to enable quantization |
| `kv_group_size` | `--kv-group-size` | `32`; accepts `32/64/128` |
| `quantized_kv_start` | `--quantized-kv-start` | `0`; a nonnegative integer |
| `kv_quantization_scope` | `--kv-quantization-scope` | `full-attention`; quantizes only full-attention layers when enabled; also accepts `all` |

All entry points default to unquantized KV caches: BF16 models use BF16 KV.
`kv_bits=None` retains model precision without forcing other dtypes to BF16.
Chat and server reject these KV fields in `--config` and retain upstream KV
and prompt chunking behavior. Operator settings belong to `load()` or the
model configuration, not the inference JSON. Run `spark-mlx-generate --help`
for operator switches. Unavailable extension implementations use the native
MLX / MLX-LM path. Generate-specific options such as `--device`, `--dtype`,
`--seed`, and `--ignore-chat-template`
are listed in that command's `--help`. `--ignore-chat-template` skips the
entire template, is not a substitute for disabling thinking, and cannot be
combined with template settings.

### Reusing configuration in Python

To share a JSON configuration in application code, use the following calls.
The `model`, `tokenizer`, and `MESSAGES` come from the earlier generation example:

```python
from mlx_lm.sample_utils import make_sampler
from spark_mlx_llm import GenerationConfig, generate

config = GenerationConfig.from_file("example/inference.json")
config = config.overlay({"max_tokens": 1024, "enable_thinking": False})
prompt = tokenizer.apply_chat_template(
    MESSAGES,
    tokenize=False,
    add_generation_prompt=True,
    **config.template_kwargs(),
)
text = generate(
    model,
    tokenizer,
    prompt=prompt,
    max_tokens=config.max_tokens,
    sampler=make_sampler(**config.sampler_kwargs()),
    **config.kv_kwargs(),
)
```

`GenerationConfig.from_dict({...})` strictly checks field names.
`from_options({...})` extracts known configuration fields from mixed options
and still validates their values. `chat_template_kwargs` is a read-only
snapshot with input and nested values isolated from external objects. Use
`overlay()` to create an updated configuration and `template_kwargs()` to
obtain an independently editable dictionary. For direct upstream calls,
`config.prepare_kv_kwargs(model, ...)` prepares the cache arguments.

### Thinking modes

The example Spark model's template supports `enable_thinking=True` with
`reasoning_effort="low"` or `"high"`. Disable thinking with the boolean
`False`, not a string. `medium` and `none` are not valid effort values.
Setting the effort alone does not enable thinking if it has been disabled.
For HTTP requests, put both fields inside `chat_template_kwargs`; the current
server does not forward a top-level `reasoning_effort` field to the template.

`max_tokens` limits the total length of thinking and the final answer.
`low/high` do not specify fixed thinking-token budgets. These conventions
come from the model's chat template and do not apply to all MLX models.

## 6. How the entry points relate to upstream MLX LM

The four command implementations live in
[commands/](../../spark_mlx_llm/commands/); their names are declared in
`[project.scripts]` in [pyproject.toml](../../pyproject.toml).

| Entry point | Loading and execution | Configuration scope |
| --- | --- | --- |
| Python `load()` + `generate()` / `stream_generate()`, and `spark-mlx-generate` | Project loader and generation wrappers, with generation delegated to MLX LM | Apply Spark KV and prompt chunking defaults |
| `spark-mlx-chat` / `spark-mlx-server` | Register the model, then use upstream loading, caching, generation, and serving with shared inference settings | Share sampling and thinking settings; retain their upstream KV behavior |
| `spark-mlx-convert` / `convert_to_mlx()` | Project CPU conversion implementation | Offline conversion without importing MLX inference modules |

[registration.py](../../spark_mlx_llm/registration.py) prefers an installed
`mlx_lm.models.spark2_5`. If absent, it registers this project's model in the
current process without modifying `mlx_lm` on disk. Chat adapts the upstream
interactive loop; server forwards settings upstream and retains MLX LM's HTTP
routes, response formats, and scheduling. Neither entry point calls this
package's `load()` or generation wrappers.

The project's [load()](../../spark_mlx_llm/loader.py) explicitly selects the
Spark model and accepts `~/...` local paths or `org/model` repository IDs.
A missing explicit local path fails immediately. Dtype and operator overrides
are validated before downloading or loading weights. `load()` itself does
not configure generation KV precision.

Generation wrappers default to retaining KV precision in all layers (BF16 KV for
BF16 models) and processing prompts in chunks of 512 tokens. Explicitly setting
`kv_bits=8` enables INT8/G32 quantization for full-attention KV by default,
preserving model precision and rotation state in sliding-window caches. Direct `mlx_lm.generate`
calls retain upstream defaults. Model operator optimizations and generation
cache settings are controlled separately. Command option sets also differ;
use `--help` in the same environment for the complete list.

## 7. Tool-call parsing and device selection

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
