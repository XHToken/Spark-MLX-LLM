# Model conversion and quantization

[Project overview](../../README.md) · [简体中文](../zh/conversion.md)

`spark-mlx-convert` converts local Spark BF16 or compressed-tensors INT4
checkpoints to MLX format. Conversion runs on CPU in chunks, without loading
the model for inference. Its implementation depends only on NumPy.

## 1. Supported inputs

Prepare a local model directory containing `config.json`, single-file or sharded
Safetensors weights, and tokenizer files. The converter reads layer counts,
hidden dimensions, attention heads, and vocabulary size from the configuration.
Download the published Spark-X2.6-4B BF16 or INT4 model using the links in the
[README](../../README.md#1-download-a-model).

**Convert compressed-tensors checkpoints before using them with
`spark-mlx-generate`, `spark-mlx-chat`, or `spark-mlx-server`.** Pass the converted
output directory to `--model`.

If the input contains floating-point `draft_model.*` tensors, the converter
preserves their names, data, and draft configuration without quantizing them.
The inference examples in this guide cover models without draft weights.

## 2. Command-line conversion

Follow the [environment guide](environment.md) first. Run commands from the
repository root and replace `/path/to/...` with your local paths.

Inspect configuration and tensor metadata without writing output:

```sh
.venv/bin/spark-mlx-convert \
    --source /path/to/Spark-X2.6-4B-INT4 \
    --inspect
```

`--inspect` checks structure and metadata; it does not scan all weight values
for NaNs or precision errors.

Convert the compressed INT4 model with the default mixed-precision scheme:

```sh
.venv/bin/spark-mlx-convert \
    --source /path/to/Spark-X2.6-4B-INT4 \
    --output /path/to/Spark-X2.6-4B-MLX
```

Quantize a BF16 model with the same scheme:

```sh
.venv/bin/spark-mlx-convert \
    --source /path/to/Spark-X2.6-4B \
    --output /path/to/Spark-X2.6-4B-from-BF16-MLX
```

To retain BF16 main-model weights, use `--quantization none` with a BF16 source:

```sh
.venv/bin/spark-mlx-convert \
    --source /path/to/Spark-X2.6-4B \
    --output /path/to/Spark-X2.6-4B-BF16-MLX \
    --quantization none
```

`--output` must be a new directory, must not be a symlink, and must neither
contain nor be contained by the source directory. Conversion never overwrites
the source or an existing output directory. `--chunk-rows` defaults to `256`;
smaller values reduce temporary memory use during CPU conversion.

## 3. Default mixed-precision scheme

`--quantization mixed` is the default. The main model uses the following weight
formats. The compute column assumes W8A8 is available and enabled; see the
backend choices below for floating-point activation paths.

| Module | Weight quantization | Scale / affine bias dtype | Compute |
| --- | --- | --- | --- |
| QKV and MLP gate/up/down | Asymmetric INT4, group size 32 | BF16 | W4A16 |
| Attention `out_proj` | Symmetric INT8, one scale per output channel | BF16 | W8A8, dynamic per-token INT8 activations |
| `lm_head` | Symmetric INT8, one scale per vocabulary row | FP32 | W8A8, dynamic per-token INT8 activations |
| Embedding, `g_proj`, and RMSNorm | BF16, unquantized | — | Floating point |

Compressed input retains its existing integer weight codes without requantizing
them. MLX stores affine weights as UINT32-packed `weight`, `scales`, and `biases`.
W8 per-channel scales repeat across G32 groups, with `biases = -128 * scales`.
The `lm_head` scales remain FP32.

Scale and affine-bias conversion can introduce rounding; the conversion manifest
records coefficient rounding errors. Preserving integer codes does not imply
bit-for-bit identical inference results.

BF16 input uses per-group or per-channel min/max quantization with
round-to-nearest, without calibration data. This differs from the original
compressed model's GPTQ/MSE procedure, so evaluate quality on your target tasks.
When the source ties embedding and `lm_head`, mixed conversion creates a separate
INT8 `lm_head` and keeps the embedding in BF16. `--quantization none` requires
BF16 input; it does not dequantize an INT4 checkpoint.

## 4. W8 execution backend

`--w8-backend` selects the inference policy recorded in `config.json` and
`conversion_manifest.json`. It applies to W8 `out_proj` and `lm_head` under
mixed quantization and does not change their stored weight codes or scales.

| Value | Behavior |
| --- | --- |
| `w8a8` (default) | Dynamic per-token INT8 activations on the supported Metal path; requires a compatible `spark-mlx-ext`, Apple M5 or newer, and macOS >= 26.2 |
| `w8a16` | Floating-point activations; uses the optimized per-channel W8A16 path for eligible single-row GPU inputs |
| `native` | Floating-point activations using ordinary MLX affine operators |

Unavailable optimized backends fall back to MLX affine operators with
floating-point activations. W8A8 and floating-point activation paths can produce
different numerical results. All conversions still execute on CPU.

To explicitly select the native path:

```sh
.venv/bin/spark-mlx-convert \
    --source /path/to/Spark-X2.6-4B-INT4 \
    --output /path/to/Spark-X2.6-4B-native-MLX \
    --w8-backend native
```

The output sets `quantize_activations=false` to disable MLX LM's generic
QQLinear path. When selected and available, this project's W8A8 backend
performs activation quantization separately.

## 5. Python API and arguments

The CLI and Python API use the same conversion implementation:

| CLI option | Python argument | Default and meaning |
| --- | --- | --- |
| `--source` | `source` | Required local BF16 or compressed-tensors model directory |
| `--output` | `output` | Required for conversion; a new directory disjoint from the source |
| `--inspect` | `inspect_only` | `False`; inspect metadata without writing output or scanning all weight values |
| `--quantization` | `quantization` | `"mixed"`; `"none"` retains BF16 main-model weights and requires BF16 input |
| `--w8-backend` | `w8_backend` | `"w8a8"`; also accepts `"w8a16"` or `"native"` |
| `--chunk-rows` | `chunk_rows` | `256`; a positive number of rows processed per CPU chunk |

```python
from spark_mlx_llm import convert_to_mlx

report = convert_to_mlx(
    source="/path/to/Spark-X2.6-4B-INT4",
    output="/path/to/Spark-X2.6-4B-python-MLX",
    quantization="mixed",
    w8_backend="native",
    chunk_rows=256,
)
print(report["quantized_module_counts"])
```

`source` and `output` accept strings or `pathlib.Path`. For inspection, call
`convert_to_mlx(source, inspect_only=True)` without an output directory.
The return value is a JSON-serializable report containing `status`,
`source_format`, `quantized_module_counts`, `lm_head_untied`, and draft metadata.
Completed conversions also include `coefficient_rounding` and save the full
report as `conversion_manifest.json`.

Invalid or unsupported values raise `ValueError`; missing files raise
`FileNotFoundError`; an existing output directory raises `FileExistsError`.
Unsupported group-index/permutation tensors, activation schemes, weight
transforms, and sparsity are rejected.

The module entry point is `python -m spark_mlx_llm.commands.convert`;
`python -m spark_mlx_llm.conversion` remains a compatibility entry point.
Use `--source` and `--output`; upstream options such as `--hf-path`, `--mlx-path`,
and `--q-bits` are not supported. See the
[Python conversion example](examples.md#1-model-conversion-in-python) for a runnable script.

## 6. Output and inference

The output contains `model.safetensors`, `model.safetensors.index.json`,
`config.json`, tokenizer files, and `conversion_manifest.json`. If draft tensors
are present, they are stored separately in `model-draft.safetensors` and included
in the index. Output is prepared in a temporary directory and published after
validation; a failed conversion cleans up the temporary directory.

Run the converted main model:

```sh
.venv/bin/spark-mlx-generate \
    --model /path/to/Spark-X2.6-4B-MLX \
    --prompt '你好，请介绍一下自己。' \
    --max-tokens 256 \
    --no-thinking
```

KV precision is controlled separately during generation. All generation entry
points default to unquantized KV caches, so BF16 models use BF16 KV regardless
of the INT4/INT8 weight scheme. See the [interface guide](examples.md) for
interactive chat, Python generation, and HTTP serving.
