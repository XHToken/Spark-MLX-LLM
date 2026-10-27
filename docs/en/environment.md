# Environment setup

[Project overview](../../README.md) · [简体中文](../zh/environment.md)

This guide covers the inference and development environment for Spark-MLX-LLM.
Use uv to manage Python, dependencies, and the extension in this repository.
Python 3.12 is recommended. Commands target
macOS / Linux shells. After obtaining the source, run all commands from the
repository root.

## 1. System requirements

| Environment | Requirement |
| --- | --- |
| Python | Native CPython 3.12 recommended; the main package declares Python >= 3.10, subject to dependency requirements |
| macOS / Metal | Apple Silicon, native `arm64` Python, macOS >= 26.2, and an Xcode toolchain with the Metal 4 compiler and MetalPerformancePrimitives headers |
| Linux / CUDA 12 | NVIDIA GPU (SM >= 7.5), driver >= 550.54.14, CUDA Toolkit >= 12.0, and glibc >= 2.35 |
| Linux / CUDA 13 | A GPU, toolchain, and driver compatible with CUDA 13; driver >= 580 or a CUDA compatibility package meeting upstream requirements |
| Model files | Obtain a Spark2_5 checkpoint separately; installing the Python package does not download model weights |

See the [MLX installation guide](https://ml-explore.github.io/mlx/build/html/install.html)
for Linux and CUDA requirements. Installation on Apple Silicon automatically
builds the extension, so prepare the Xcode toolchain first.

Inference requires a Metal or CUDA GPU. Checkpoint conversion runs on CPU.

On Apple Silicon, check the terminal architecture; the output should be `arm64`:

```sh
uname -m
```

## 2. Environment configuration files

| File | Purpose |
| --- | --- |
| [`.python-version`](../../.python-version) | Default development Python version, currently `3.12`; does not pin a patch version |
| [`pyproject.toml`](../../pyproject.toml) | Package metadata, Python compatibility, backends, and entry points; `tool.uv.sources` specifies the local extension directory |
| [`uv.lock`](../../uv.lock) | Dependency lockfile; currently includes MLX 0.32.4, MLX LM 0.31.3, and local spark-mlx-ext 0.2.0 |
| [`uv.toml`](../../uv.toml) | uv package index configuration; currently defaults to the Aliyun PyPI mirror |
| `.venv/` | Local virtual environment created during installation; excluded from Git |

`.python-version` selects the development interpreter, while `requires-python`
declares package compatibility. The main package still supports Python >= 3.10;
automatic extension installation requires Python >= 3.11.

## 3. Install with uv

Install uv using the [official instructions](https://docs.astral.sh/uv/getting-started/installation/).
On macOS, if you already use Homebrew:

```sh
brew install uv
```

Obtain the source, or enter your existing local checkout:

```sh
git clone https://github.com/XHToken/Spark-MLX-LLM.git
cd Spark-MLX-LLM
uv python install 3.12
```

Run **only one** of the following installation commands for your platform.

### Apple Silicon / Metal

Install Xcode with the Metal 4 compiler and finish installing its required
components. Check the active toolchain, then synchronize the environment:

```sh
xcrun --find metal
xcrun --sdk macosx --show-sdk-version
uv sync --locked
```

This automatically builds `extensions/spark-mlx-ext` and installs it in editable
mode. No separate wheel installation is needed. uv installs build dependencies
such as CMake and nanobind in an isolated environment. The extension requires
macOS >= 26.2; see the toolchain and compatibility details below.

### Linux / CUDA 12

```sh
uv sync --locked --extra cuda12
```

### Linux / CUDA 13

```sh
uv sync --locked --extra cuda13
```

Linux / CUDA environments do not install the Apple Silicon extension.

uv creates `.venv` and installs this project in editable mode with its runtime
dependencies. Select the appropriate backend instead of installing every backend
with `--all-extras`.

`--locked` requires the lockfile to match the project configuration and reports
an error instead of updating it. See [uv synchronization](https://docs.astral.sh/uv/concepts/projects/sync/).

The following sections use `.venv/bin/python`, so activation is unnecessary.
To use commands such as `python` and `spark-mlx-generate` directly, activate the
environment. Configure your editor to use `.venv/bin/python` in the repository:

```sh
source .venv/bin/activate
python --version
```

## 4. Verify the base environment

Check the interpreter, architecture, dependency versions, and available backends,
then execute a GPU computation. This does not require a model:

```sh
.venv/bin/python - <<'PY'
import platform
import sys
from importlib.metadata import version

import mlx.core as mx
import spark_mlx_llm

print("Python:", sys.version.split()[0])
print("Executable:", sys.executable)
print("Architecture:", platform.machine())
for package in ("spark-mlx-llm", "mlx", "mlx-lm"):
    print(f"{package}: {version(package)}")
for name in ("metal", "cuda"):
    backend = getattr(mx, name, None)
    print(f"{name} available:", bool(backend and backend.is_available()))
mx.set_default_device(mx.gpu)
assert (mx.array([1, 2, 3]) + 1).tolist() == [2, 3, 4]
print("GPU computation: OK")
PY

.venv/bin/spark-mlx-generate --help
```

A Metal environment should report `metal available: True`; a CUDA environment
should report `cuda available: True`. A working GPU backend is required for
inference.

## 5. Apple Silicon extension

`spark-mlx-ext` is a conditional dependency on macOS `arm64` with Python >= 3.11.
`uv sync` installs it automatically from this repository. Linux and Python 3.10
environments skip it. Extension 0.2.0 requires official
`mlx==0.32.4`; the current Metal binary configuration requires
`mlx-metal==0.32.4` and macOS >= 26.2. The existing `cp312-cp312` wheel targets
standard CPython 3.12 / arm64. W8A8 execution additionally requires Apple M5 or
newer. Skip this section on Linux / CUDA.

Building requires an Xcode toolchain with the Metal 4 compiler and
MetalPerformancePrimitives SDK headers. A separate MLX source checkout is
unnecessary. Installation commands and supported operators are listed in the
[extension README](../../extensions/spark-mlx-ext/README.md).

Restart Python / Notebook processes using the environment, then verify:

```sh
.venv/bin/python - <<'PY'
from importlib.metadata import version

import mlx.core as mx
import spark_mlx_ext as ext
from spark_mlx_llm.ops import qkv

assert version("mlx") == version("mlx-metal") == "0.32.4"
assert mx.metal.is_available()
assert qkv.spark_qkv_rope is ext.spark_qkv_rope
print("Extension ready:", version("spark-mlx-ext"))
PY
```

The extension is included in the lockfile. Ordinary
`uv sync --locked` retains it without `--inexact`. After changing
C++ / Metal sources, rebuild it on Apple Silicon:

```sh
uv sync --locked --reinstall-package spark-mlx-ext
```

Restart Python / Notebook processes after reinstalling. Model configuration
selects optimized operators when their extension implementation is available.
Otherwise, inference uses the native MLX / MLX-LM path. W8A8 fallback retains
the packed W8 weights but uses floating-point activations, so it is not
numerically equivalent to W8A8 activation quantization.

## 6. Run model inference

Prepare a local Spark2_5 model directory and replace the path below.
INT4 / INT8 checkpoints in `compressed-tensors` format must first be
[converted to MLX format](examples.md#1-model-conversion-in-python).

```sh
.venv/bin/spark-mlx-generate \
  --model /path/to/spark2_5-mlx \
  --device gpu \
  --prompt 'Hello, please briefly introduce yourself.' \
  --max-tokens 32 \
  --temp 0 \
  --seed 0 \
  --w8-qmv-backend native \
  --no-fused-quantized-sdpa
```

This example explicitly selects the native W8 path and disables extension INT8
KV fusion to verify basic inference first. Run `spark-mlx-generate --help`
for operator switches and see [examples](examples.md) for more usage.

## 7. Troubleshooting

| Symptom | Action |
| --- | --- |
| Missing `metal` compiler or MetalPerformancePrimitives headers | Install a complete compatible Xcode toolchain and required components; check `xcrun --find metal` and the active SDK. Command Line Tools alone may be insufficient |
| `spark-mlx-generate` not found | Use `.venv/bin/spark-mlx-generate` or activate the environment; confirm this project is installed |
| No matching MLX wheel on Apple Silicon | Check for `arm64` Python and compatible Python / macOS versions; avoid x86 Python under Rosetta |
| No matching MLX wheel on Linux | Check CPU architecture, glibc version, and backend selection; current Linux wheels require glibc >= 2.35 |
| CUDA backend unavailable | Check the driver and devices with `nvidia-smi`, then confirm the matching `cuda12` or `cuda13` extra is installed |
| `--locked` reports an outdated lockfile | Confirm `pyproject.toml`, `uv.toml`, and `uv.lock` come from the same revision; run `uv lock` and review its changes only when intentionally changing dependencies |
| Extension import fails or it is not installed | Confirm this repository's uv workflow, macOS `arm64`, and Python >= 3.11; check the toolchain, rebuild using the command above, and restart Python |
| Aliyun mirror unavailable | Change the index URL in `uv.toml` to `https://pypi.org/simple`, run `uv lock` and review the changes, then run `uv sync --locked` (with the appropriate backend extra on Linux). The original lockfile contains mirror URLs |
