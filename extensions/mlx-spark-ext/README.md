# mlx-spark-ext

Optional Spark QKV/RoPE and W8A8 TensorOps extension for official MLX 0.32.4.
The Python package is `mlx_spark_ext`; its compiled module is
`mlx_spark_ext._ext`. This directory is an independent Python package with
its own version, build dependencies, and MIT license.

The QKV/RoPE operation composes existing MLX operations in C++. The W8A8
operations use custom Metal 4 INT8 TensorOps kernels adapted from Cider.
Both link to the official MLX library installed in the same Python environment;
building this package does not rebuild MLX or replace `mlx.core`.
See [third-party sources and local changes](THIRD_PARTY.md).

## Install a wheel

See the main [installation instructions](../../README.md#optional-spark-c-extension-on-apple-silicon)
for downloads, compatibility requirements, and model integration verification.
The current `0.2.0` wheel targets standard CPython 3.12, Apple Silicon, official
`mlx==0.32.4` / `mlx-metal==0.32.4`, and macOS 26.2 or newer. It was tested on
macOS 26.5.1. W8A8 execution requires Apple M5 or newer. The QKV/RoPE API remains
usable on other supported Apple Silicon GPUs. A different Python ABI requires
a separate build.

Wheel assets belong in the project's
[GitHub Releases](https://github.com/XHToken/Spark-MLX-LLM/releases).
Until a matching asset is published, build from this source directory.
Do not assume that `pip install mlx-spark-ext` resolves to a published project
package on PyPI; use an explicit wheel path or this source checkout.

## Build from source

Use an Apple Silicon Mac with a compatible official MLX 0.32.4 backend,
CPython 3.11 or newer, and an Xcode toolchain with the Metal 4 compiler and
MetalPerformancePrimitives SDK headers. Python 3.12 is the tested
configuration. Building with another Python version still requires official
MLX wheels for that interpreter. The backend's macOS 26.2 minimum is not
removed by rebuilding only this extension.

Run these commands from the **Spark-MLX-LLM repository root**, after creating
the project's `.venv` as described in the main README. If the compiler is
missing, install it with `xcode-select --install` and complete that installation
before building.

Keep build output in the workspace's `.build` directory. You may set
`MLX_SPARK_BUILD_ROOT` to another absolute path before running these commands:

```sh
export MLX_SPARK_BUILD_ROOT="${MLX_SPARK_BUILD_ROOT:-$(pwd)/../.build/mlx-spark-ext}"
export PIP_CACHE_DIR="$MLX_SPARK_BUILD_ROOT/cache/pip"
export PYTHONPYCACHEPREFIX="$MLX_SPARK_BUILD_ROOT/python-cache"
mkdir -p "$MLX_SPARK_BUILD_ROOT/tmp" "$MLX_SPARK_BUILD_ROOT/wheels"
export TMPDIR="$MLX_SPARK_BUILD_ROOT/tmp"

DEBUG=0 \
MACOSX_DEPLOYMENT_TARGET=26.2 \
CMAKE_ARGS='-DCMAKE_OSX_DEPLOYMENT_TARGET=26.2' \
.venv/bin/python -m pip wheel \
  --no-deps \
  --wheel-dir "$MLX_SPARK_BUILD_ROOT/wheels" \
  ./extensions/mlx-spark-ext
```

Pip installs the build requirements from `pyproject.toml` into an isolated
environment. These include setuptools, wheel, CMake, nanobind 3.1.0, and MLX
0.32.4. `--no-deps` here skips building runtime dependency wheels; it does not
skip the build requirements. CMake also compiles and bundles
`spark_w8a8.metallib`; it does not compile any MLX core libraries.

Install the generated wheel, using its actual filename. For Python 3.12:

```sh
.venv/bin/python -m pip install \
  'mlx==0.32.4' 'mlx-metal==0.32.4' \
  "$MLX_SPARK_BUILD_ROOT/wheels/mlx_spark_ext-0.2.0-cp312-cp312-macosx_26_0_arm64.whl"
```

Use a new build root for a different Python ABI, SDK, or source checkout to
avoid reusing incompatible CMake caches. When rebuilding an already installed
extension with the same version, first ensure the official MLX versions above
are installed, then reinstall only the extension:

```sh
.venv/bin/python -m pip install --force-reinstall --no-deps \
  "$MLX_SPARK_BUILD_ROOT/wheels/mlx_spark_ext-0.2.0-cp312-cp312-macosx_26_0_arm64.whl"
```

Restart the inference process or notebook kernel after installation. The
current extension pins MLX because its C++ ABI is not stable across releases;
upgrading MLX requires a matching rebuild and validation. See
[MLX's extension documentation](https://ml-explore.github.io/mlx/build/html/dev/extensions.html).

## Validate

From the repository root, install the test dependency and run the tests against
the installed wheel:

```sh
.venv/bin/python -m pip install numpy
.venv/bin/python -m unittest discover \
  -s extensions/mlx-spark-ext/tests -p 'test_*.py' -v
```

For model integration tests, install the project's test dependencies and run:

```sh
.venv/bin/python -m pytest \
  tests/test_model.py \
  -o "cache_dir=$MLX_SPARK_BUILD_ROOT/pytest-cache"
```

## W8A8 API

Use existing symmetric INT8 weights and FP32 per-output-channel scales:

```python
import mlx.core as mx
from mlx_spark_ext import W8A8Linear, is_w8a8_available, w8a8_linear

assert is_w8a8_available(), "W8A8 requires Apple M5+ and macOS 26.2+"

# Existing checkpoint tensors; no weight re-quantization occurs.
# weight: int8 [N, K]; scales: float32 [N] or [N, 1].
y = w8a8_linear(x, weight, scales)
layer = W8A8Linear(weight, scales)  # optional floating bias[N]
y = layer(x)
mx.eval(y)
```

`x` has shape `[..., K]` and dtype FP16, BF16, or FP32. The output has shape
`[..., N]` and the same dtype. K must be a multiple of 32 in `[32, 131040]`;
M and N need not align to tile sizes. Scales must be finite and positive,
and activations must be finite. Noncontiguous arrays, explicit GPU streams,
and ordinary `mx.compile` calls are supported. These are inference operations;
CPU execution, gradients, `vmap`, and shapeless compilation are unsupported.

The operation computes a per-token scale `max(abs(x)) / 127` (1 for a zero
row), rounds ties to even, clips to `[-127, 127]`, and multiplies INT8
activations by INT8 weights with INT32 accumulation. The epilogue applies
activation and weight scales plus the optional bias in FP32, then casts to
the output dtype. Single-token calls also use this W8A8 path. BF16 inputs
do not pass through FP16. These two behaviors differ from Cider's default
per-channel wrapper.

`int8_matmul_int32(x, weight, *, stream=None)` exposes the underlying raw
`x[M,K] @ weight[N,K].T` operation with two signed INT8 inputs and an INT32
output. It is useful for checking integer arithmetic independently of scaling.

Installing the extension exposes these APIs. It does not automatically replace
Spark's `out_proj` layers or convert compressed-tensors checkpoints. Explicit
layer loading/integration is a separate step. W4A8 and Cider's SDPA are not
included. No Cider Python dependency is required.

## Distribution

Keep the source in `extensions/mlx-spark-ext/` in this repository. The main
`spark-mlx-llm` package and this extension are built and installed separately.
Native binaries, compiler caches, and wheels stay under `.build/` and are not
committed to Git.

For a release, build and test each supported Python/platform combination, then
attach its `mlx_spark_ext-*.whl` to a versioned GitHub Release (for example,
`mlx-spark-ext-v0.2.0`). Attach `SHA256SUMS` and release notes recording the
Python ABI, architecture, minimum macOS version, exact MLX version, and test
results. GitHub's source archive also includes this directory. See
[GitHub's release documentation](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases).

Only publish compatibility combinations that have been tested. In particular,
the `macosx_26_0_arm64` wheel tag alone does not express the official backend's
26.2 minimum. Downloading the wheel requires no C++ build tools on the user's
machine. PyPI can be added later as a distribution channel once the package
name and release process have been established.
