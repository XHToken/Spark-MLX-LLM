# 环境配置

[项目首页](../../README.zh.md) · [English](../en/environment.md)

本文介绍 Spark-MLX-LLM 的推理和开发环境配置。统一使用 uv 管理 Python、
依赖和仓库内的扩展，推荐 Python 3.12。下列命令适用于 macOS / Linux 的
终端；获取源码后，所有命令都在仓库根目录执行。

## 1. 系统要求

| 环境 | 要求 |
| --- | --- |
| Python | 推荐原生 CPython 3.12；主包声明支持 Python >= 3.10，实际安装还需满足依赖包要求 |
| macOS / Metal | Apple Silicon、原生 `arm64` Python、macOS >= 26.2，以及包含 Metal 4 编译器和 MetalPerformancePrimitives 头文件的 Xcode 工具链 |
| Linux / CUDA 12 | NVIDIA GPU（SM >= 7.5）、驱动 >= 550.54.14、CUDA Toolkit >= 12.0，以及 glibc >= 2.35 |
| Linux / CUDA 13 | 使用兼容 CUDA 13 的 GPU、工具链与驱动；驱动 >= 580，或满足上游要求的 CUDA 兼容包 |
| 模型文件 | 单独准备 Spark2_5 checkpoint；安装 Python 包不会下载模型权重 |

Linux 和 CUDA 要求参见 [MLX 官方安装说明](https://ml-explore.github.io/mlx/build/html/install.html)。
Apple Silicon 上的安装流程会自动编译扩展，请先准备好上述 Xcode 工具链。

推理需要 Metal 或 CUDA GPU；模型转换使用 CPU。

在 Apple Silicon 上检查终端架构，输出应为 `arm64`：

```sh
uname -m
```

## 2. 仓库中的环境配置文件

| 文件 | 作用 |
| --- | --- |
| [`.python-version`](../../.python-version) | 默认开发 Python 版本，目前为 `3.12`，不固定补丁版本 |
| [`pyproject.toml`](../../pyproject.toml) | 包信息、Python 兼容范围、后端和命令入口；通过 `tool.uv.sources` 指定本地扩展目录 |
| [`uv.lock`](../../uv.lock) | 依赖锁文件；当前包含 MLX 0.32.4、MLX LM 0.31.3 和本地 spark-mlx-ext 0.2.0 |
| [`uv.toml`](../../uv.toml) | uv 的包索引配置，目前默认使用阿里云 PyPI 镜像 |
| `.venv/` | 本机虚拟环境，安装时生成，不提交到 Git |

`.python-version` 选择开发环境的 Python，`requires-python` 声明包的兼容范围，
两者作用不同。主包仍声明支持 Python >= 3.10；扩展自动安装要求 Python >= 3.11。

## 3. 使用 uv 安装

先按 [uv 安装说明](https://docs.astral.sh/uv/getting-started/installation/) 安装 uv。
已经使用 Homebrew 的 macOS 用户可以运行：

```sh
brew install uv
```

获取源码；如果已有本地仓库，直接进入对应目录即可：

```sh
git clone https://github.com/XHToken/Spark-MLX-LLM.git
cd Spark-MLX-LLM
uv python install 3.12
```

根据运行平台，**只执行下面一组安装命令**。

### Apple Silicon / Metal

先安装包含 Metal 4 编译器的 Xcode，并完成所需组件的安装。确认当前工具链可用，
然后同步环境：

```sh
xcrun --find metal
xcrun --sdk macosx --show-sdk-version
uv sync --locked
```

此命令会从 `extensions/spark-mlx-ext` 自动编译并以可编辑方式安装扩展，
无需单独安装 wheel。uv 会在隔离的构建环境中安装 CMake、nanobind 等构建依赖。
扩展要求 macOS >= 26.2；具体工具链和版本要求见下文。

### Linux / CUDA 12

```sh
uv sync --locked --extra cuda12
```

### Linux / CUDA 13

```sh
uv sync --locked --extra cuda13
```

Linux / CUDA 环境不会安装 Apple Silicon 扩展。

uv 会创建 `.venv`，以可编辑方式安装本项目及运行依赖。
选择对应后端即可，不要使用 `--all-extras` 同时安装所有后端。

`--locked` 要求锁文件与项目配置一致，遇到不一致会报错而不改写锁文件。
具体行为见 [uv 同步说明](https://docs.astral.sh/uv/concepts/projects/sync/)。

下文直接使用 `.venv/bin/python`，无需先激活环境。若希望直接使用 `python`
和 `spark-mlx-generate` 等命令，可以激活环境，并将编辑器解释器设为
仓库下的 `.venv/bin/python`：

```sh
source .venv/bin/activate
python --version
```

## 4. 验证基础环境

检查解释器、架构、依赖版本、后端可用性，并执行一个 GPU 运算；此步骤不需要模型：

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

Metal 环境应显示 `metal available: True`，CUDA 环境应显示
`cuda available: True`。推理需要可用的 GPU 后端。

## 5. Apple Silicon 扩展

`spark-mlx-ext` 是主项目在 macOS `arm64`、Python >= 3.11 上的条件依赖，
由 `uv sync` 从同仓库源码自动安装。Linux 和 Python 3.10 环境不会安装它。
当前扩展 0.2.0 要求官方
`mlx==0.32.4`，当前 Metal 二进制配置要求 `mlx-metal==0.32.4` 和
macOS >= 26.2。已有的 `cp312-cp312` wheel 适用于标准 CPython 3.12 / arm64；
W8A8 算子执行还要求 Apple M5 或更新的芯片。Linux / CUDA 环境跳过此节。

源码构建要求包含 Metal 4 编译器和 MetalPerformancePrimitives SDK 头文件的
Xcode 工具链；不需要单独克隆 MLX 源码。安装命令和支持的算子见
[扩展 README](../../extensions/spark-mlx-ext/README.zh.md)。

安装后重启使用该环境的 Python / Notebook 进程，并验证：

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

扩展已纳入锁定依赖，普通 `uv sync --locked` 会保留它，无需
`--inexact`。修改 C++ / Metal 源码后需要重新构建；在 Apple Silicon 上执行：

```sh
uv sync --locked --reinstall-package spark-mlx-ext
```

重新安装后重启 Python / Notebook。模型配置选择的优化算子会在对应扩展实现可用时启用，
否则回退到 MLX / MLX-LM 原生路径。W8A8 回退保留打包的 W8 权重，
使用浮点激活，因此与 W8A8 激活量化的数值结果不完全等价。

## 6. 运行一次模型推理

准备本地 Spark2_5 模型目录，并替换下面的路径。`compressed-tensors` 格式的
INT4 / INT8 checkpoint 需要先[转换为 MLX 格式](examples.md#1-python-模型转换)。

```sh
.venv/bin/spark-mlx-generate \
  --model /path/to/spark2_5-mlx \
  --device gpu \
  --prompt '你好，请简单介绍自己。' \
  --max-tokens 32 \
  --temp 0 \
  --seed 0 \
  --w8-qmv-backend native \
  --no-fused-quantized-sdpa
```

这里显式选择原生 W8 路径并关闭扩展 INT8 KV 融合，便于先验证基础推理。
通过 `spark-mlx-generate --help` 查看算子开关，更多用法见[示例](examples.md)。

## 7. 常见问题

| 现象 | 处理方式 |
| --- | --- |
| 找不到 `metal` 编译器或 MetalPerformancePrimitives 头文件 | 安装完整的兼容 Xcode 工具链及所需组件，检查 `xcrun --find metal` 和当前 SDK；仅有命令行工具可能不足 |
| 找不到 `spark-mlx-generate` | 使用 `.venv/bin/spark-mlx-generate`，或先激活环境；确认已安装本项目 |
| Apple Silicon 找不到匹配的 MLX wheel | 检查 Python 架构是否为 `arm64`，以及 Python / macOS 版本是否匹配 wheel；避免通过 Rosetta 使用 x86 Python |
| Linux 找不到匹配的 MLX wheel | 检查 CPU 架构、glibc 版本及所选后端；当前 Linux wheel 要求 glibc >= 2.35 |
| CUDA 后端不可用 | 先用 `nvidia-smi` 检查驱动与设备，再确认选择了匹配的 `cuda12` 或 `cuda13` extra |
| `--locked` 提示锁文件过期 | 确认 `pyproject.toml`、`uv.toml`、`uv.lock` 来自同一版本；主动修改依赖后才运行 `uv lock` 并检查差异 |
| 扩展导入失败或未安装 | 确认使用本仓库的 uv 流程，以及 macOS `arm64` / Python >= 3.11；检查工具链，按上面的命令重新构建并重启 Python |
| 阿里云镜像无法访问 | 可将 `uv.toml` 的索引 URL 改为 `https://pypi.org/simple`，运行 `uv lock` 并检查差异，再执行 `uv sync --locked`（Linux 加相应后端 extra）；原锁文件包含镜像 URL |
