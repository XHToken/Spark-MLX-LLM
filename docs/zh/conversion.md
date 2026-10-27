# 模型转换与量化

[项目首页](../../README.zh.md) · [English](../en/conversion.md)

`spark-mlx-convert` 将本地 Spark BF16 或 compressed-tensors INT4 checkpoint
转换为 MLX 格式。转换在 CPU 上分块执行，无需加载模型进行推理，转换实现仅依赖 NumPy。

## 1. 支持的输入

准备包含 `config.json`、单文件或分片 Safetensors 权重，以及 tokenizer 文件的本地模型目录。
转换器从配置读取层数、隐藏维度、注意力头数和词表大小。
已公开的 Spark-X2.6-4B BF16 与 INT4 模型下载链接见
[README](../../README.zh.md#1-模型下载)。

**compressed-tensors checkpoint 必须先转换，再用于 `spark-mlx-generate`、
`spark-mlx-chat` 或 `spark-mlx-server`。** 将转换后的输出目录传给 `--model`。

如果输入包含浮点 `draft_model.*` 张量，转换器会原样保留张量名称、数据和 draft 配置，
不对其量化。本文推理示例仅针对不含 draft 权重的模型。

## 2. 命令行转换

先按[环境配置](environment.md)安装项目，所有命令均在仓库根目录执行，
并将 `/path/to/...` 替换为本地路径。

检查配置与张量元数据，不写入输出：

```sh
.venv/bin/spark-mlx-convert \
    --source /path/to/Spark-X2.6-4B-INT4 \
    --inspect
```

`--inspect` 检查结构和元数据，不扫描完整权重中的 NaN 或精度误差。

使用默认混合精度方案转换已压缩的 INT4 模型：

```sh
.venv/bin/spark-mlx-convert \
    --source /path/to/Spark-X2.6-4B-INT4 \
    --output /path/to/Spark-X2.6-4B-MLX
```

使用相同方案量化 BF16 模型：

```sh
.venv/bin/spark-mlx-convert \
    --source /path/to/Spark-X2.6-4B \
    --output /path/to/Spark-X2.6-4B-from-BF16-MLX
```

如需保留主模型的 BF16 权重，对 BF16 输入使用 `--quantization none`：

```sh
.venv/bin/spark-mlx-convert \
    --source /path/to/Spark-X2.6-4B \
    --output /path/to/Spark-X2.6-4B-BF16-MLX \
    --quantization none
```

`--output` 必须是尚不存在的新目录，不能是符号链接，且不能与源目录相互包含。
转换不会覆盖源模型或已有输出目录。`--chunk-rows` 默认为 `256`，
减小该值可以降低 CPU 转换期间的临时内存占用。

## 3. 默认混合精度方案

`--quantization mixed` 为默认值，主模型采用以下权重格式。
计算列描述 W8A8 可用且启用时的行为；使用浮点激活的路径见下方后端说明。

| 模块 | 权重量化 | scale / affine bias 类型 | 计算 |
| --- | --- | --- | --- |
| QKV 与 MLP gate/up/down | 非对称 INT4，group size 32 | BF16 | W4A16 |
| 注意力 `out_proj` | 对称 INT8，每输出通道一个 scale | BF16 | W8A8，按 token 动态量化为 INT8 激活 |
| `lm_head` | 对称 INT8，每词表行一个 scale | FP32 | W8A8，按 token 动态量化为 INT8 激活 |
| Embedding、`g_proj` 与 RMSNorm | 保留 BF16，不量化 | — | 浮点计算 |

压缩输入保留已有整数权重编码，不重新量化。MLX 使用 UINT32 打包的 `weight`、
`scales` 和 `biases` 表达 affine 权重。W8 的通道 scale 沿 G32 分组重复，
`biases = -128 * scales`；`lm_head` 的 scale 保持 FP32。

scale 与 affine bias 的类型转换可能产生舍入，转换清单会记录系数舍入误差。
保留整数编码不代表推理结果逐位一致。

BF16 输入使用逐组或逐通道的 min/max 和 round-to-nearest 量化，无需校准数据。
该流程不同于原压缩模型的 GPTQ/MSE 量化方法，转换后应根据目标任务评估质量。
如果源模型共享 embedding 与 `lm_head`，混合量化会创建独立的 INT8 `lm_head`，
embedding 保持 BF16。`--quantization none` 仅支持 BF16 输入，不会对 INT4 checkpoint 反量化。

## 4. W8 执行后端

`--w8-backend` 指定记录在 `config.json` 和 `conversion_manifest.json` 中的推理策略，
作用于混合量化下的 W8 `out_proj` 与 `lm_head`，不改变其保存的权重编码或 scale。

| 取值 | 行为 |
| --- | --- |
| `w8a8`（默认） | 在受支持的 Metal 路径上按 token 动态量化为 INT8 激活；需要兼容的 `spark-mlx-ext`、Apple M5 或更新芯片，以及 macOS >= 26.2 |
| `w8a16` | 使用浮点激活；对满足条件的单行 GPU 输入启用优化的逐通道 W8A16 路径 |
| `native` | 使用浮点激活与 MLX 原生 affine 算子 |

优化后端不可用时，回退到使用浮点激活的 MLX affine 算子。
W8A8 与浮点激活路径的数值结果可能不同，所有转换操作仍在 CPU 上执行。

显式选择原生路径：

```sh
.venv/bin/spark-mlx-convert \
    --source /path/to/Spark-X2.6-4B-INT4 \
    --output /path/to/Spark-X2.6-4B-native-MLX \
    --w8-backend native
```

输出设置 `quantize_activations=false`，用于关闭 MLX LM 的通用 QQLinear 路径。
选用且可用时，本项目的 W8A8 后端会单独执行激活量化。

## 5. Python API 与参数

命令行与 Python API 共用同一转换实现：

| 命令行参数 | Python 参数 | 默认值与含义 |
| --- | --- | --- |
| `--source` | `source` | 必填，本地 BF16 或 compressed-tensors 模型目录 |
| `--output` | `output` | 转换时必填，与源目录互不包含的新目录 |
| `--inspect` | `inspect_only` | `False`，仅检查元数据，不写入输出或扫描完整权重 |
| `--quantization` | `quantization` | `"mixed"`；`"none"` 保留主模型 BF16 权重，要求 BF16 输入 |
| `--w8-backend` | `w8_backend` | `"w8a8"`，也可选择 `"w8a16"` 或 `"native"` |
| `--chunk-rows` | `chunk_rows` | `256`，每次在 CPU 上处理的行数，必须为正整数 |

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

`source` 和 `output` 接受字符串或 `pathlib.Path`。
仅检查时调用 `convert_to_mlx(source, inspect_only=True)`，无需输出目录。
返回值为可序列化为 JSON 的报告，包含 `status`、`source_format`、
`quantized_module_counts`、`lm_head_untied` 和 draft 元数据。
完成转换后还包含 `coefficient_rounding`，完整报告保存为 `conversion_manifest.json`。

非法或不支持的值会触发 `ValueError`，缺少文件会触发 `FileNotFoundError`，
输出目录已存在会触发 `FileExistsError`。
不支持的 group-index/permutation 张量、激活方案、权重变换和稀疏格式会被拒绝。

模块入口为 `python -m spark_mlx_llm.commands.convert`，
`python -m spark_mlx_llm.conversion` 仍可作为兼容入口。
请使用 `--source` 和 `--output`；本转换器不支持上游的 `--hf-path`、`--mlx-path`
或 `--q-bits` 参数。可直接运行的脚本见
[Python 模型转换示例](examples.md#1-python-模型转换)。

## 6. 输出与推理

输出包含 `model.safetensors`、`model.safetensors.index.json`、`config.json`、
tokenizer 文件和 `conversion_manifest.json`。若存在 draft 张量，
还会单独保存 `model-draft.safetensors`，并纳入分片索引。
转换先写入临时目录，验证通过后发布输出；失败时清理临时目录。

运行转换后的主模型：

```sh
.venv/bin/spark-mlx-generate \
    --model /path/to/Spark-X2.6-4B-MLX \
    --prompt '你好，请介绍一下自己。' \
    --max-tokens 256 \
    --no-thinking
```

KV 精度在生成阶段单独控制。所有生成入口默认不量化 KV cache，
因此 BF16 模型使用 BF16 KV，与权重的 INT4/INT8 方案无关。
交互式聊天、Python 生成与 HTTP 服务用法见[接口使用说明](examples.md)。
