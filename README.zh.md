<p align="center">
  <img src="docs/assets/spark-mlx-logo.png" alt="星火与 MLX" width="480">
</p>

<h1 align="center">Spark-MLX-LLM</h1>

<p align="center">
  <a href="README.md">English</a> | <strong>简体中文</strong>
</p>

Spark-MLX-LLM 是面向 **Spark 2.6 和 Spark 2.5** 的本地推理项目，基于
[MLX](https://github.com/ml-explore/mlx) 和 [MLX LM](https://github.com/ml-explore/mlx-lm)。
本项目针对星火模型结构、混合精度量化和 Apple Silicon 上的推理计算进行了专门适配与优化，
提供模型转换、文本生成、交互式聊天和 HTTP 服务，方便在本地部署和使用星火模型。

## 项目特色

- **星火模型适配与转换**：将 BF16 和 compressed-tensors INT4 checkpoint 转换为 MLX 格式。
- **混合精度量化**：按模块采用 INT4、INT8、BF16 权重，并支持配置 W8 执行后端。
- **Apple Silicon 推理优化**：提供专用 QKV/RoPE、线性层和 INT8 KV 注意力算子，
  根据配置、设备与输入形状选择执行路径。
- **多种使用方式**：提供命令行生成与聊天、Python 普通与流式生成 API，以及 HTTP 服务。

## 环境配置

以下 Apple Silicon 安装流程需要：

| 项目 | 要求 |
| --- | --- |
| 硬件与系统 | Apple Silicon，macOS >= 26.2 |
| Python | 原生 `arm64` CPython，推荐 3.12 |
| 编译工具链 | 包含 Metal 4 编译器和 MetalPerformancePrimitives 头文件的 Xcode |

W8A8 加速还需要 **Apple M5 或更新芯片**及兼容的 `spark-mlx-ext`。
该后端不可用时，W8 投影回退到使用浮点激活的 MLX affine 算子，详见[量化与兼容性说明](#量化与兼容性说明)。

使用 [uv](https://docs.astral.sh/uv/getting-started/installation/) 管理 Python 和依赖。
准备好 Xcode 并安装 uv 后，执行：

```sh
git clone https://github.com/XHToken/Spark-MLX-LLM.git
cd Spark-MLX-LLM
uv python install 3.12
uv sync --locked
```

安装会自动创建 `.venv`、安装项目依赖并编译仓库内的 Apple Silicon 扩展。
后续命令均在仓库根目录执行。推理需要 Metal 或 CUDA GPU，模型转换在 CPU 上执行。
Linux/CUDA 安装方式、环境检查与常见问题见[环境配置说明](docs/zh/environment.md)。

## 快速开始

### 1. 模型下载

**推荐使用 Spark-X2.6-4B-INT4。** 从以下任一平台将模型文件下载到本地目录。
两种格式均可作为第二步的转换输入。

| 模型 | 格式 | Hugging Face | 魔搭（ModelScope） |
| --- | --- | --- | --- |
| Spark-X2.6-4B | BF16 | [下载](https://huggingface.co/XHToken/Spark-X2.6-4B) | [下载](https://www.modelscope.cn/models/XHToken/Spark-X2.6-4B) |
| Spark-X2.6-4B-INT4 | INT4 | [下载](https://huggingface.co/XHToken/Spark-X2.6-4B-INT4) | [下载](https://www.modelscope.cn/models/XHToken/Spark-X2.6-4B-INT4) |

### 2. 模型量化和转换

将下载的模型转换为 MLX 格式：

```sh
.venv/bin/spark-mlx-convert \
    --source /path/to/Spark-X2.6-4B-INT4 \
    --output /path/to/Spark-X2.6-4B-MLX
```

`--source` 为第一步下载的本地目录，`--output` 必须是尚不存在的新目录。
默认转换采用 INT4/INT8/BF16 混合精度权重，具体方案见下方[量化与兼容性说明](#量化与兼容性说明)，
完整转换选项见[模型转换文档](docs/zh/conversion.md)。

### 3. 使用

**将第二步转换得到的输出目录传给 `--model`**，即可加载模型进行推理。

生成一次回答：

```sh
.venv/bin/spark-mlx-generate \
    --model /path/to/Spark-X2.6-4B-MLX \
    --prompt '你好，请介绍一下自己。' \
    --max-tokens 256 \
    --no-thinking
```

启动交互式聊天：

```sh
.venv/bin/spark-mlx-chat \
    --model /path/to/Spark-X2.6-4B-MLX \
    --no-thinking
```

以上示例通过 `--no-thinking` 关闭思考。模型模板支持思考时，可将它替换为
`--thinking --reasoning-effort low`（或 `high`）。`--max-tokens` 限制思考与最终回答的
总长度，开启思考后可按需调大，详见[思考模式说明](docs/zh/examples.md#思考模式)。

### 4. 启动 HTTP 服务

在本机提供转换后模型的 HTTP 服务：

```sh
.venv/bin/spark-mlx-server \
    --model /path/to/Spark-X2.6-4B-MLX \
    --host 127.0.0.1 \
    --port 8080 \
    --no-thinking
```

保持该终端运行，在另一个终端或应用中向 `http://127.0.0.1:8080` 发送请求，
使用 Ctrl-C 停止服务。聊天、流式响应和文本续写的请求方式见
[HTTP 调用示例](docs/zh/examples.md#4-http-聊天流式聊天和文本续写)。

`generate`、`chat`、`server`、Python 普通与流式生成 API，以及 HTTP 请求的完整用法见
[接口使用说明](docs/zh/examples.md)。

## 量化与兼容性说明

默认转换对主模型采用以下方案。解码器内的模块在每层使用相同方案，embedding 和
`lm_head` 为模型级模块。激活列描述 W8A8 后端可用且启用时的计算方式，
权重与 scale 列描述转换后 checkpoint 的存储格式。

| 层 / 模块 | 权重量化方式 | 激活（启用 W8A8 时） | 权重 scale 类型 |
| --- | --- | --- | --- |
| `self_attn.q_k_v_proj` | 非对称 INT4，group size 32 | BF16 | BF16 |
| `mlp.gate_proj` | 非对称 INT4，group size 32 | BF16 | BF16 |
| `mlp.up_proj` | 非对称 INT4，group size 32 | BF16 | BF16 |
| `mlp.down_proj` | 非对称 INT4，group size 32 | BF16 | BF16 |
| `self_attn.out_proj` | 对称 INT8，逐输出通道 | 按 token 动态 INT8 | BF16 |
| `lm_head` | 对称 INT8，逐词表行 | 按 token 动态 INT8 | FP32 |
| `self_attn.g_proj` | 保留 BF16，不量化 | BF16 | — |
| `model.embedding` | 保留 BF16，不量化 | — | — |
| RMSNorm | 保留 BF16，不量化 | 不量化 | — |

转换器默认使用 `--w8-backend w8a8`。加载模型时若 W8A8 不可用，W8 层保留使用
浮点激活的 MLX affine 路径；也可以在转换命令中加入 `--w8-backend native`
显式选择该路径。W8 权重保持不变，浮点激活与 INT8 激活路径的数值结果可能不同。
转换选项与精度细节见[模型转换文档](docs/zh/conversion.md)。

所有生成入口默认保留模型精度，不量化 KV cache：**BF16 模型使用 BF16 KV**，
全局注意力层与滑动窗口层均如此。如需对全局注意力层启用 INT8/G32 KV，
可向 `spark-mlx-generate` 显式传入 `--kv-bits 8`，或向本包 Python 生成函数传入
`kv_bits=8`。generate 与 Python 包装默认按 512 tokens 分块处理输入，chat/server
保留上游 MLX LM 的分块行为。这些设置与权重量化分别控制，详见
[KV 缓存与输入分块](docs/zh/examples.md#kv-缓存与输入分块)。

## 文档导航

全部文档见[文档目录](docs/README.md)。

| 主题 | 文档 |
| --- | --- |
| 安装、平台要求与常见问题 | [环境配置](docs/zh/environment.md) |
| CLI、Python、HTTP 与思考模式 | [接口使用说明](docs/zh/examples.md) |
| 模型转换与量化 | [模型转换](docs/zh/conversion.md) |
| 性能测试命令与指标口径 | [Benchmark 指南](docs/zh/benchmark.md) |

## 许可证

参见 [LICENSE](LICENSE)。
