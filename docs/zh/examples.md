# 接口调用示例

[项目首页](../../README.zh.md) · [English](../en/examples.md)

这些脚本分别演示本项目的 Python API、四个命令行入口和 HTTP 接口。
Python API 示例直接在代码中设置参数、调用函数，不解析命令行参数；可以把调用片段
复制到自己的业务代码中。需要从终端传参时使用对应的 shell 示例。
公共配置和各入口的区别见下文的[配置参考](#5-公共推理配置)与[实现差异](#6-各入口与上游-mlx-lm-的关系)。

| 脚本 | 调用接口 | 用途 |
| --- | --- | --- |
| [python_convert.py](../../example/python_convert.py) | `convert_to_mlx()` | 检查或转换模型，打印 JSON 报告 |
| [python_generate.py](../../example/python_generate.py) | `load()` + `generate()` | 一次性返回完整生成文本 |
| [python_stream_generate.py](../../example/python_stream_generate.py) | `load()` + `stream_generate()` | 逐段输出生成文本 |
| [benchmark.py](../../example/benchmark.py) | `load()` + `stream_generate()` | 从命令行指定多个输入长度，测量 prefill/decode 吞吐 |
| [cli_convert.sh](../../example/cli_convert.sh) | `spark-mlx-convert` 的模块入口 | 命令行检查或转换模型 |
| [cli_generate.sh](../../example/cli_generate.sh) | `spark-mlx-generate` 的模块入口 | 单次命令行生成 |
| [cli_chat.sh](../../example/cli_chat.sh) | `spark-mlx-chat` 的模块入口 | 终端多轮交互聊天 |
| [cli_server.sh](../../example/cli_server.sh) | `spark-mlx-server` 的模块入口 | 启动本地 HTTP 服务 |
| [http_chat.py](../../example/http_chat.py) | `POST /v1/chat/completions` | 普通聊天或 SSE 流式聊天 |
| [http_completions.py](../../example/http_completions.py) | `POST /v1/completions` | 原始 prompt 文本续写 |

## 运行准备

以下命令在仓库根目录执行，使用已经按[环境配置](environment.md)
安装本项目的 `.venv`。Python API 和命令行推理不需要先启动 HTTP 服务。
HTTP 客户端只使用 Python 标准库，也可以单独复制到其他机器，用 `python3` 运行。

Python API 与 HTTP 示例请先修改文件顶部的模型路径或请求参数；benchmark 从命令行传参。
推理需要未量化的 Hugging Face
模型或已经转换的 MLX 模型；原始 compressed-tensors INT4/INT8 模型需先转换。

Python 示例不自动读取配置文件。三个推理 shell 示例读取
[`inference.json`](../../example/inference.json)，再用脚本中列出的参数覆盖文件值，详见下面的
命令行示例。配置文件也可以通过 `GenerationConfig.from_file()` 在自己的 Python
代码中使用，具体用法见[Python 配置复用](#python-中复用配置)。

## 1. Python 模型转换

编辑 [python_convert.py](../../example/python_convert.py) 的 `SOURCE` 和 `OUTPUT`，再直接运行。
`SOURCE` 指向转换前的模型；`OUTPUT` 必须是尚不存在的新目录，且两者不能互相包含。

```sh
.venv/bin/python example/python_convert.py
```

脚本明确传入 `quantization="mixed"`、`w8_backend="native"`、`chunk_rows=256`。
BF16 输入若要保留主模型精度，把调用中的 `quantization` 改成 `"none"`。
只检查结构时，将转换调用替换为：

```python
report = convert_to_mlx(source=SOURCE, inspect_only=True)
```

返回的 `report` 是字典，可以用于检查转换结果或写入业务日志。
通过 `spark-mlx-convert --help` 查看完整转换参数。
本版本支持范围为不带 draft 权重的模型转换与推理。

## 2. Python 普通生成与流式生成

编辑两个文件顶部的 `MODEL_PATH` 和 `MESSAGES`。它们使用相同的输入、256-token
上限、温度 0、关闭思考，并保留模型的 KV 精度（BF16 模型使用 BF16 KV），
只演示不同的结果获取方式。

| Python API | 返回值 | 业务代码如何使用 |
| --- | --- | --- |
| `generate()` | 完整字符串 `str` | 等待生成完成后保存、处理或返回给调用方 |
| `stream_generate()` | 逐步产生响应对象的迭代器 | 在循环中使用 `response.text` 更新界面或发送增量 |

这两个函数都已由 [spark_mlx_llm/__init__.py](../../spark_mlx_llm/__init__.py)
导出，实现位于 [generation.py](../../spark_mlx_llm/generation.py)。命令行只有一个
`spark-mlx-generate` 入口，没有单独的 `spark-mlx-stream-generate` 命令。
`generate(verbose=True)` 可以边生成边在终端打印，但函数返回值仍是完整字符串；
业务代码要逐段消费结果时使用 `stream_generate()`。

```sh
.venv/bin/python example/python_generate.py
.venv/bin/python example/python_stream_generate.py
```

参数直接写在对应调用中：模型与算子参数传给 `load()`，思考参数传给
`tokenizer.apply_chat_template()`，采样参数传给 `make_sampler()`，输出长度、KV
与分块参数传给生成接口。开启思考时，在模板调用中设置
`enable_thinking=True, reasoning_effort="low"`。返回文本可能包含思考标记。

### 按输入长度测试 prefill/decode 速度

[`benchmark.py`](../../example/benchmark.py) 是独立的小脚本，直接调用本项目的推理
API，不依赖 `benchmark/` 或 `spark_mlx_llm/benchmark/`。

```sh
.venv/bin/python example/benchmark.py --model /path/to/converted-mlx-model \
  --lengths 128 512 2048 8192 --decode-tokens 256 --runs 3
# 显式开启 INT8 KV 量化，修改 prefill 分块大小：
.venv/bin/python example/benchmark.py --model /path/to/converted-mlx-model \
  --lengths 512 2048 --kv-bits 8 --prefill-step-size 256
```

默认 batch size 为 1，输入长度为 128/512/2048/8192 tokens，输出 256 tokens。
每个长度完整预热一次，再测量 3 次，按长度打印 prefill 与 decode 的 tok/s 中位数。
模型只加载一次，每次生成新建 KV cache。输入使用固定种子的随机 token，不套聊天模板；
贪心采样并忽略 EOS，保证输出长度固定，因此这些输入用于吞吐测试。

统计直接使用 MLX-LM 的 `prompt_tps` / `generation_tps`：prefill 计时到首个输出
token，decode 沿用上游的输出计数和异步流水线计时口径。模型加载、输入准备和预热
不计入结果。W8 后端固定为 `native`，与其他 Python 示例一致；默认保留模型的
KV 精度、不量化，prefill 分块 512 tokens。用 `--help` 查看完整 KV 参数。

完整参数表、测试流程、计时公式和结果解读见 [Benchmark 用法与指标](benchmark.md)。

## 3. 四个命令行入口

Shell 示例可通过环境变量或命令行参数设置路径，每个都支持 `--help`。
脚本使用项目 `.venv/bin/python`，可通过 `SPARK_PYTHON` 指定其他已安装本包的解释器。
三个推理脚本读取共享 JSON 后显式关闭思考，生成、聊天、服务的输出上限分别为
256、512、2048 tokens；追加参数可覆盖脚本值。

```sh
export SPARK_MODEL="/path/to/converted-mlx-model"
export SPARK_SOURCE="/path/to/source-model"

bash example/cli_convert.sh --source "$SPARK_SOURCE" --inspect
# 实际转换：脚本明确传入 source、output、quantization、w8-backend、chunk-rows。
# 输出目录必须尚不存在，避免与前面 Python 转换示例的输出重复。
bash example/cli_convert.sh \
  --source "$SPARK_SOURCE" --output /path/to/another-new-mlx-model \
  --quantization mixed --w8-backend native --chunk-rows 256

# 脚本已给出问题、长度、采样和 KV 参数。
bash example/cli_generate.sh
# 按需覆盖脚本中的示例参数。
bash example/cli_generate.sh --prompt '你好，请介绍一下自己。' --max-tokens 128

bash example/cli_chat.sh

bash example/cli_server.sh
```

聊天界面输入 `q` 退出、`r` 清空会话缓存、`h` 显示帮助。
服务默认监听 `127.0.0.1:8080`，使用 Ctrl-C 停止；可追加 `--host`、`--port` 修改。
服务运行期间，在另一个终端执行下面的 HTTP 示例。

`chat/server` 使用上游 KV 与分块设置，不接受 generate 的全部参数；
例如不要向它们传 `--w8-qmv-backend` 或包含 Spark KV 字段的配置文件。

## 4. HTTP 聊天、流式聊天和文本续写

先运行 `cli_server.sh`。两个 HTTP 示例都在文件顶部直接定义 `BASE_URL`、
`TIMEOUT` 和请求字典 `PAYLOAD`，编辑后运行，不接收命令行参数。

```sh
# 健康检查和模型列表。
curl --fail-with-body http://127.0.0.1:8080/health
curl --fail-with-body http://127.0.0.1:8080/v1/models

# 聊天；将文件中 PAYLOAD 的 stream 改成 True 可演示 SSE 流式响应。
.venv/bin/python example/http_chat.py

# 原始文本续写；该接口不会套用聊天模板。
.venv/bin/python example/http_completions.py
```

`BASE_URL` 不含 `/v1`。请求中的 `model="default_model"` 使用服务启动时指定的模型，
客户端无需本地模型目录。
`http_chat.py` 默认请求 2048 tokens，`http_completions.py` 默认请求 128 tokens，
温度均为 0；这些请求字段会覆盖服务启动值。聊天示例在 `chat_template_kwargs`
中关闭思考；修改其中的 `enable_thinking` 和 `reasoning_effort` 可开启思考，
删除整个 `chat_template_kwargs` 字段则继承服务默认值。

HTTP 最终回答写到 stdout，思考文本、结束原因和普通响应的 token 用量写到 stderr。
`finish_reason=length` 表示达到输出上限，回答可能被截断。
`http_chat.py` 把思考字段放在 `chat_template_kwargs` 中，并处理 SSE 的
`delta.content`、`delta.reasoning` 和 `[DONE]`。多轮对话需在脚本的 `messages`
列表中补入历史消息；服务不会替客户端保存完整对话。

## 5. 公共推理配置

`generate`、`chat` 和 `server` 共用
[GenerationConfig](../../spark_mlx_llm/generation.py) 管理采样、输出长度和思考设置。
以下是接口本身的默认值；示例脚本可以显式覆盖它们。

| 配置字段 | 命令行参数 | 默认值 |
| --- | --- | --- |
| `max_tokens` | `--max-tokens` / `-m` | `512` |
| `temperature` | `--temp` / `--temperature` | `0.0` |
| `top_p` | `--top-p` | `1.0` |
| `top_k` | `--top-k` | `0` |
| `min_p` | `--min-p` | `0.0` |
| `enable_thinking` | `--thinking` / `--no-thinking` | 不覆盖模型模板 |
| `reasoning_effort` | `--reasoning-effort low/high` | 不覆盖模型模板 |
| `chat_template_kwargs` | `--chat-template-args '{...}'` | 空对象，透传其他模板参数 |

三个命令都接受 `--config example/inference.json`，但命令本身不会自动加载该文件。
文件路径相对于当前工作目录；示例 shell 脚本显式传入按脚本位置定位的配置路径。
Python `generate()` / `stream_generate()` 未指定长度时也默认最多生成 `512` tokens；
两个 Python 示例显式传入 `256`，共享 JSON 则选择 `2048` 和低强度思考。

覆盖顺序为 **代码默认值 → 配置文件 → 显式命令行参数 → 单次 HTTP 请求**。
模板参数按字段合并；同一层同时提供思考专用字段和 `chat_template_kwargs` 时，
思考专用字段优先。配置文件中的思考字段设为 JSON `null` 表示恢复模型模板默认值。

未知配置字段或非法类型会报错。`max_tokens` 必须是正整数，`top_k` 是非负整数；
采样数值必须有限，`temperature >= 0`，`top_p/min_p` 在 `[0, 1]` 内。
`top_k/top_p/min_p` 的 `0` 表示关闭对应过滤。
`messages/tools/tokenize/add_generation_prompt` 由调用入口管理，不能放入
`chat_template_kwargs`。

### KV 缓存与输入分块

以下字段适用于本包的 Python 生成函数和 `spark-mlx-generate` 的配置文件：

| 配置字段 | generate 命令行参数 | 默认值与有效值 |
| --- | --- | --- |
| `prefill_step_size` | `--prefill-step-size` | `512`，必须为正整数 |
| `kv_bits` | `--kv-bits` | 默认 JSON `null` / CLI `none`，保留模型精度；显式设为 `4/8` 可启用量化 |
| `kv_group_size` | `--kv-group-size` | `32`；支持 `32/64/128` |
| `quantized_kv_start` | `--quantized-kv-start` | `0`，必须为非负整数 |
| `kv_quantization_scope` | `--kv-quantization-scope` | `full-attention`；启用量化时仅作用于全局层，也支持 `all` |

所有入口默认均不量化 KV，BF16 模型使用 BF16 KV。`kv_bits=None` 保留模型精度，
不会强制转换其他 dtype。`chat/server` 的 `--config` 拒绝这些 KV 字段，
保留上游 KV 和分块设置。
算子参数属于 `load()` 或模型配置，不属于推理 JSON；可通过
`spark-mlx-generate --help` 查看算子开关。对应扩展实现不可用时使用
MLX / MLX-LM 原生路径。`--device`、`--dtype`、`--seed` 和
`--ignore-chat-template` 等 generate 专用参数可通过该命令的 `--help` 查看。
`--ignore-chat-template` 会跳过整个聊天模板，不能代替关闭思考，且不能与模板配置同时使用。

### Python 中复用配置

需要共用 JSON 时，可以在自己的 Python 代码中采用以下调用方式。
`model`、`tokenizer` 和 `MESSAGES` 沿用前面的生成示例：

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

`GenerationConfig.from_dict({...})` 严格检查字段；`from_options({...})` 从混合参数
中提取已知配置字段，仍校验这些字段的值。`chat_template_kwargs` 是只读快照，
输入和嵌套值都与外部对象隔离。用 `overlay()` 生成修改后的配置，用
`template_kwargs()` 获取可独立修改的模板参数字典。
直接调用上游接口时，可用 `config.prepare_kv_kwargs(model, ...)` 准备缓存参数。

### 思考模式

Spark 示例模型的模板支持 `enable_thinking=True` 配合 `reasoning_effort="low"`
或 `"high"`；关闭时使用布尔值 `False`，不是字符串。
`medium` / `none` 不是有效强度值。仅设置强度不会打开已被关闭的思考。
在 HTTP 请求中，这两个字段放在 `chat_template_kwargs` 内；当前服务不把请求顶层的
`reasoning_effort` 传入模板。

`max_tokens` 包含思考和最终回答的总长度，`low/high` 不对应固定的思考 token 预算。
这些是模型聊天模板的约定，并非所有 MLX 模型通用的行为。

## 6. 各入口与上游 MLX LM 的关系

四个命令的实现位于 [commands/](../../spark_mlx_llm/commands/)，命令名称由
[pyproject.toml](../../pyproject.toml) 的 `[project.scripts]` 声明。

| 入口 | 模型加载与执行路径 | 配置范围 |
| --- | --- | --- |
| Python `load()` + `generate()` / `stream_generate()`，以及 `spark-mlx-generate` | 本项目加载器和生成包装，再委托 MLX LM 执行生成 | 应用 Spark KV 与分块默认值 |
| `spark-mlx-chat` / `spark-mlx-server` | 注册模型后使用上游加载、缓存、生成与服务能力，并适配公共推理设置 | 共享采样与思考设置，保留各自上游 KV 行为 |
| `spark-mlx-convert` / `convert_to_mlx()` | 本项目 CPU 转换实现 | 离线转换，不导入 MLX 推理模块 |

[registration.py](../../spark_mlx_llm/registration.py) 优先使用已安装的
`mlx_lm.models.spark2_5`；缺失时仅在当前进程中注册本项目模型，不修改磁盘上的
`mlx_lm`。chat 适配上游交互循环；server 将参数转交上游，HTTP 路由、响应格式和调度
沿用 MLX LM。这两个入口不调用本包的 `load()` 或生成包装函数。

本项目的 [load()](../../spark_mlx_llm/loader.py) 明确使用 Spark 模型，支持 `~/...`
本地目录和 `org/model` 仓库 ID；显式本地路径不存在时直接报错。
`dtype` 与算子覆盖值在下载或加载权重前校验。`load()` 本身不设置生成时的 KV 精度。

生成包装默认保留所有层的 KV 精度（BF16 模型使用 BF16 KV），输入按 512 tokens 分块。
显式传入 `kv_bits=8` 后，默认仅对全局注意力 KV 启用 INT8/G32 量化，
滑窗缓存仍保留模型精度和旋转状态。直接调用 `mlx_lm.generate` 仍使用上游默认值。
模型层的算子优化与生成阶段的缓存配置是分别控制的。
不同命令的参数集合也不同，完整选项以同一环境中的 `--help` 为准。

## 7. 工具调用解析与执行设备

当模型提供支持工具调用的 Spark2_5 聊天模板时，返回的 tokenizer 会自动提供
MLX LM 的工具解析器：

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

使用 Python API 时，在加载模型前选择执行设备：

```python
import mlx.core as mx

mx.set_default_device(mx.gpu)  # 或 mx.cpu
```
