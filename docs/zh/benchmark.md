# Benchmark 用法与指标

[项目首页](../../README.zh.md) · [接口示例](examples.md) · [English](../en/benchmark.md)

[`example/benchmark.py`](../../example/benchmark.py) 用于测量不同输入长度下，
单条请求的 prefill（处理输入）和 decode（生成输出）吞吐，单位为 token/秒。
它直接调用本项目的 `load()` 和 `stream_generate()`，不依赖 `benchmark/`
或 `spark_mlx_llm/benchmark/`，也不需要启动 HTTP 服务。

## 1. 运行方法

先按[环境配置](environment.md)安装项目，并准备 Spark2_5 模型。可以使用未量化的
Hugging Face 模型或已转换的 MLX 模型；原始 compressed-tensors INT4/INT8
模型需要先转换。以下命令都在仓库根目录执行。

```sh
.venv/bin/python example/benchmark.py \
  --model /path/to/converted-mlx-model \
  --lengths 128 512 2048 8192 \
  --decode-tokens 256 \
  --runs 3
```

这条命令加载一次模型，依次测试 4 个输入长度；每个长度完整预热 1 次，再正式
测量 3 次，每次生成 256 个 token。`--model` 也接受 Hugging Face 仓库 ID。
脚本使用 MLX 默认设备，启动时打印 `Device`，请据此确认实际运行设备。

查看全部参数，或把终端结果保存到文件：

```sh
.venv/bin/python example/benchmark.py --help
.venv/bin/python example/benchmark.py --model /path/to/converted-mlx-model \
  --lengths 512 2048 --decode-tokens 256 --runs 5 > benchmark-output.txt
```

脚本输出文本表格，不生成 JSON/CSV，也不输出生成的文本或逐轮原始数据。

## 2. 参数说明

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--model` | 必填 | 本地模型目录或 Hugging Face 仓库 ID。 |
| `--lengths` | `128 512 2048 8192` | 一个或多个输入 token 数，按给定顺序测试；不是字符数或字节数。 |
| `--decode-tokens` | `256` | 每次生成的输出 token 数；所有输入长度共用此值。 |
| `--runs` | `3` | 每个输入长度的正式测量次数，不含固定的 1 次预热。 |
| `--prefill-step-size` | `512` | prefill 每个分块最多处理的 token 数；不改变总输入长度。 |
| `--kv-bits` | `none` | 默认不量化 KV，保留模型的缓存精度；显式设为 `4` 或 `8` 可启用量化。 |
| `--kv-group-size` | `32` | KV 量化分组大小，可选 `32`、`64`、`128`。 |
| `--quantized-kv-start` | `0` | 启用 KV 量化的缓存 offset 阈值，必须为非负整数。 |
| `--kv-quantization-scope` | `full-attention` | 可选 `full-attention`、`all`；启用量化时默认仅作用于全局注意力层。 |

长度、输出 token 数、测量次数和分块大小必须为正整数。若模型配置给出了
`max_position_embeddings`，脚本会检查最大输入长度加输出长度不超过该限制。

默认所有层均保留模型的 KV 精度，BF16 模型使用 BF16 KV。显式设置 `--kv-bits 8`
后，默认对全局注意力层使用 INT8/G32 KV，滑动窗口注意力（SWA）的缓存仍保留模型精度。
含 `RotatingKVCache` 的模型不能在开启 KV 量化时使用 `all`，应保留 `full-attention`。
`--kv-bits none` 不会把模型强制转换为 BF16，KV 量化参数也不会改变模型权重的量化位数。

量化阈值在 prefill 分块或模型步骤之后检查；`0` 表示从第一次检查起允许转换，
并非所有缓存从第一步计算前就已经量化。阈值过大时，短输入可能全程不发生量化。

W8 后端在脚本中固定为 `native`，其余算子设置沿用模型加载配置。batch size
固定为 1，采样温度固定为 0；脚本不读取 `example/inference.json`，也不接受
`--config`、`--temp`、`--w8-qmv-backend` 等生成命令的参数。
KV 和分块参数的公共实现见 [generation.py](../../spark_mlx_llm/generation.py)。

## 3. 测试流程

1. 加载模型并求值权重，使用种子 `0` 从模型词表中生成固定的随机 token ID 序列。
   不同输入长度使用同一序列的前缀，直接传入 token ID，不套聊天模板、不做文本分词。
2. 针对每个长度，先完整执行一次 prefill 和 decode 作为预热，然后测量 `--runs` 次。
   每次都创建新的模型 KV cache；只有同一次生成的 prefill 和 decode 共用缓存。
3. 每次调用前同步设备并清理 MLX 分配器缓存，结束后再次同步。这个清理操作与
   模型的 KV cache 是两回事，预热也不会使下一轮跳过输入处理。
4. 使用贪心采样，关闭 EOS 提前停止。脚本检查实际输入、输出 token 数及
   `finish_reason="length"`，长度不符时直接报错。
5. 从每轮最后一条响应读取吞吐，分别计算 prefill 和 decode 的中位数并打印一行。

预热次数固定为 1，没有对应命令行开关。两列中位数分别计算，可能来自不同轮次；
它们既不是各轮速度的算术平均，也不是总 token 数除以多轮总耗时。

## 4. 输出与指标口径

启动信息包含设备、batch size、输出长度、预热次数、正式次数以及 W8/KV/分块配置。
随后输出如下表格，数值仅为格式示例，不代表任何机器的实测性能：

```text
Input tokens    Prefill tok/s     Decode tok/s
         128          1000.00           120.00
         512          1600.00           110.00
        2048          2000.00           100.00
        8192          1800.00            80.00
```

| 输出列 | 来源与含义 |
| --- | --- |
| `Input tokens` | 本行测量使用的输入 token 数，即对应的 `--lengths` 值。 |
| `Prefill tok/s` | 各正式轮次最终响应的 `prompt_tps` 中位数，反映输入处理吞吐。 |
| `Decode tok/s` | 各正式轮次最终响应的 `generation_tps` 中位数，反映给定输入上下文下的输出生成吞吐。 |

`tok/s` 表示每秒 token 数，数值越大表示吞吐越高。decode 每生成一个 token，
上下文都继续增长，因此一行测量的是该输入长度起步、连续生成指定输出长度的整段表现。

### Prefill tok/s

脚本直接使用 MLX-LM 返回的统计，不另外实现计时器。以下边界对应当前锁定的
MLX-LM **0.31.3** 中 `mlx_lm/generate.py` 的 `stream_generate()`；升级依赖后
应重新核对实现，版本记录见 [uv.lock](../../uv.lock)。

对单次正式测量，设输入长度为 `P`：

```text
T_prompt = 上游开始迭代生成器，到取得首个输出 token 的时间
prompt_tps = P / T_prompt
```

这个区间包含输入分块处理、模型前向、期间发生的 KV 更新/量化及首个 token 的采样。
它反映生成路径的吞吐，不是单独计时某个 GPU 算子。比如某轮输入 2048 tokens、
`prompt_tps=1024`，则该轮上游 prompt 计时约为 2 秒。

它与推理内部的首 token 等待时间有关，但不等于用户请求的端到端 TTFT（首 token
延迟）：模型加载、下载、输入准备、HTTP 排队和网络传输均不在这个区间内。

### Decode tok/s

取得首个输出 token 后，上游重置计时器；直到生成完指定数量的 token、完成最终
反分词并构造最后一条响应时，计算生成吞吐。设输出总数为 `D`：

```text
T_generation = 上游取得首个输出 token 后重置计时器，到最终统计的时间
generation_tps = D / T_generation
```

**分子 `D` 包含首个输出 token，而其生成时间已计入 prompt 区间。**
因此这里沿用的是上游统计口径，并不是 `(D - 1)` 除以后续 token 的耗时。
例如输出 256 tokens、`generation_tps=128`，对应的上游 generation 计时约为 2 秒。

上游采用异步流水线，首 token 返回时可能已经提交下一步计算；本脚本不会逐 token
额外同步来切分纯 GPU decode 耗时。采样、反分词以及区间内的 Python 迭代开销保留
在统计中。`--decode-tokens 1` 虽然有效，但分母主要剩余收尾开销，数值容易虚高；
比较性能时应固定一个足够长的输出长度，例如 256。

### 包含与排除的工作

| 工作 | 是否计入显示的 tok/s |
| --- | --- |
| 模型下载、加载、权重求值，随机输入构造 | 不计入。 |
| 每个长度的完整预热 | 不计入中位数。 |
| 正式生成区间内的模型计算、KV 更新/量化、采样、反分词 | 按上述上游边界计入对应阶段。 |
| 脚本在调用前后的同步、调用前的分配器缓存清理、表格打印 | 不计入；上游生成内部的等待和缓存操作仍按其所在区间计入。 |

本表不报告端到端请求耗时、逐 token 延迟（ITL/TPOT）、P95/P99 延迟、并发吞吐
或峰值内存。不要把 `1 / Decode tok/s` 直接视为严格的平均 token 间隔，也不要
把两列中位数换算后相加当作程序的总运行时间。

## 5. 比较不同配置

例如比较关闭 KV 量化和 INT8 KV 时，保持模型、输入/输出长度、分块和重复次数一致：

```sh
for bits in none 8; do
  .venv/bin/python example/benchmark.py --model /path/to/converted-mlx-model \
    --lengths 512 2048 8192 --decode-tokens 256 --runs 5 \
    --prefill-step-size 512 --kv-bits "$bits"
done
```

内存紧张时，可以单独尝试较小的 prefill 分块，例如 `--prefill-step-size 128`；
这也会改变吞吐，应把分块大小作为对比条件记录下来。较长上下文会增加缓存需求，
超过可用内存并出现 swap 时，结果会受到明显影响。

比较时使用相同设备、依赖版本和算子配置，并尽量保持相近的系统负载、温度与内存状态。
脚本本身不做设备锁、空闲检测、温度或 swap 监控，也不自动剔除异常轮次。
随机 token 用于固定工作量，不能说明真实聊天文本的生成质量或服务并发能力。
