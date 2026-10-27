# Benchmark usage and metrics

[Project overview](../../README.md) · [API examples](examples.md) · [简体中文](../zh/benchmark.md)

[`example/benchmark.py`](../../example/benchmark.py) measures single-request
prefill (input processing) and decode (output generation) throughput at different
input lengths, in tokens per second. It calls the project's `load()` and
`stream_generate()` APIs directly, without depending on `benchmark/` or
`spark_mlx_llm/benchmark/`. No HTTP server is needed.

## 1. Running the benchmark

Install the project using the [environment guide](environment.md) and prepare a
Spark2_5 model. Use an unquantized Hugging Face model or a model already converted
to MLX format; original compressed-tensors INT4/INT8 models require conversion.
Run all commands below from the repository root.

```sh
.venv/bin/python example/benchmark.py \
  --model /path/to/converted-mlx-model \
  --lengths 128 512 2048 8192 \
  --decode-tokens 256 \
  --runs 3
```

This loads the model once and tests four input lengths in order. Each length
gets one complete warmup followed by three measured runs, each generating 256
tokens. `--model` also accepts a Hugging Face repository ID. The script uses
MLX's default device; check the printed `Device` value to confirm the device used.

View all options or save the terminal output to a file:

```sh
.venv/bin/python example/benchmark.py --help
.venv/bin/python example/benchmark.py --model /path/to/converted-mlx-model \
  --lengths 512 2048 --decode-tokens 256 --runs 5 > benchmark-output.txt
```

The script prints a text table. It does not produce JSON/CSV, generated text,
or individual measurement records.

## 2. Arguments

| Argument | Default | Meaning |
| --- | --- | --- |
| `--model` | Required | Local model directory or Hugging Face repository ID. |
| `--lengths` | `128 512 2048 8192` | One or more input token counts, tested in the given order; these are not character or byte counts. |
| `--decode-tokens` | `256` | Output tokens generated per run, shared by all input lengths. |
| `--runs` | `3` | Measured runs per input length, excluding the fixed single warmup. |
| `--prefill-step-size` | `512` | Maximum tokens processed per prefill chunk; does not change the total input length. |
| `--kv-bits` | `none` | Retain the model's cache precision without quantization by default; explicitly set `4` or `8` to quantize KV. |
| `--kv-group-size` | `32` | KV quantization group size: `32`, `64`, or `128`. |
| `--quantized-kv-start` | `0` | Cache offset threshold for enabling KV quantization; must be a nonnegative integer. |
| `--kv-quantization-scope` | `full-attention` | `full-attention` or `all`; when quantization is enabled, it defaults to full-attention layers only. |

Input lengths, output length, run count, and chunk size must be positive integers.
If the model configuration provides `max_position_embeddings`, the script checks
that the largest input length plus the output length fits within that limit.

By default, all layers retain model KV precision: BF16 models use BF16 KV.
Explicitly setting `--kv-bits 8` enables INT8/G32 full-attention KV by default,
while sliding-window attention (SWA) caches retain model precision.
Models with `RotatingKVCache` cannot
use `all` with KV quantization enabled; keep `full-attention` instead.
`--kv-bits none` does not force the model to BF16, and KV quantization settings
do not change weight quantization bits.

The quantization threshold is checked after a prefill chunk or model step. A value
of `0` permits conversion at the first check; it does not mean every cache is
already quantized before the first computation. A large threshold can leave a
short run entirely unquantized.

The script fixes the W8 backend to `native`; other operator settings follow the
model loading configuration. Batch size is fixed at 1 and sampling temperature
at 0. The script does not read `example/inference.json` or accept generation CLI
options such as `--config`, `--temp`, or `--w8-qmv-backend`. Shared KV and chunking
settings are implemented in [generation.py](../../spark_mlx_llm/generation.py).

## 3. Measurement procedure

1. Load the model and evaluate its weights. Generate a fixed sequence of random
   token IDs from the model vocabulary using seed `0`. Different input lengths
   use prefixes of that sequence. IDs are passed directly, without a chat template
   or text tokenization.
2. For each length, run one complete prefill and decode warmup, then `--runs`
   measured generations. Each invocation creates a fresh model KV cache; only
   prefill and decode within the same invocation share that cache.
3. Synchronize the device and clear MLX's allocator cache before each invocation,
   then synchronize again afterward. The allocator cache is separate from the
   model's KV cache; warmup does not let later runs skip input processing.
4. Use greedy sampling with EOS stopping disabled. Check actual input/output
   token counts and `finish_reason="length"`, raising an error on a mismatch.
5. Read throughput from each run's final response, compute separate prefill and
   decode medians, and print one row per input length.

The single warmup is fixed and has no command-line option. The two medians can
come from different runs. They are neither arithmetic means of run throughput
nor total tokens divided by the combined duration of all runs.

## 4. Output and metric definitions

Startup output reports the device, batch size, output length, warmup and measured
run counts, and W8/KV/chunking settings. The table then looks like this; these
numbers illustrate the format and are not measured hardware results:

```text
Input tokens    Prefill tok/s     Decode tok/s
         128          1000.00           120.00
         512          1600.00           110.00
        2048          2000.00           100.00
        8192          1800.00            80.00
```

| Column | Source and meaning |
| --- | --- |
| `Input tokens` | Input token count for this row, matching a value in `--lengths`. |
| `Prefill tok/s` | Median final-response `prompt_tps` across measured runs; input processing throughput. |
| `Decode tok/s` | Median final-response `generation_tps` across measured runs; output generation throughput for the given input context. |

`tok/s` means tokens per second; higher values indicate greater throughput. The
context grows as each output token is generated, so a row measures the whole
generation window starting from that input length.

### Prefill tok/s

The script uses MLX-LM's statistics directly, without an independent timer. The
boundaries below describe `stream_generate()` in `mlx_lm/generate.py` from the
currently locked MLX-LM **0.31.3**. Recheck the implementation when upgrading;
dependency versions are recorded in [uv.lock](../../uv.lock).

For one measured run with input length `P`:

```text
T_prompt = time from upstream starting generator iteration to obtaining the first output token
prompt_tps = P / T_prompt
```

This interval includes prompt chunks, model forward passes, KV updates and
quantization performed during the interval, and sampling the first token. It
measures the generation path rather than an isolated GPU operator. For example,
2048 input tokens at `prompt_tps=1024` imply an upstream prompt interval of about
2 seconds for that run.

This relates to waiting for the first token inside inference, but is not
end-to-end request TTFT (time to first token): model loading, downloads, input
preparation, HTTP queuing, and network transfer are outside the interval.

### Decode tok/s

Upstream resets its timer after obtaining the first output token. Generation
throughput is computed after producing the requested number of tokens and
finalizing detokenization, when constructing the final response. For a total
output count of `D`:

```text
T_generation = time from the timer reset after the first output token to the final statistics
generation_tps = D / T_generation
```

**The numerator `D` includes the first output token, whose generation time was
already counted in the prompt interval.** This preserves upstream's convention;
it is not `(D - 1)` divided by the time for subsequent tokens. For example,
256 output tokens at `generation_tps=128` imply an upstream generation interval
of about 2 seconds.

Upstream uses an asynchronous pipeline and may already have submitted the next
step when the first token returns. This script does not add per-token
synchronization to isolate GPU decode time. Sampling, detokenization, and Python
iteration overhead within the interval remain included. Although
`--decode-tokens 1` is valid, its denominator mostly contains finishing overhead,
which can inflate the result. Use a fixed, sufficiently long output length, such
as 256, when comparing performance.

### Included and excluded work

| Work | Included in displayed tok/s? |
| --- | --- |
| Model downloading/loading, weight evaluation, random input construction | No. |
| Complete warmup for each length | Excluded from the median. |
| Model computation, KV updates/quantization, sampling, and detokenization during measured generation | Included in the appropriate phase according to the boundaries above. |
| Script-level synchronization before/after invocation, allocator clearing before invocation, and table printing | No; waits and cache operations inside upstream generation remain included in their respective intervals. |

The table does not report end-to-end request duration, per-token latency
(ITL/TPOT), P95/P99 latency, concurrent throughput, or peak memory. Do not treat
`1 / Decode tok/s` as a strict average inter-token interval, or convert the two
column medians into durations and add them to obtain total program runtime.

## 5. Comparing configurations

For example, compare unquantized and INT8 KV while keeping the model, input/output
lengths, chunk size, and run count fixed:

```sh
for bits in none 8; do
  .venv/bin/python example/benchmark.py --model /path/to/converted-mlx-model \
    --lengths 512 2048 8192 --decode-tokens 256 --runs 5 \
    --prefill-step-size 512 --kv-bits "$bits"
done
```

If memory is tight, separately try smaller prefill chunks, such as
`--prefill-step-size 128`. This also changes throughput, so record chunk size as
part of the comparison. Longer contexts require more cache memory; swapping
after exhausting available memory can substantially affect results.

Use the same device, dependency versions, and operator configuration, and keep
system load, temperature, and memory conditions similar. The script does not
lock the device, check for idleness, monitor temperature or swap, or discard
outlier runs. Random tokens provide a fixed workload; they do not measure
generation quality on real chat text or concurrent serving capacity.
