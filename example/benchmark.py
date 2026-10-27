"""按输入 token 长度测试 prefill/decode 吞吐；从仓库根目录运行：

.venv/bin/python example/benchmark.py --model /path/to/converted-mlx-model \
    --lengths 128 512 2048 8192 --decode-tokens 256 --runs 3

每个长度完整预热一次，再取多次测量的中位数。使用固定随机 token、贪心采样，
不套聊天模板；忽略 EOS，以保证输出长度一致。计时沿用 MLX-LM 的
prompt_tps / generation_tps，包含其采样与解码流水线开销。
"""

import argparse
import random
import statistics
from contextlib import closing

import mlx.core as mx
from mlx_lm.sample_utils import make_sampler

from spark_mlx_llm import GenerationConfig, load, stream_generate


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def run_once(model, tokenizer, prompt, decode_tokens, kv_options):
    # 每次调用都由 stream_generate() 新建 KV cache，不复用上次的 prompt。
    mx.synchronize()
    mx.clear_cache()
    response = None
    try:
        with closing(
            stream_generate(
                model,
                tokenizer,
                prompt,
                max_tokens=decode_tokens,
                sampler=make_sampler(temp=0.0),
                **kv_options,
            )
        ) as responses:
            for response in responses:
                pass
    finally:
        mx.synchronize()
    if (
        response is None
        or response.prompt_tokens != prompt.size
        or response.generation_tokens != decode_tokens
        or response.finish_reason != "length"
    ):
        raise RuntimeError("Generation did not complete the requested token counts")
    return response.prompt_tps, response.generation_tps


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark prefill/decode tok/s at different prompt lengths."
    )
    parser.add_argument("--model", required=True, help="Model directory or HF repo ID")
    parser.add_argument(
        "--lengths",
        type=positive_int,
        nargs="+",
        default=[128, 512, 2048, 8192],
        help="Input token counts (default: 128 512 2048 8192)",
    )
    parser.add_argument("--decode-tokens", type=positive_int, default=256)
    parser.add_argument("--runs", type=positive_int, default=3)
    GenerationConfig.add_kv_arguments(parser)
    args = parser.parse_args()
    kv_options = GenerationConfig.from_options(vars(args)).kv_kwargs()

    # 与其他 Python 推理示例一致：已有 W8 层使用 native 实现。
    print(f"Loading {args.model} ...", flush=True)
    model, tokenizer, config = load(
        args.model, w8_qmv_backend="native", return_config=True
    )
    text_config = config.get("text_config", config)
    context_limit = text_config.get("max_position_embeddings")
    if context_limit and max(args.lengths) + args.decode_tokens > context_limit:
        parser.error(
            f"input + decode tokens exceed model context limit {context_limit}"
        )
    mx.eval(model.parameters())
    tokenizer.eos_token_ids = set()

    # 固定种子，各输入长度取同一序列的前缀；准备输入不计入吞吐。
    rng = random.Random(0)
    tokens = [
        rng.randrange(text_config["vocab_size"]) for _ in range(max(args.lengths))
    ]
    print(
        f"Device: {mx.default_device()} | Batch: 1 | Decode: {args.decode_tokens} | "
        f"Warmup: 1 per length | Runs: {args.runs} (median)",
        flush=True,
    )
    print(f"W8 backend: native | KV/prefill: {kv_options}", flush=True)
    print("Timing: MLX-LM prompt_tps / generation_tps | EOS: ignored", flush=True)
    print(
        f"{'Input tokens':>12} {'Prefill tok/s':>16} {'Decode tok/s':>16}", flush=True
    )
    for length in args.lengths:
        prompt = mx.array(tokens[:length], dtype=mx.uint32)
        mx.eval(prompt)
        run_once(model, tokenizer, prompt, args.decode_tokens, kv_options)
        samples = [
            run_once(model, tokenizer, prompt, args.decode_tokens, kv_options)
            for _ in range(args.runs)
        ]
        prefill = statistics.median(sample[0] for sample in samples)
        decode = statistics.median(sample[1] for sample in samples)
        print(f"{length:>12} {prefill:>16.2f} {decode:>16.2f}", flush=True)


if __name__ == "__main__":
    main()
