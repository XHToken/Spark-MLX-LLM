import argparse

import mlx.core as mx
from mlx_lm import generate
from mlx_lm.sample_utils import make_sampler

from .loader import load


def build_parser():
    parser = argparse.ArgumentParser(description="Generate text with Spark2_5")
    parser.add_argument("--model", required=True, help="Local path or HF repo")
    parser.add_argument("--prompt", "-p", default="你好")
    parser.add_argument("--max-tokens", "-m", type=int, default=128)
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="MLX execution device (default: auto)",
    )
    parser.add_argument(
        "--dtype",
        choices=("checkpoint", "float32", "bfloat16"),
        default="checkpoint",
        help="Weight dtype override (default: keep checkpoint dtype)",
    )
    parser.add_argument("--ignore-chat-template", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    if args.device == "cpu":
        mx.set_default_device(mx.cpu)
    elif args.device == "gpu":
        mx.set_default_device(mx.gpu)
    mx.random.seed(args.seed)

    model, tokenizer = load(
        args.model,
        dtype=None if args.dtype == "checkpoint" else args.dtype,
    )
    prompt = args.prompt
    if not args.ignore_chat_template and tokenizer.has_chat_template:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    sampler = make_sampler(
        temp=args.temp,
        top_p=args.top_p,
        top_k=args.top_k,
    )
    generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=args.max_tokens,
        sampler=sampler,
        verbose=True,
    )


if __name__ == "__main__":
    main()
