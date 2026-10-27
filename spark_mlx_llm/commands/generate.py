import argparse

import mlx.core as mx
from mlx_lm.sample_utils import make_sampler

from ..generation import GenerationConfig, generate
from ..loader import load
from ..ops.linear import LinearConfig
from ..ops.sdpa import SDPAConfig

__all__ = ["build_parser", "main"]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate text with Spark2_5", allow_abbrev=False
    )
    parser.add_argument("--model", required=True, help="Local path or HF repo")
    parser.add_argument("--prompt", "-p", default="你好")
    GenerationConfig.add_arguments(parser)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=("auto", "gpu"),
        default="auto",
        help="MLX GPU execution device (default: auto; requires Metal or CUDA)",
    )
    parser.add_argument(
        "--dtype",
        choices=("checkpoint", "float32", "bfloat16"),
        default="checkpoint",
        help="Weight dtype override (default: keep checkpoint dtype)",
    )
    parser.add_argument("--ignore-chat-template", action="store_true")
    parser.add_argument(
        "--w8-qmv-backend",
        choices=LinearConfig.choices("w8_qmv_backend"),
        default=None,
        help="Existing W8 out_proj/lm_head only: w8a16 (default), native, auto, w8a8; leaves W4 layers unchanged",
    )
    parser.add_argument(
        "--w4-qmv-backend",
        choices=LinearConfig.choices("w4_qmv_backend"),
        default=None,
        help="W4 QKV/down decode: native (default) or optimized splitk",
    )
    for name in (
        "fused_quantized_sdpa",
        "fused_quantized_sdpa_prefill",
        "tuned_bf16_sdpa",
    ):
        parser.add_argument(
            "--" + name.replace("_", "-"),
            action=argparse.BooleanOptionalAction,
            default=None,
            help=f"Model default: {getattr(SDPAConfig(), name)}",
        )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    inference = GenerationConfig.from_args(parser, args)
    if args.ignore_chat_template and inference.template_kwargs():
        parser.error(
            "Chat-template settings cannot be used with --ignore-chat-template"
        )
    if args.device == "gpu":
        mx.set_default_device(mx.gpu)
    if mx.default_device() != mx.gpu:
        parser.error("Spark inference requires a Metal or CUDA GPU backend")
    mx.random.seed(args.seed)

    model, tokenizer = load(
        args.model,
        dtype=None if args.dtype == "checkpoint" else args.dtype,
        w8_qmv_backend=args.w8_qmv_backend,
        w4_qmv_backend=args.w4_qmv_backend,
        model_config={
            name: value
            for name, value in vars(args).items()
            if name in SDPAConfig.__dataclass_fields__ and value is not None
        },
    )
    prompt = args.prompt
    if not args.ignore_chat_template and tokenizer.has_chat_template:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            **inference.template_kwargs(),
        )
    elif inference.template_kwargs():
        parser.error("The tokenizer has no chat template for the requested settings")

    sampler = make_sampler(**inference.sampler_kwargs())
    generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=inference.max_tokens,
        sampler=sampler,
        verbose=True,
        **inference.kv_kwargs(),
    )


if __name__ == "__main__":
    main()
