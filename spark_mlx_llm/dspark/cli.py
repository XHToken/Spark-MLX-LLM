"""Inspect checkpoints, generate text, and run paired DSpark benchmarks.

Argument parsing and checkpoint inspection require only the standard library.
Model execution dependencies are imported only by generate/benchmark commands.
"""

import argparse
import json
import sys
from dataclasses import asdict

from .checkpoint import inspect_checkpoint, validate_draft_shapes
from .config import DSparkConfig
from .metrics import GenerationResult
from .tokenization import eos_token_ids


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="Inspect local checkpoint metadata")
    inspect.add_argument("model", help="Local merged checkpoint directory")
    for command in ("generate", "benchmark"):
        runtime = commands.add_parser(command, help=f"Run DSpark {command}")
        runtime.add_argument("--model", required=True)
        runtime.add_argument("--prompt", "-p", default="你好")
        runtime.add_argument("--max-new-tokens", type=int, default=512)
        runtime.add_argument("--mode", choices=("baseline", "dspark"), default="dspark")
        runtime.add_argument("--max-draft-tokens", type=int)
        runtime.add_argument("--confidence-threshold", type=float, default=0.0)
        runtime.add_argument(
            "--greedy-verification",
            choices=("sequential", "batched"),
            default="batched",
            help="Batched verification (default); sequential is for correctness tests",
        )
        runtime.add_argument("--temperature", type=float, default=0.0)
        runtime.add_argument("--top-p", type=float, default=1.0)
        runtime.add_argument("--top-k", type=int, default=0)
        runtime.add_argument(
            "--draft-temperature",
            type=float,
            help="Draft temperature (default: inherit --temperature)",
        )
        runtime.add_argument(
            "--draft-top-p",
            type=float,
            help="Draft top-p (default: inherit --top-p)",
        )
        runtime.add_argument(
            "--draft-top-k",
            type=int,
            help="Draft top-k (default: inherit --top-k)",
        )
        runtime.add_argument("--seed", type=int, default=0)
        runtime.add_argument("--prefill-chunk-size", type=int, default=512)
        runtime.add_argument("--raw-prompt", action="store_true")
        runtime.add_argument("--dtype", choices=("bfloat16", "float32"))
        runtime.add_argument("--device", choices=("auto", "cpu", "gpu"), default="auto")
        runtime.add_argument(
            "--json", action="store_true", help="Print the result as JSON"
        )
        runtime.add_argument(
            "--stats", action="store_true", help="Print generation statistics to stderr"
        )
        runtime.add_argument("--trials", type=int, default=3)
        runtime.add_argument("--warmup-tokens", type=int, default=8)
    return parser


def _run_inspect(args):
    config = DSparkConfig.from_json(args.model)
    checkpoint = inspect_checkpoint(args.model)
    validate_draft_shapes(config, checkpoint)
    print(
        json.dumps(
            {
                "config": asdict(config),
                "checkpoint": checkpoint.summary(),
                "draft_shapes_valid": True,
                "prediction_layout_resolved": config.logits_start is not None,
            },
            indent=2,
        )
    )
    return 0


def _run_runtime(args):
    import mlx.core as mx

    from .benchmark import benchmark
    from .generation import stream_generate
    from .loader import load_pair
    from .sampling import SamplingOptions

    if args.device != "auto":
        mx.set_default_device(mx.cpu if args.device == "cpu" else mx.gpu)
    sampling = SamplingOptions(args.temperature, args.top_p, args.top_k)
    draft_sampling = sampling.with_overrides(
        temperature=args.draft_temperature,
        top_p=args.draft_top_p,
        top_k=args.draft_top_k,
    )
    pair = load_pair(args.model, dtype=args.dtype)
    options = {
        "max_new_tokens": args.max_new_tokens,
        "max_draft_tokens": args.max_draft_tokens,
        "confidence_threshold": args.confidence_threshold,
        "greedy_verification": args.greedy_verification,
        "sampling": sampling,
        "draft_sampling": draft_sampling,
        "seed": args.seed,
        "prefill_chunk_size": args.prefill_chunk_size,
        "apply_chat_template": not args.raw_prompt,
    }
    if args.command == "benchmark":
        report = benchmark(
            pair,
            args.prompt,
            trials=args.trials,
            warmup_tokens=args.warmup_tokens,
            **options,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1 if report["greedy_match"] is False else 0
    detokenizer = pair.tokenizer.detokenizer
    detokenizer.reset()
    eos = eos_token_ids(pair)
    for event in stream_generate(pair, args.prompt, mode=args.mode, **options):
        if isinstance(event, GenerationResult):
            result = event
        elif not args.json and event.token_id not in eos:
            detokenizer.add_token(event.token_id)
            print(detokenizer.last_segment, end="", flush=True)
    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        detokenizer.finalize()
        print(detokenizer.last_segment, flush=True)
        if args.stats:
            print(json.dumps(result.to_dict(), ensure_ascii=False), file=sys.stderr)
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # Preserve the original `python -m dspark /path/to/checkpoint` shorthand.
    if (
        argv
        and argv[0] not in ("inspect", "generate", "benchmark")
        and not argv[0].startswith("-")
    ):
        argv.insert(0, "inspect")
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return _run_inspect(args) if args.command == "inspect" else _run_runtime(args)
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        parser.exit(1, f"{error}\n")
