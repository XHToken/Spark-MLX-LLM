"""Capture MLX-LM prefill and generation together as a Metal GPU trace."""

import argparse
import json
import os
import sys
from contextlib import contextmanager
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

__all__ = ["capture", "main"]


@contextmanager
def capture(path):
    import mlx.core as mx

    mx.synchronize()
    mx.metal.start_capture(str(path))
    try:
        yield
    finally:
        try:
            mx.synchronize()
        finally:
            mx.metal.stop_capture()
    print(f"Saved: {path}", flush=True)


def main():
    # Enable the capture layer before Python loads MLX or the Metal framework.
    if os.environ.get("MTL_CAPTURE_ENABLED") != "1":
        os.execve(
            sys.executable,
            [
                sys.executable,
                "-m",
                "spark_mlx_llm.benchmark.profile_metal",
                *sys.argv[1:],
            ],
            {**os.environ, "MTL_CAPTURE_ENABLED": "1"},
        )

    import mlx.core as mx

    from .bench import (
        add_arguments,
        kv_generation_options,
        load_benchmark,
        run_generation,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument(
        "--phase",
        choices=("generation", "both"),
        default="generation",
        help="Capture the complete generation; 'both' is a compatibility alias",
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = (
        (
            args.output_dir
            or Path(".build") / f"profile-{datetime.now():%Y%m%d-%H%M%S-%f}"
        )
        .expanduser()
        .resolve()
    )
    if output.exists():
        parser.error(f"output directory already exists: {output}")
    try:
        model, tokenizer, prompt = load_benchmark(args)
    except ValueError as error:
        parser.error(str(error))

    print("Warmup...", flush=True)
    kv_options = kv_generation_options(args)
    run_generation(
        model,
        tokenizer,
        prompt,
        args.decode_tokens,
        args.prefill_step_size,
        **kv_options,
    )
    output.mkdir(parents=True, exist_ok=False)
    print("Capturing MLX-LM prefill and generation...", flush=True)
    with capture(output / "generation.gputrace"):
        response = run_generation(
            model,
            tokenizer,
            prompt,
            args.decode_tokens,
            args.prefill_step_size,
            **kv_options,
        )
    metadata = {
        "model": args.model,
        "phase": "generation",
        "generator": "mlx_lm.stream_generate",
        "ignore_eos": True,
        "prompt_tokens": args.prompt_tokens,
        "decode_tokens": args.decode_tokens,
        "generated_tokens": response.generation_tokens,
        "prefill_step_size": args.prefill_step_size,
        **kv_options,
        "batch_size": 1,
        "seed": 0,
        "device": mx.device_info()["device_name"],
        "versions": {name: version(name) for name in ("mlx", "mlx-lm")},
        "captures": sorted(p.name for p in output.glob("*.gputrace")),
    }
    (output / "profile.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
