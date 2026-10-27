"""Command-line conversion of local Spark BF16 and INT4 checkpoints."""

import argparse
import json
from pathlib import Path

from .. import conversion


def main(argv=None, *, prog=None):
    parser = argparse.ArgumentParser(
        prog=prog,
        description=conversion.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument(
        "--source", required=True, type=Path, help="Local model directory"
    )
    parser.add_argument("--output", type=Path, help="New MLX directory; must not exist")
    parser.add_argument(
        "--quantization",
        choices=("mixed", "none"),
        default="mixed",
        help="mixed: W4/G32 + per-channel W8; none: retain BF16 target",
    )
    parser.add_argument(
        "--w8-backend",
        choices=("w8a8", "w8a16", "native"),
        default="w8a8",
        help="Preferred W8 backend; unavailable optimized backends use MLX affine operators",
    )
    parser.add_argument(
        "--inspect", action="store_true", help="Validate metadata; write nothing"
    )
    parser.add_argument(
        "--chunk-rows", type=int, default=256, help="Rows processed on CPU at once"
    )
    args = parser.parse_args(argv)
    try:
        result = conversion.convert_to_mlx(
            args.source,
            args.output,
            quantization=args.quantization,
            w8_backend=args.w8_backend,
            inspect_only=args.inspect,
            chunk_rows=args.chunk_rows,
        )
    except (ValueError, OSError, KeyError, NotImplementedError) as error:
        parser.exit(1, f"Conversion failed: {error}\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k
                not in (
                    "coefficient_rounding",
                    "source_quantization_config",
                    "newly_quantized_modules",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
