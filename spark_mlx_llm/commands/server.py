"""Translate shared Spark settings into the upstream server's CLI options."""

import argparse
import json
import sys

from mlx_lm import server as mlx_server

from ..generation import GenerationConfig
from ..registration import register_model


def build_parser():
    parser = argparse.ArgumentParser(
        add_help=False,
        allow_abbrev=False,
        usage=argparse.SUPPRESS,
        description="Shared Spark configuration (generate, chat and server):",
    )
    GenerationConfig.add_arguments(parser, include_kv=False)
    return parser


def upstream_arguments(config):
    arguments = []
    for flag, value in (
        ("--max-tokens", config.max_tokens),
        ("--temp", config.temperature),
        ("--top-p", config.top_p),
        ("--top-k", config.top_k),
        ("--min-p", config.min_p),
        ("--chat-template-args", json.dumps(config.template_kwargs())),
    ):
        arguments.extend((flag, str(value)))
    return arguments


def main():
    parser = build_parser()
    args, remaining = parser.parse_known_args()
    inference = GenerationConfig.from_args(parser, args, include_kv=False)
    register_model()
    help_requested = "--help" in remaining or "-h" in remaining
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *upstream_arguments(inference), *remaining]
        return mlx_server.main()
    except SystemExit as error:
        if help_requested and error.code == 0:
            print()
            parser.print_help()
        raise
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
