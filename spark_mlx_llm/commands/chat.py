# Chat loop adapted from MLX LM (Copyright © 2023-2024 Apple Inc.).
#
# MIT License
# Copyright © 2023 Apple Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Upstream-style terminal chat with Spark's shared inference settings."""

import argparse

import mlx.core as mx
from mlx_lm.chat import setup_arg_parser
from mlx_lm.generate import stream_generate
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler
from mlx_lm.utils import load, sharded_load

from ..generation import GenerationConfig
from ..registration import register_model


def build_parser():
    parser = argparse.ArgumentParser(
        parents=[setup_arg_parser()],
        add_help=False,
        conflict_handler="resolve",
        allow_abbrev=False,
        description="Chat with Spark using shared inference settings",
    )
    GenerationConfig.add_arguments(parser, include_kv=False)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    inference = GenerationConfig.from_args(parser, args, include_kv=False)
    register_model()
    group = mx.distributed.init()
    rank = group.rank()
    mx.random.seed(args.seed)

    if group.size() > 1:
        if args.adapter_path:
            parser.error("Adapters not supported in distributed mode")
        model, tokenizer = sharded_load(
            args.model,
            group if args.pipeline else None,
            None if args.pipeline else group,
        )
    else:
        model, tokenizer = load(
            args.model,
            adapter_path=args.adapter_path,
            tokenizer_config={
                "trust_remote_code": True if args.trust_remote_code else None
            },
        )

    def rprint(*values, **kwargs):
        if rank == 0:
            print(*values, **kwargs)

    def print_help():
        rprint("The command list: 'q' to exit, 'r' to reset, 'h' for help")

    rprint(f"[INFO] Starting chat session with {args.model}.")
    print_help()
    prompt_cache = make_prompt_cache(model, args.max_kv_size)
    while True:
        try:
            query = input(">> " if rank == 0 else "")
        except EOFError:
            break
        if query == "q":
            break
        if query == "r":
            prompt_cache = make_prompt_cache(model, args.max_kv_size)
            continue
        if query == "h":
            print_help()
            continue
        messages = []
        if args.system_prompt is not None:
            messages.append({"role": "system", "content": args.system_prompt})
        messages.append({"role": "user", "content": query})
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, **inference.template_kwargs()
        )
        for response in stream_generate(
            model,
            tokenizer,
            prompt,
            max_tokens=inference.max_tokens,
            sampler=make_sampler(
                **inference.sampler_kwargs(),
                xtc_threshold=args.xtc_threshold,
                xtc_probability=args.xtc_probability,
                xtc_special_tokens=tokenizer.encode("\n")
                + list(tokenizer.eos_token_ids),
            ),
            prompt_cache=prompt_cache,
        ):
            rprint(response.text, flush=True, end="")
        rprint()


if __name__ == "__main__":
    main()
