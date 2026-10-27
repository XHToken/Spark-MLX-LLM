"""Shared inference settings, generation defaults and mixed KV cache handling."""

import argparse
import json
import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

from mlx_lm import generate as mlx_generate
from mlx_lm import stream_generate as mlx_stream_generate
from mlx_lm.models.cache import QuantizedKVCache, RotatingKVCache
from mlx_lm.models.cache import make_prompt_cache as mlx_make_prompt_cache

__all__ = [
    "GenerationConfig",
    "generate",
    "make_prompt_cache",
    "stream_generate",
]

_KV_FIELDS = (
    "prefill_step_size",
    "kv_bits",
    "kv_group_size",
    "quantized_kv_start",
    "kv_quantization_scope",
)


def _json_object(value):
    try:
        result = json.loads(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"Invalid JSON: {error}") from error
    if not isinstance(result, dict):
        raise argparse.ArgumentTypeError("Expected a JSON object")
    return result


class _TemplateArguments(Mapping):
    """Read-only snapshot whose nested values are copied on access."""

    def __init__(self, values):
        self._values = deepcopy(dict(values))

    def __getitem__(self, name):
        return deepcopy(self._values[name])

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def __repr__(self):
        return repr(self._values)


@dataclass(frozen=True, kw_only=True)
class GenerationConfig:
    """Sampling, templates and Spark KV defaults, exported by purpose.

    None for thinking settings preserves the model template. KV caches retain
    model precision by default, matching chat/server's upstream behavior.
    KV quantization is opt-in and independent of checkpoint quantization.
    """

    max_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    enable_thinking: bool | None = None
    reasoning_effort: str | None = None
    chat_template_kwargs: Mapping = field(default_factory=dict)
    prefill_step_size: int = 512
    kv_bits: int | None = None
    kv_group_size: int = 32
    quantized_kv_start: int = 0
    kv_quantization_scope: str = "full-attention"

    def __post_init__(self):
        for name, minimum in (
            ("max_tokens", 1),
            ("top_k", 0),
            ("prefill_step_size", 1),
            ("quantized_kv_start", 0),
        ):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name in ("temperature", "top_p", "min_p"):
            value = getattr(self, name)
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value < 0
                or (name != "temperature" and value > 1)
            ):
                bounds = ">= 0" if name == "temperature" else "between 0 and 1"
                raise ValueError(f"{name} must be a finite number {bounds}")
        if not isinstance(self.chat_template_kwargs, Mapping):
            raise TypeError("chat_template_kwargs must be a JSON object")
        extras = dict(self.chat_template_kwargs)
        reserved = {
            "messages",
            "tools",
            "tokenize",
            "add_generation_prompt",
        } & extras.keys()
        if reserved:
            raise ValueError(f"Reserved chat template arguments: {sorted(reserved)}")
        # Normalize known template fields so later layers can override either form.
        for name in ("enable_thinking", "reasoning_effort"):
            if name in extras:
                value = extras.pop(name)
                if getattr(self, name) is None:
                    object.__setattr__(self, name, value)
        object.__setattr__(self, "chat_template_kwargs", _TemplateArguments(extras))
        if self.enable_thinking is not None and type(self.enable_thinking) is not bool:
            raise ValueError("enable_thinking must be true, false or null")
        if self.reasoning_effort not in (None, "low", "high"):
            raise ValueError("reasoning_effort must be 'low', 'high' or null")
        if self.kv_bits is not None and (
            type(self.kv_bits) is not int or self.kv_bits not in (4, 8)
        ):
            raise ValueError("kv_bits must be None, 4 or 8")
        if type(self.kv_group_size) is not int or self.kv_group_size not in (
            32,
            64,
            128,
        ):
            raise ValueError("kv_group_size must be 32, 64 or 128")
        if self.kv_quantization_scope not in ("all", "full-attention"):
            raise ValueError("kv_quantization_scope must be 'all' or 'full-attention'")

    def overlay(self, values):
        """Merge a higher-priority layer; named thinking options win within a layer."""
        if not isinstance(values, dict):
            raise TypeError("Generation configuration must be a JSON object")
        unknown = values.keys() - {item.name for item in fields(self)}
        if unknown:
            raise ValueError(f"Unknown generation settings: {sorted(unknown)}")
        changes = dict(values)
        extras = changes.pop("chat_template_kwargs", {})
        if not isinstance(extras, Mapping):
            raise TypeError("chat_template_kwargs must be a JSON object")
        extras = dict(extras)
        for name in ("enable_thinking", "reasoning_effort"):
            if name in extras:
                changes.setdefault(name, extras.pop(name))
        return replace(
            self,
            chat_template_kwargs={**self.chat_template_kwargs, **extras},
            **changes,
        )

    @classmethod
    def from_file(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_dict(cls, values):
        """Build a configuration, rejecting unknown fields and invalid values."""
        return cls().overlay(values)

    @classmethod
    def from_options(cls, values):
        """Extract configuration fields from mixed options, such as vars(args)."""
        if not isinstance(values, Mapping):
            raise TypeError("Generation options must be a mapping")
        return cls.from_dict(
            {f.name: values[f.name] for f in fields(cls) if f.name in values}
        )

    def sampler_kwargs(self):
        return {
            "temp": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
        }

    def template_kwargs(self):
        result = dict(self.chat_template_kwargs)
        for name in ("enable_thinking", "reasoning_effort"):
            value = getattr(self, name)
            if value is not None:
                result[name] = value
        return result

    @classmethod
    def add_arguments(cls, parser, *, include_kv=True):
        defaults = cls()
        group = parser.add_argument_group("shared inference settings")
        group.add_argument(
            "--config", help="Shared inference settings from a JSON file"
        )
        # Suppressed defaults distinguish explicit CLI values from file defaults.
        for flags, name, kind in (
            (("--max-tokens", "-m"), "max_tokens", int),
            (("--temp", "--temperature"), "temperature", float),
            (("--top-p",), "top_p", float),
            (("--top-k",), "top_k", int),
            (("--min-p",), "min_p", float),
        ):
            group.add_argument(
                *flags,
                dest=name,
                type=kind,
                default=argparse.SUPPRESS,
                help=f"{name} (default: {getattr(defaults, name)})",
            )
        group.add_argument(
            "--thinking",
            dest="enable_thinking",
            action=argparse.BooleanOptionalAction,
            default=argparse.SUPPRESS,
            help="Enable/disable thinking (default: model chat template)",
        )
        group.add_argument(
            "--reasoning-effort",
            choices=("low", "high"),
            default=argparse.SUPPRESS,
            help="Thinking effort (default: model chat template)",
        )
        group.add_argument(
            "--chat-template-args",
            dest="chat_template_kwargs",
            type=_json_object,
            default=argparse.SUPPRESS,
            help="Additional chat-template arguments as a JSON object",
        )
        if include_kv:
            cls.add_kv_arguments(parser, suppress_defaults=True)

    @classmethod
    def from_args(cls, parser, args, *, include_kv=True):
        try:
            values = (
                json.loads(Path(args.config).read_text(encoding="utf-8"))
                if args.config
                else {}
            )
            if not include_kv and isinstance(values, dict):
                unsupported = values.keys() & set(_KV_FIELDS)
                if unsupported:
                    raise ValueError(
                        "KV/prefill settings in --config are only supported by "
                        f"spark-mlx-generate: {sorted(unsupported)}"
                    )
            config = cls.from_dict(values)
            values = vars(args)
            return config.overlay(
                {
                    item.name: values[item.name]
                    for item in fields(cls)
                    if item.name in values
                    and (include_kv or item.name not in _KV_FIELDS)
                }
            )
        except (OSError, TypeError, ValueError) as error:
            parser.error(str(error))

    @classmethod
    def add_kv_arguments(cls, parser, *, suppress_defaults=False):
        """Register only KV/prefill flags for benchmarks and dataset runners."""
        defaults = cls()
        group = parser.add_argument_group("Spark KV and prefill settings")
        group.add_argument(
            "--prefill-step-size",
            type=_positive_int,
            default=argparse.SUPPRESS
            if suppress_defaults
            else defaults.prefill_step_size,
            help=f"Prompt chunk size (default: {defaults.prefill_step_size})",
        )
        group.add_argument(
            "--kv-bits",
            type=_kv_bits,
            choices=(None, 4, 8),
            metavar="{none,4,8}",
            default=argparse.SUPPRESS if suppress_defaults else defaults.kv_bits,
            help="KV quantization bits (default: none); 'none' keeps model precision",
        )
        group.add_argument(
            "--kv-group-size",
            type=int,
            choices=(32, 64, 128),
            default=argparse.SUPPRESS if suppress_defaults else defaults.kv_group_size,
        )
        group.add_argument(
            "--quantized-kv-start",
            type=_nonnegative_int,
            default=argparse.SUPPRESS
            if suppress_defaults
            else defaults.quantized_kv_start,
        )
        group.add_argument(
            "--kv-quantization-scope",
            choices=("all", "full-attention"),
            default=(
                argparse.SUPPRESS
                if suppress_defaults
                else defaults.kv_quantization_scope
            ),
            help="When KV quantization is enabled, default to full-attention layers only",
        )

    def kv_kwargs(self):
        """Options for this package's generate/stream_generate, excluding sampling."""
        return {name: getattr(self, name) for name in _KV_FIELDS}

    def prepare_kv_kwargs(self, model, *, prompt_cache=None, max_kv_size=None):
        """Prepare upstream options, retaining MLX-LM's conversion timing."""
        options = self.kv_kwargs()
        options.pop("kv_quantization_scope")
        if self.kv_bits is not None:
            prompt_cache = make_prompt_cache(
                model,
                self.kv_quantization_scope,
                prompt_cache=prompt_cache,
                max_kv_size=max_kv_size,
            )
        if prompt_cache is not None:
            options["prompt_cache"] = prompt_cache
        return options


class _UnquantizedRotatingKVCache(RotatingKVCache):
    """Preserve sliding-window state during upstream KV quantization checks."""

    def to_quantized(self, group_size=32, bits=8):
        return self


def make_prompt_cache(
    model,
    kv_quantization_scope="full-attention",
    *,
    prompt_cache=None,
    max_kv_size=None,
):
    """Prepare mixed caches; conversion still happens at MLX-LM's step boundaries.

    A supplied cache list is updated in place so repeated generation continues
    the same cache. Newly created caches preserve all rotating-cache state.
    """
    if kv_quantization_scope not in ("all", "full-attention"):
        raise ValueError(f"Unknown KV quantization scope: {kv_quantization_scope}")
    caches = (
        mlx_make_prompt_cache(model, max_kv_size)
        if prompt_cache is None
        else prompt_cache
    )
    replacements = []
    for index, cache in enumerate(caches):
        if isinstance(cache, RotatingKVCache):
            if kv_quantization_scope == "all":
                raise ValueError(
                    "KV quantization scope 'all' includes unsupported RotatingKVCache; "
                    "use --kv-quantization-scope full-attention"
                )
            if type(cache) is _UnquantizedRotatingKVCache:
                continue
            if type(cache) is not RotatingKVCache:
                raise ValueError(
                    "KV quantization scope 'full-attention' cannot preserve custom "
                    f"RotatingKVCache subclass {type(cache).__name__}"
                )
            preserved = _UnquantizedRotatingKVCache.__new__(_UnquantizedRotatingKVCache)
            preserved.__dict__.update(cache.__dict__)
            replacements.append((index, preserved))
        elif not hasattr(cache, "to_quantized") and not isinstance(
            cache, QuantizedKVCache
        ):
            raise ValueError(
                f"KV cache layer {index} ({type(cache).__name__}) does not support KV quantization"
            )
    for index, cache in replacements:
        caches[index] = cache
    return caches


def _prepare_generation_kwargs(model, kwargs):
    # Only consume KV settings: max_tokens, sampler and all other upstream options
    # retain their existing values and validation semantics (e.g. max_tokens=-1).
    kwargs = dict(kwargs)
    config = GenerationConfig(
        **{name: kwargs.pop(name) for name in _KV_FIELDS if name in kwargs}
    )
    # MLX-LM speculative decoding expects one combined target/draft cache list.
    if config.kv_bits is not None and kwargs.get("prompt_cache") is None:
        draft = kwargs.get("draft_model")
        if draft is not None:
            kwargs["prompt_cache"] = mlx_make_prompt_cache(
                model
            ) + mlx_make_prompt_cache(draft)
    kwargs.update(
        config.prepare_kv_kwargs(
            model,
            prompt_cache=kwargs.get("prompt_cache"),
            max_kv_size=kwargs.get("max_kv_size"),
        )
    )
    return kwargs


def generate(
    model,
    tokenizer,
    prompt,
    verbose=False,
    *,
    max_tokens=GenerationConfig.max_tokens,
    **kwargs,
):
    """Generate up to 512 tokens, keeping model KV precision unless kv_bits is set."""
    return mlx_generate(
        model,
        tokenizer,
        prompt,
        verbose=verbose,
        max_tokens=max_tokens,
        **_prepare_generation_kwargs(model, kwargs),
    )


def stream_generate(
    model,
    tokenizer,
    prompt,
    max_tokens=GenerationConfig.max_tokens,
    draft_model=None,
    **kwargs,
):
    """Stream up to 512 tokens, keeping model KV precision unless kv_bits is set."""
    return mlx_stream_generate(
        model,
        tokenizer,
        prompt,
        max_tokens=max_tokens,
        **_prepare_generation_kwargs(model, {**kwargs, "draft_model": draft_model}),
    )


def _positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def _nonnegative_int(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return value


def _kv_bits(value):
    return None if value.lower() == "none" else int(value)
