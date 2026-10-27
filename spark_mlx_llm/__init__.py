"""Spark conversion and inference APIs, loaded on demand to keep MLX optional."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .conversion import convert_to_mlx
    from .generation import GenerationConfig, generate, stream_generate
    from .loader import load
    from .model import Model, ModelArgs

__all__ = [
    "GenerationConfig",
    "Model",
    "ModelArgs",
    "convert_to_mlx",
    "generate",
    "load",
    "stream_generate",
]


def __getattr__(name):
    if name == "convert_to_mlx":
        from .conversion import convert_to_mlx

        globals()[name] = convert_to_mlx
    elif name == "GenerationConfig":
        from .generation import GenerationConfig

        globals()[name] = GenerationConfig
    elif name in {"Model", "ModelArgs"}:
        from .model import Model, ModelArgs

        globals().update(Model=Model, ModelArgs=ModelArgs)
    elif name in {"generate", "stream_generate"}:
        from .generation import generate, stream_generate

        globals().update(generate=generate, stream_generate=stream_generate)
    elif name == "load":
        from .loader import load

        globals()["load"] = load
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return globals()[name]


def __dir__():
    return sorted(set(globals()) | set(__all__))
