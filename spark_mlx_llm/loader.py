from pathlib import Path
from typing import Any

import mlx.core as mx
from huggingface_hub import snapshot_download
from mlx.utils import tree_map_with_path
from mlx_lm.tokenizer_utils import load as load_tokenizer
from mlx_lm.utils import load_model

from .model import Model, ModelArgs
from .ops.linear import LinearConfig, QuantizedProjections
from .ops.sdpa import SDPAConfig

MODEL_ALLOW_PATTERNS = [
    "*.json",
    "model*.safetensors",
    "*.py",
    "tokenizer.model",
    "*.tiktoken",
    "tiktoken.model",
    "*.txt",
    "*.jsonl",
    "*.jinja",
]


def _resolve_model_path(path_or_repo: str | Path, revision: str | None):
    model_path = Path(path_or_repo).expanduser()
    if model_path.exists():
        if not model_path.is_dir():
            raise NotADirectoryError(f"Model path must be a directory: {model_path}")
        return model_path
    if (
        isinstance(path_or_repo, Path)
        or model_path.is_absolute()
        or str(path_or_repo).startswith(("~", "./", "../"))
    ):
        raise FileNotFoundError(f"Model directory does not exist: {model_path}")

    return Path(
        snapshot_download(
            str(path_or_repo),
            revision=revision,
            allow_patterns=MODEL_ALLOW_PATTERNS,
        )
    )


def _get_model_classes(config: dict):
    if config.get("model_type") != "spark2_5":
        raise ValueError(
            f"Expected model_type 'spark2_5', got {config.get('model_type')!r}"
        )
    return Model, ModelArgs


def _load_tokenizer(model_path, tokenizer_config, eos_token_ids):
    return load_tokenizer(
        model_path,
        tokenizer_config_extra=tokenizer_config,
        eos_token_ids=eos_token_ids,
    )


def load(
    path_or_repo: str | Path,
    *,
    tokenizer_config: dict[str, Any] | None = None,
    model_config: dict[str, Any] | None = None,
    lazy: bool = False,
    strict: bool = True,
    revision: str | None = None,
    dtype: str | None = None,
    w8_qmv_backend: str | None = None,
    w4_qmv_backend: str | None = None,
    return_config: bool = False,
):
    dtypes = {"float32": mx.float32, "bfloat16": mx.bfloat16}
    if dtype is not None and (not isinstance(dtype, str) or dtype not in dtypes):
        raise ValueError(f"Unsupported dtype {dtype!r}; choose from {sorted(dtypes)}")
    model_config = dict(model_config or {})
    if w4_qmv_backend is not None:
        model_config["w4_qmv_backend"] = w4_qmv_backend
    if w8_qmv_backend is not None:
        model_config["w8_qmv_backend"] = w8_qmv_backend
    LinearConfig.validate_values(model_config)
    SDPAConfig.validate_values(model_config)
    model_path = _resolve_model_path(path_or_repo, revision)
    model, config = load_model(
        model_path,
        lazy=lazy,
        strict=strict,
        model_config=model_config,
        get_model_classes=_get_model_classes,
    )
    if dtype is not None:
        target_dtype = dtypes[dtype]
        scale_paths = QuantizedProjections.scale_paths(model)
        model.update(
            tree_map_with_path(
                lambda path, value: (
                    value.astype(target_dtype)
                    if path not in scale_paths
                    and mx.issubdtype(value.dtype, mx.floating)
                    else value
                ),
                model.parameters(),
            )
        )
        if not lazy:
            mx.eval(model.parameters())
    tokenizer = _load_tokenizer(
        model_path,
        tokenizer_config or {},
        config.get("eos_token_id"),
    )

    if return_config:
        return model, tokenizer, config
    return model, tokenizer
