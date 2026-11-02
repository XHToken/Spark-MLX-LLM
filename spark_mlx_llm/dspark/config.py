"""Geometry and DSpark fields from a merged Spark checkpoint."""

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


def _check_integer(name: str, value: Any, minimum: int = 1) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")


@dataclass(frozen=True)
class DraftLayerSpec:
    """Explicit execution policy for one draft layer, separate from target config."""

    rope_theta: float
    rope_dims: int
    block_attention: str
    sliding_window: int | None = None
    rope_traditional: bool = False

    def __post_init__(self) -> None:
        if (
            isinstance(self.rope_theta, bool)
            or not isinstance(self.rope_theta, (int, float))
            or not math.isfinite(self.rope_theta)
            or self.rope_theta <= 0
        ):
            raise ValueError("rope_theta must be finite and positive")
        _check_integer("rope_dims", self.rope_dims)
        if self.rope_dims % 2:
            raise ValueError("rope_dims must be even")
        if self.block_attention not in ("bidirectional", "causal"):
            raise ValueError("block_attention must be 'bidirectional' or 'causal'")
        if self.sliding_window is not None:
            _check_integer("sliding_window", self.sliding_window)
        if type(self.rope_traditional) is not bool:
            raise ValueError("rope_traditional must be a boolean")


@dataclass(frozen=True)
class DSparkConfig:
    model_type: str
    hidden_size: int
    intermediate_size: int
    vocab_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    num_target_layers: int
    num_draft_layers: int
    block_size: int
    mask_token_id: int
    target_layer_ids: tuple[int, ...]
    draft_vocab_size: int
    markov_rank: int = 256
    markov_head_type: str = "vanilla"
    enable_confidence_head: bool = True
    confidence_head_with_markov: bool = True
    fc_norm: bool = False
    rms_norm_eps: float = 1e-6
    attention_bias: bool = False
    mlp_bias: bool = False
    hidden_act: str = "gelu"
    headwise_attn_output_gate: bool = True
    gate_attn_act_mode: str = "sigmoid"
    dspark_tie_word_embeddings: bool = False
    # The training forward must establish whether slot zero predicts a token.
    logits_start: int | None = None

    def __post_init__(self) -> None:
        if self.model_type != "spark2_5":
            raise ValueError(f"Unsupported model_type: {self.model_type!r}")
        for name in (
            "hidden_size",
            "intermediate_size",
            "vocab_size",
            "draft_vocab_size",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "num_target_layers",
            "num_draft_layers",
            "block_size",
        ):
            _check_integer(name, getattr(self, name))
        _check_integer("markov_rank", self.markov_rank, 0)
        _check_integer("mask_token_id", self.mask_token_id, 0)
        if self.mask_token_id >= self.vocab_size:
            raise ValueError("mask_token_id must be inside the target vocabulary")
        if self.draft_vocab_size > self.vocab_size:
            raise ValueError("draft_vocab_size must not exceed vocab_size")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )
        if not isinstance(self.target_layer_ids, (list, tuple)):
            raise TypeError("target_layer_ids must be a list or tuple")
        object.__setattr__(self, "target_layer_ids", tuple(self.target_layer_ids))
        if not self.target_layer_ids:
            raise ValueError("target_layer_ids must not be empty")
        for layer_id in self.target_layer_ids:
            _check_integer("target_layer_ids entry", layer_id, 0)
            if layer_id >= self.num_target_layers:
                raise ValueError(
                    "target_layer_ids entry exceeds the target layer count"
                )
        if len(set(self.target_layer_ids)) != len(self.target_layer_ids):
            raise ValueError("target_layer_ids must not contain duplicates")
        for name in (
            "enable_confidence_head",
            "confidence_head_with_markov",
            "fc_norm",
            "attention_bias",
            "mlp_bias",
            "headwise_attn_output_gate",
            "dspark_tie_word_embeddings",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        if (
            isinstance(self.rms_norm_eps, bool)
            or not isinstance(self.rms_norm_eps, (int, float))
            or not math.isfinite(self.rms_norm_eps)
            or self.rms_norm_eps <= 0
        ):
            raise ValueError("rms_norm_eps must be finite and positive")
        if self.markov_head_type != "vanilla":
            raise ValueError("Only the vanilla Markov head is supported")
        if (
            self.enable_confidence_head
            and self.confidence_head_with_markov
            and self.markov_rank == 0
        ):
            raise ValueError(
                "A confidence head with Markov features needs markov_rank > 0"
            )
        if self.hidden_act != "gelu":
            raise ValueError("Spark requires hidden_act='gelu'")
        if not self.headwise_attn_output_gate or self.gate_attn_act_mode != "sigmoid":
            raise ValueError("Spark requires sigmoid head-wise attention output gates")
        if self.logits_start is not None:
            _check_integer("logits_start", self.logits_start, 0)
            if self.logits_start not in (0, 1) or self.logits_start >= self.block_size:
                raise ValueError("logits_start must be 0 or 1 and below block_size")

    @property
    def q_size(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_size(self) -> int:
        return self.num_key_value_heads * self.head_dim

    @property
    def context_input_size(self) -> int:
        return len(self.target_layer_ids) * self.hidden_size

    @property
    def confidence_input_size(self) -> int:
        return self.hidden_size + (
            self.markov_rank if self.confidence_head_with_markov else 0
        )

    @property
    def max_draft_tokens(self) -> int:
        if self.logits_start is None:
            raise ValueError("logits_start is unresolved; check the training forward")
        return self.block_size - self.logits_start

    @classmethod
    def from_dict(cls, config: Mapping[str, Any]) -> "DSparkConfig":
        """Read merged-checkpoint fields without inferring draft attention or RoPE."""
        if not isinstance(config, Mapping):
            raise TypeError("The model config must be a JSON object")
        required = (
            "model_type",
            "hidden_size",
            "intermediate_size",
            "vocab_size",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "num_hidden_layers",
            "num_draft_layers",
            "dspark_block_size",
            "mask_token_id",
            "target_layer_ids",
        )
        missing = [name for name in required if name not in config]
        if missing:
            raise ValueError(f"Missing DSpark config fields: {missing}")
        values = {name: config[name] for name in required}
        values["num_target_layers"] = values.pop("num_hidden_layers")
        values["block_size"] = values.pop("dspark_block_size")
        values["draft_vocab_size"] = config.get(
            "draft_vocab_size", config["vocab_size"]
        )
        for name in cls.__dataclass_fields__:
            if name not in values and name in config:
                values[name] = config[name]
        return cls(**values)

    @classmethod
    def from_spark_dict(cls, raw: Mapping[str, Any]) -> "DSparkConfig":
        """Resolve the Spark vocabulary, confidence, and anchor prediction contract."""
        config = cls.from_dict(raw)
        if config.draft_vocab_size != config.vocab_size:
            raise ValueError(
                "The Spark DSpark profile requires the full target vocabulary"
            )
        if not config.confidence_head_with_markov:
            raise ValueError(
                "The Spark DSpark profile requires Markov confidence features"
            )
        from_anchor = raw.get("sample_from_anchor", True)
        if type(from_anchor) is not bool:
            raise ValueError("sample_from_anchor must be a boolean")
        start = 0 if from_anchor else 1
        if config.logits_start is not None and config.logits_start != start:
            raise ValueError("logits_start conflicts with sample_from_anchor")
        return replace(config, logits_start=start)

    @classmethod
    def from_json(cls, path: str | Path) -> "DSparkConfig":
        path = Path(path)
        if path.is_dir():
            path = path / "config.json"
        with path.open(encoding="utf-8") as stream:
            return cls.from_dict(json.load(stream))


def spark_layer_specs(config: Mapping[str, Any]) -> tuple[DraftLayerSpec, ...]:
    """Spark draft layers always use SWA, independently of layer-type metadata."""
    geometry = DSparkConfig.from_dict(config)
    rope_parameters = config.get("rope_parameters")
    if not isinstance(rope_parameters, Mapping):
        raise TypeError("Spark layer policies require rope_parameters")
    params = rope_parameters.get("sliding_attention")
    if not isinstance(params, Mapping):
        raise TypeError("Missing RoPE parameters for sliding_attention")
    factor = params.get("partial_rotary_factor", 1.0)
    if (
        isinstance(factor, bool)
        or not isinstance(factor, (int, float))
        or not math.isfinite(factor)
        or not 0 < factor <= 1
    ):
        raise ValueError("partial_rotary_factor must be finite and in (0, 1]")
    window = config.get("sliding_window")
    _check_integer("sliding_window", window)
    # Neither target layer_types nor draft_layer_types select draft attention.
    # Use the sliding-attention RoPE and bounded KV window for every draft layer.
    spec = DraftLayerSpec(
        rope_theta=params.get("rope_theta", 10000.0),
        rope_dims=int(geometry.head_dim * factor),
        block_attention="bidirectional",
        sliding_window=window,
    )
    return (spec,) * geometry.num_draft_layers
