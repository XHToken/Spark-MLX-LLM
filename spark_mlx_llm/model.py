"""Spark2_5 configuration, module tree and forward orchestration.

Each operator owns its configuration, dispatch and implementations.
Checkpoint parameter names follow the original Spark2_5 module tree.
"""

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
from mlx import nn
from mlx_lm.models.base import (
    BaseModelArgs,
    create_attention_mask,
)
from mlx_lm.models.cache import KVCache, RotatingKVCache

from .ops import attention_output, ffn, linear, qkv, sdpa

__all__ = [
    "MLP",
    "Attention",
    "Model",
    "ModelArgs",
    "Spark2_5Model",
    "TransformerBlock",
]


@dataclass
class ModelArgs(BaseModelArgs, linear.LinearConfig, sdpa.SDPAConfig):
    model_type: str
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    sliding_window: int
    rope_parameters: dict[str, dict[str, Any]]
    layer_types: list[str] | None = None
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 8192
    attention_bias: bool = False
    mlp_bias: bool = False
    hidden_act: str = "gelu"
    gate_attn_act_mode: str = "sigmoid"
    headwise_attn_output_gate: bool = True
    tie_word_embeddings: bool = True
    quantization_config: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, params):
        # Reject the renamed key before BaseModelArgs drops unknown metadata.
        linear.LinearConfig.validate_values(params)
        return super().from_dict(params)

    def __post_init__(self):
        self.validate()
        for name in (
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "vocab_size",
            "sliding_window",
            "max_position_embeddings",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.layer_types is None:
            # Official checkpoints list every layer type; the reference
            # pattern is three sliding layers followed by one full layer.
            pattern = ["sliding_attention"] * 3 + ["full_attention"]
            self.layer_types = [
                pattern[i % len(pattern)] for i in range(self.num_hidden_layers)
            ]

        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                "layer_types length must match num_hidden_layers: "
                f"{len(self.layer_types)} != {self.num_hidden_layers}"
            )

        unsupported = set(self.layer_types) - {
            "sliding_attention",
            "full_attention",
        }
        if unsupported:
            raise ValueError(f"Unsupported layer types: {sorted(unsupported)}")

        missing_rope = set(self.layer_types) - set(self.rope_parameters)
        if missing_rope:
            raise ValueError(
                f"Missing RoPE parameters for layer types: {sorted(missing_rope)}"
            )

        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )

        for layer_type in set(self.layer_types):
            factor = float(
                self.rope_parameters[layer_type].get("partial_rotary_factor", 1.0)
            )
            dims = int(self.head_dim * factor)
            if dims <= 0 or dims > self.head_dim or dims % 2:
                raise ValueError(
                    f"Invalid rotary dimension {dims} for {layer_type} "
                    f"with head_dim={self.head_dim}"
                )

        if self.hidden_act != "gelu":
            raise ValueError(f"Spark2_5 requires GELU, got {self.hidden_act!r}")

        if not self.headwise_attn_output_gate:
            raise ValueError("Spark2_5 requires head-wise attention output gates")

        if self.gate_attn_act_mode != "sigmoid":
            raise ValueError(
                "Spark2_5 requires sigmoid attention gates, got "
                f"{self.gate_attn_act_mode!r}"
            )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
        self.sdpa_config = sdpa.SDPAConfig.from_dict(vars(args))
        self.linear_config = linear.LinearConfig.from_dict(vars(args))

        layer_type = args.layer_types[layer_idx]
        self.is_sliding = layer_type == "sliding_attention"

        q_size = self.n_heads * self.head_dim
        kv_size = self.n_kv_heads * self.head_dim
        self.q_size = q_size
        self.kv_size = kv_size

        self.q_k_v_proj = nn.Linear(
            args.hidden_size,
            q_size + 2 * kv_size,
            bias=args.attention_bias,
        )
        self.g_proj = nn.Linear(
            args.hidden_size,
            self.n_heads,
            bias=False,
        )
        output_projection = linear.QuantizedProjections.output_type(args, layer_idx)
        self.out_proj = output_projection(
            q_size,
            args.hidden_size,
            bias=args.attention_bias,
        )
        rope_params = args.rope_parameters[layer_type]
        self.rope = nn.RoPE(
            int(self.head_dim * float(rope_params.get("partial_rotary_factor", 1.0))),
            traditional=False,
            base=float(rope_params.get("rope_theta", 10000.0)),
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: Any | None = None,
    ) -> mx.array:
        # Cache offset is the absolute position, including rotating-cache wraps.
        queries, keys, values = qkv.dispatch_qkv(
            x,
            self.q_k_v_proj,
            self.rope,
            q_size=self.q_size,
            kv_size=self.kv_size,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim,
            offset=cache.offset if cache is not None else 0,
            config=self.linear_config,
        )

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = sdpa.dispatch_sdpa(
            queries,
            keys,
            values,
            cache=cache,
            scale=self.scale,
            mask=mask,
            config=self.sdpa_config,
            is_sliding=self.is_sliding,
        )
        return attention_output.dispatch_attention_output(
            x, output, self.g_proj, self.out_proj
        )


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.linear_config = linear.LinearConfig.from_dict(vars(args))
        self.gate_proj = nn.Linear(
            args.hidden_size,
            args.intermediate_size,
            bias=args.mlp_bias,
        )
        self.up_proj = nn.Linear(
            args.hidden_size,
            args.intermediate_size,
            bias=args.mlp_bias,
        )
        self.down_proj = nn.Linear(
            args.intermediate_size,
            args.hidden_size,
            bias=args.mlp_bias,
        )

    def __call__(self, x: mx.array) -> mx.array:
        hidden = nn.gelu(self.gate_proj(x)) * self.up_proj(x)
        return linear.dispatch_linear(hidden, self.down_proj, config=self.linear_config)


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(args, layer_idx)
        self.mlp = MLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    @property
    def is_sliding(self) -> bool:
        return self.self_attn.is_sliding

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: Any | None = None,
    ) -> mx.array:
        attention_output = self.self_attn(self.input_layernorm(x), mask, cache)
        return ffn.dispatch_ffn(
            x,
            attention_output,
            self.post_attention_layernorm,
            self.mlp,
            mlp_type=MLP,
        )


class Spark2_5Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embedding = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args, layer_idx)
            for layer_idx in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        self.first_full_layer = next(
            (
                idx
                for idx, layer_type in enumerate(args.layer_types)
                if layer_type == "full_attention"
            ),
            None,
        )
        self.first_sliding_layer = next(
            (
                idx
                for idx, layer_type in enumerate(args.layer_types)
                if layer_type == "sliding_attention"
            ),
            None,
        )

    def __call__(
        self,
        inputs: mx.array,
        cache: list[Any] | None = None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        hidden_states = (
            input_embeddings if input_embeddings is not None else self.embedding(inputs)
        )

        if cache is None:
            cache = [None] * len(self.layers)
        elif len(cache) != len(self.layers):
            raise ValueError(
                f"Expected {len(self.layers)} cache entries, got {len(cache)}"
            )

        full_mask = None
        if self.first_full_layer is not None:
            full_mask = create_attention_mask(
                hidden_states, cache[self.first_full_layer]
            )
        sliding_mask = None
        if self.first_sliding_layer is not None:
            sliding_mask = create_attention_mask(
                hidden_states,
                cache[self.first_sliding_layer],
                window_size=self.args.sliding_window,
            )

        for layer, layer_cache in zip(self.layers, cache):
            mask = sliding_mask if layer.is_sliding else full_mask
            hidden_states = layer(hidden_states, mask, layer_cache)

        return self.norm(hidden_states)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Spark2_5Model(args)
        if not args.tie_word_embeddings:
            head = linear.QuantizedProjections.head_type(args)
            self.lm_head = head(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: list[Any] | None = None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        hidden_states = self.model(inputs, cache, input_embeddings)
        if self.args.tie_word_embeddings:
            return self.model.embedding.as_linear(hidden_states)
        return self.lm_head(hidden_states)

    def load_weights(self, file_or_weights, strict=True):
        # Unwrap adapters so reloading accepts the original checkpoint names.
        linear.QuantizedProjections.unwrap(self)
        super().load_weights(file_or_weights, strict=strict)
        linear.QuantizedProjections.prepare(self)
        return self

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [
            (
                RotatingKVCache(max_size=self.args.sliding_window, keep=0)
                if layer.is_sliding
                else KVCache()
            )
            for layer in self.layers
        ]

    @property
    def quant_predicate(self):
        # The head-gate is a tiny per-head gating projection; keeping it at
        # source precision preserves the sigmoid gate values.
        def predicate(path, _):
            return not path.endswith("self_attn.g_proj")

        return predicate

    def sanitize(self, weights):
        # Some MLX conversions rename the embedding to embed_tokens while the
        # Hugging Face checkpoint ships it as model.embedding.weight.
        if (
            "model.embed_tokens.weight" in weights
            and "model.embedding.weight" not in weights
        ):
            weights["model.embedding.weight"] = weights.pop("model.embed_tokens.weight")
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        # Drop the serialized RoPE inverse-frequency buffer some conversions ship.
        return {
            name: value
            for name, value in weights.items()
            if "rotary_emb.inv_freq" not in name
        }
