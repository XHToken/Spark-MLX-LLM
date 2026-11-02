"""Draft network, feature fusion, Markov head, and confidence head.

The backbone queries from the draft block and projects keys/values from both
the fused target features and the block. Layer policies and readout binding
are explicit because the merged checkpoint does not establish them.
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import mlx.core as mx
from mlx import nn

from .cache import DraftKVCache
from .config import DraftLayerSpec, DSparkConfig, spark_layer_specs


def _hidden_shape(hidden: mx.array, width: int, name: str) -> None:
    if hidden.ndim != 3 or hidden.shape[-1] != width:
        raise ValueError(f"{name} must have shape [batch, length, {width}]")


def _token_shape(tokens: mx.array, shape: tuple[int, ...]) -> None:
    if tokens.shape != shape or not mx.issubdtype(tokens.dtype, mx.integer):
        raise ValueError(f"Token IDs must be an integer array with shape {shape}")


def _offset(value: int, name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


class SparkDraftAttention(nn.Module):
    def __init__(self, config: DSparkConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.q_size = config.q_size
        self.kv_size = config.kv_size
        self.q_k_v_proj = nn.Linear(
            config.hidden_size,
            config.q_size + 2 * config.kv_size,
            bias=config.attention_bias,
        )
        self.g_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads, bias=config.attention_bias
        )
        self.out_proj = nn.Linear(
            config.q_size, config.hidden_size, bias=config.attention_bias
        )
        self.spec = None
        self.rope = None

    def configure(self, spec: DraftLayerSpec) -> None:
        if not isinstance(spec, DraftLayerSpec):
            raise TypeError("Draft attention needs a DraftLayerSpec")
        if spec.rope_dims > self.head_dim:
            raise ValueError("Draft rope_dims must not exceed head_dim")
        self.rope = nn.RoPE(
            spec.rope_dims,
            traditional=spec.rope_traditional,
            base=spec.rope_theta,
        )
        self.spec = spec

    def _heads(self, projected: mx.array, count: int) -> mx.array:
        batch, length, _ = projected.shape
        return projected.reshape(batch, length, count, self.head_dim).transpose(
            0, 2, 1, 3
        )

    def _mask(self, hidden, context_length, block_offset, context_offset, mask):
        batch, length, _ = hidden.shape
        total = context_length + length
        if mask is not None:
            shape = (batch, self.num_heads, length, total)
            if (
                mask.dtype != mx.bool_
                or mask.ndim > 4
                or any(a not in (1, b) for a, b in zip(mask.shape[::-1], shape[::-1]))
            ):
                raise ValueError(
                    "Attention mask must be boolean and broadcast to [B,H,Q,K]"
                )
        if self.spec.block_attention == "causal" or self.spec.sliding_window:
            queries = block_offset + mx.arange(length)[:, None]
            keys = mx.concatenate(
                [
                    context_offset + mx.arange(context_length),
                    block_offset + mx.arange(length),
                ]
            )[None, :]
            allowed = mx.ones((length, total), dtype=mx.bool_)
            if self.spec.block_attention == "causal":
                allowed = allowed & (keys <= queries)
            if self.spec.sliding_window is not None:
                # The window counts committed history before the anchor. Every
                # slot sees that same history; block visibility is controlled
                # separately by block_attention, even when length > window.
                allowed = allowed & (
                    keys >= block_offset - self.spec.sliding_window
                )
            mask = allowed if mask is None else allowed & mask
        return mask

    def __call__(
        self,
        hidden: mx.array,
        context: mx.array,
        *,
        block_offset: int,
        context_offset: int = 0,
        mask: mx.array | None = None,
    ) -> mx.array:
        if self.spec is None:
            raise RuntimeError(
                "Draft attention/RoPE is unresolved; configure layer specs"
            )
        _hidden_shape(hidden, self.hidden_size, "hidden")
        _hidden_shape(context, self.hidden_size, "context")
        _offset(block_offset, "block_offset")
        _offset(context_offset, "context_offset")
        if hidden.shape[0] != context.shape[0]:
            raise ValueError("Draft block and context must have the same batch size")
        if context_offset + context.shape[1] > block_offset:
            raise ValueError("Target context positions must precede the draft block")
        context_k, context_v = self.project_context(
            context, context_offset=context_offset
        )
        return self._attend(
            hidden, context_k, context_v, block_offset, context_offset, mask
        )

    def project_context(self, context, *, context_offset):
        if self.spec is None:
            raise RuntimeError(
                "Draft attention/RoPE is unresolved; configure layer specs"
            )
        _, context_k, context_v = mx.split(
            self.q_k_v_proj(context), [self.q_size, self.q_size + self.kv_size], axis=-1
        )
        context_k = self.rope(
            self._heads(context_k, self.num_kv_heads), offset=context_offset
        )
        return context_k, self._heads(context_v, self.num_kv_heads)

    def from_cache(self, hidden, cache, *, block_offset):
        if self.spec is None:
            raise RuntimeError(
                "Draft attention/RoPE is unresolved; configure layer specs"
            )
        if cache.offset != block_offset:
            raise ValueError("Draft context offset differs from the block position")
        if cache.keys is None:
            keys = values = mx.zeros(
                (hidden.shape[0], self.num_kv_heads, 0, self.head_dim),
                dtype=hidden.dtype,
            )
        else:
            keys, values = cache.keys, cache.values
        return self._attend(hidden, keys, values, block_offset, cache.start, None)

    def _attend(self, hidden, context_k, context_v, block_offset, context_offset, mask):
        q, block_k, block_v = mx.split(
            self.q_k_v_proj(hidden), [self.q_size, self.q_size + self.kv_size], axis=-1
        )
        q = self.rope(self._heads(q, self.num_heads), offset=block_offset)
        block_k = self.rope(
            self._heads(block_k, self.num_kv_heads), offset=block_offset
        )
        k = mx.concatenate([context_k, block_k], axis=2)
        v = mx.concatenate(
            [
                context_v,
                self._heads(block_v, self.num_kv_heads),
            ],
            axis=2,
        )
        dtype = mx.result_type(q, k, v)
        attended = mx.fast.scaled_dot_product_attention(
            q.astype(dtype),
            k.astype(dtype),
            v.astype(dtype),
            scale=self.head_dim**-0.5,
            mask=self._mask(
                hidden, context_k.shape[2], block_offset, context_offset, mask
            ),
        )
        attended = attended.transpose(0, 2, 1, 3)
        gate = mx.sigmoid(self.g_proj(hidden).astype(mx.float32)).astype(attended.dtype)
        attended = attended * gate[..., None]
        return self.out_proj(attended.reshape(*hidden.shape[:2], self.q_size))


class SparkDraftMLP(nn.Module):
    def __init__(self, config: DSparkConfig):
        super().__init__()
        h, intermediate = config.hidden_size, config.intermediate_size
        self.gate_proj = nn.Linear(h, intermediate, bias=config.mlp_bias)
        self.up_proj = nn.Linear(h, intermediate, bias=config.mlp_bias)
        self.down_proj = nn.Linear(intermediate, h, bias=config.mlp_bias)

    def __call__(self, hidden: mx.array) -> mx.array:
        return self.down_proj(nn.gelu(self.gate_proj(hidden)) * self.up_proj(hidden))


class SparkDraftLayer(nn.Module):
    def __init__(self, config: DSparkConfig):
        super().__init__()
        self.self_attn = SparkDraftAttention(config)
        self.mlp = SparkDraftMLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(self, hidden, context, *, block_offset, context_offset=0, mask=None):
        hidden = hidden + self.self_attn(
            self.input_layernorm(hidden),
            context,
            block_offset=block_offset,
            context_offset=context_offset,
            mask=mask,
        )
        return hidden + self.mlp(self.post_attention_layernorm(hidden))

    def from_cache(self, hidden, cache, *, block_offset):
        hidden = hidden + self.self_attn.from_cache(
            self.input_layernorm(hidden), cache, block_offset=block_offset
        )
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class VanillaMarkovHead(nn.Module):
    def __init__(self, config: DSparkConfig):
        super().__init__()
        if config.markov_rank <= 0:
            raise ValueError("A Markov head requires markov_rank > 0")
        self.markov_w1 = nn.Embedding(config.vocab_size, config.markov_rank)
        self.markov_w2 = nn.Linear(
            config.markov_rank, config.draft_vocab_size, bias=False
        )

    def prev_embeddings(self, token_ids: mx.array) -> mx.array:
        return self.markov_w1(token_ids)

    def step_bias(self, token_ids: mx.array) -> mx.array:
        return self.markov_w2(self.prev_embeddings(token_ids))


class ConfidenceHead(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.proj = nn.Linear(input_size, 1, bias=True)

    def __call__(self, features: mx.array) -> mx.array:
        return self.proj(features).squeeze(-1)


class DSparkDrafter(nn.Module):
    """Spark draft backbone with checkpoint-compatible parameter names.

    Construction needs only geometry. Execution needs explicit layer specs.
    __call__ returns all normalized block hidden states; projection, Markov
    correction and confidence are separate operations.
    """

    def __init__(
        self,
        config: DSparkConfig,
        *,
        layer_specs: Sequence[DraftLayerSpec] | None = None,
        readout_mode: str = "unresolved",
    ):
        super().__init__()
        if readout_mode not in ("unresolved", "external", "embedding", "linear"):
            raise ValueError("Unknown draft readout_mode")
        if readout_mode == "embedding" and config.draft_vocab_size != config.vocab_size:
            raise ValueError("Embedding readout requires the full target vocabulary")
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.fc_norm = (
            [
                nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
                for _ in config.target_layer_ids
            ]
            if config.fc_norm
            else None
        )
        self.fc = nn.Linear(config.context_input_size, config.hidden_size, bias=False)
        self.hidden_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers = [SparkDraftLayer(config) for _ in range(config.num_draft_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.markov_head = VanillaMarkovHead(config) if config.markov_rank else None
        self.confidence_head = (
            ConfidenceHead(config.confidence_input_size)
            if config.enable_confidence_head
            else None
        )
        self.lm_head = (
            nn.Linear(config.hidden_size, config.draft_vocab_size, bias=False)
            if readout_mode == "linear"
            else None
        )
        self._readout_mode = readout_mode
        # External model parameters must not become part of the drafter's parameter tree.
        object.__setattr__(self, "_external_lm_head", None)
        if layer_specs is not None:
            self.configure_layers(layer_specs)

    @classmethod
    def from_spark_config(
        cls, raw: Mapping[str, Any], *, readout_mode: str = "external"
    ) -> "DSparkDrafter":
        """Apply the Spark contract in llama.cpp PR #31168 (053208b).

        Tap IDs denote target layer outputs. The default readout is supplied by
        the target model; bind_lm_head accepts its projection without importing it.
        """
        return cls(
            DSparkConfig.from_spark_dict(raw),
            layer_specs=spark_layer_specs(raw),
            readout_mode=readout_mode,
        )

    def configure_layers(self, specs: Sequence[DraftLayerSpec]) -> None:
        specs = tuple(specs)
        if len(specs) != len(self.layers):
            raise ValueError("Need one DraftLayerSpec per draft layer")
        for spec in specs:
            if not isinstance(spec, DraftLayerSpec):
                raise TypeError("Each draft layer needs a DraftLayerSpec")
            if spec.rope_dims > self.config.head_dim:
                raise ValueError("Draft rope_dims must not exceed head_dim")
        for layer, spec in zip(self.layers, specs):
            layer.self_attn.configure(spec)

    def fuse_target(self, target_hidden_states: Sequence[mx.array]) -> mx.array:
        """Project taps supplied in config.target_layer_ids order."""
        taps = tuple(target_hidden_states)
        if len(taps) != len(self.config.target_layer_ids):
            raise ValueError("Need one target hidden-state tensor per configured tap")
        for tap in taps:
            _hidden_shape(tap, self.config.hidden_size, "target tap")
            if tap.shape != taps[0].shape:
                raise ValueError("All target taps must have the same shape")
        if self.fc_norm is not None:
            taps = tuple(norm(tap) for norm, tap in zip(self.fc_norm, taps))
        return self.hidden_norm(self.fc(mx.concatenate(taps, axis=-1)))

    def make_context_cache(self):
        if any(layer.self_attn.spec is None for layer in self.layers):
            raise RuntimeError("Configure draft layers before creating context caches")
        return [
            DraftKVCache(layer.self_attn.spec.sliding_window) for layer in self.layers
        ]

    def update_context(self, taps, caches):
        if len(caches) != len(self.layers) or len({c.offset for c in caches}) != 1:
            raise ValueError("Invalid draft context cache state")
        fused = self.fuse_target(taps)
        if fused.shape[1] == 0:
            return
        for layer, cache in zip(self.layers, caches):
            keys, values = layer.self_attn.project_context(
                fused, context_offset=cache.offset
            )
            cache.append(keys, values)

    def forward_cached(self, block_tokens, caches):
        if len(caches) != len(self.layers) or len({c.offset for c in caches}) != 1:
            raise ValueError("Invalid draft context cache state")
        if block_tokens.ndim != 2 or block_tokens.shape[1] != self.config.block_size:
            raise ValueError("Draft tokens must have shape [batch, block_size]")
        _token_shape(block_tokens, block_tokens.shape)
        hidden = self.embed_tokens(block_tokens)
        block_offset = caches[0].offset
        for layer, cache in zip(self.layers, caches):
            hidden = layer.from_cache(hidden, cache, block_offset=block_offset)
        return self.norm(hidden)

    def __call__(
        self,
        block_tokens: mx.array,
        target_hidden_states: Sequence[mx.array],
        *,
        block_offset: int,
        context_offset: int = 0,
        mask: mx.array | None = None,
    ) -> mx.array:
        if any(layer.self_attn.spec is None for layer in self.layers):
            raise RuntimeError(
                "Draft attention/RoPE is unresolved; configure layer specs"
            )
        if block_tokens.ndim != 2 or block_tokens.shape[1] != self.config.block_size:
            raise ValueError("Draft tokens must have shape [batch, block_size]")
        _token_shape(block_tokens, block_tokens.shape)
        context = self.fuse_target(target_hidden_states)
        hidden = self.embed_tokens(block_tokens)
        for layer in self.layers:
            hidden = layer(
                hidden,
                context,
                block_offset=block_offset,
                context_offset=context_offset,
                mask=mask,
            )
        return self.norm(hidden)

    def prediction_hidden(self, block_hidden: mx.array) -> mx.array:
        _hidden_shape(block_hidden, self.config.hidden_size, "block_hidden")
        if block_hidden.shape[1] != self.config.block_size:
            raise ValueError("Expected all trained block positions")
        _ = self.config.max_draft_tokens
        return block_hidden[:, self.config.logits_start :, :]

    def bind_lm_head(self, project: Callable[[mx.array], mx.array]) -> None:
        if self._readout_mode not in ("unresolved", "external"):
            raise ValueError(
                "External readout cannot replace a linear or embedding head"
            )
        if not callable(project):
            raise TypeError("External LM head must be callable")
        object.__setattr__(self, "_external_lm_head", project)
        self._readout_mode = "external"

    def compute_logits(self, hidden: mx.array) -> mx.array:
        if hidden.shape[-1] != self.config.hidden_size:
            raise ValueError("LM-head input has the wrong hidden size")
        if self._readout_mode == "linear":
            logits = self.lm_head(hidden)
        elif self._readout_mode == "embedding":
            logits = self.embed_tokens.as_linear(hidden)
        elif self._readout_mode == "external" and self._external_lm_head is not None:
            logits = self._external_lm_head(hidden)
        else:
            raise RuntimeError("Draft readout is unresolved; choose or bind an LM head")
        if logits.shape != (*hidden.shape[:-1], self.config.draft_vocab_size):
            raise ValueError("LM-head output has the wrong vocabulary shape")
        return logits

    def add_markov_bias(
        self, base_logits: mx.array, prev_token_ids: mx.array
    ) -> mx.array:
        if base_logits.shape[-1] != self.config.draft_vocab_size:
            raise ValueError("Draft logits have the wrong vocabulary size")
        _token_shape(prev_token_ids, base_logits.shape[:-1])
        if self.markov_head is None:
            return base_logits
        return base_logits + self.markov_head.step_bias(prev_token_ids)

    def confidence_logits(
        self, hidden: mx.array, prev_token_ids: mx.array | None = None
    ) -> mx.array:
        if self.confidence_head is None:
            raise RuntimeError("The draft confidence head is disabled")
        if hidden.shape[-1] != self.config.hidden_size:
            raise ValueError("Confidence input has the wrong hidden size")
        if self.config.confidence_head_with_markov:
            if prev_token_ids is None:
                raise ValueError(
                    "Confidence with Markov features needs previous token IDs"
                )
            _token_shape(prev_token_ids, hidden.shape[:-1])
            hidden = mx.concatenate(
                [hidden, self.markov_head.prev_embeddings(prev_token_ids)], axis=-1
            )
        return self.confidence_head(hidden)
