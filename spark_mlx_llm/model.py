from dataclasses import dataclass
from functools import cache
from typing import Any

import mlx.core as mx
from mlx import nn
from mlx_lm.models.base import (
    BaseModelArgs,
    create_attention_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models.cache import KVCache, RotatingKVCache

try:
    from mlx_spark_ext import spark_qkv_rope as _native_qkv_rope
except ImportError:
    _native_qkv_rope = None


@dataclass
class ModelArgs(BaseModelArgs):
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

    def __post_init__(self):
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

        if self.num_attention_heads * self.head_dim <= 0:
            raise ValueError("num_attention_heads * head_dim must be positive")

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


def _make_rope(args: ModelArgs, layer_type: str):
    params = args.rope_parameters[layer_type]
    base = float(params.get("rope_theta", 10000.0))
    factor = float(params.get("partial_rotary_factor", 1.0))
    dims = int(args.head_dim * factor)
    return nn.RoPE(dims, traditional=False, base=base)


def _validate_w8a8_activations(activations):
    expected = {
        "num_bits": 8,
        "type": "int",
        "strategy": "token",
        "symmetric": True,
        "dynamic": True,
    }
    if (
        not isinstance(activations, dict)
        or activations.get("num_bits") != 8
        or any(activations.get(key, value) != value for key, value in expected.items())
    ):
        raise ValueError(
            "W8A8 out_proj requires symmetric dynamic per-token INT8 input_activations"
        )


def _out_proj_uses_w8a8(args: ModelArgs, layer_idx: int) -> bool:
    config = args.quantization_config or {}
    scheme = config.get(f"model.layers.{layer_idx}.self_attn.out_proj", {})
    if not isinstance(scheme, dict) or scheme.get("input_activations") is None:
        return False
    _validate_w8a8_activations(scheme["input_activations"])
    if (
        scheme.get("bits", config.get("bits", 4)) != 8
        or scheme.get("mode", config.get("mode", "affine")) != "affine"
    ):
        raise ValueError("W8A8 out_proj requires 8-bit affine packed weights")
    return True


def _w8a8_operation():
    try:
        from mlx_spark_ext import is_w8a8_available, w8a8_linear
    except ImportError:
        return None
    if not is_w8a8_available():
        return None
    return w8a8_linear


class _W8A8QuantizedLinear(nn.QuantizedLinear):
    """Keep MLX affine checkpoint parameters; cache native INT8 kernel inputs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._operation = _w8a8_operation()
        if self._operation is None:
            raise RuntimeError("Cannot create a native W8A8 layer without W8A8 support")
        # Derived buffers are not checkpoint parameters.
        object.__setattr__(self, "_w8a8_cache", None)

    def _prepare_w8a8(self):
        sources = (self.weight, self.scales, self.biases)
        cached = self._w8a8_cache
        if cached is not None and all(
            old is new for old, new in zip(cached[0], sources)
        ):
            return cached[1:]
        if self.bits != 8 or self.mode != "affine" or self.weight.dtype != mx.uint32:
            raise ValueError(
                "W8A8 out_proj requires UINT32-packed 8-bit affine weights"
            )
        _, packed_k = self.weight.shape
        if packed_k * 4 > 131040 or packed_k * 4 % 32:
            raise ValueError("W8A8 out_proj K must be a multiple of 32 in [32, 131040]")
        scales = self.scales.astype(mx.float32)
        biases = self.biases.astype(mx.float32)
        if not bool(
            mx.all(mx.isfinite(scales) & (scales > 0))
            & mx.all(scales == scales[:, :1])
            & mx.all(biases == -128 * scales)
        ):
            raise ValueError(
                "W8A8 out_proj requires repeated per-channel scales and "
                "affine biases=-128*scales; groupwise/asymmetric weights "
                "cannot use this per-channel kernel"
            )
        weight = (self.weight ^ mx.array(0x80808080, dtype=mx.uint32)).view(mx.int8)
        channel_scales = mx.contiguous(scales[:, 0])
        mx.eval(weight, channel_scales)
        object.__setattr__(self, "_w8a8_cache", (sources, weight, channel_scales))
        return weight, channel_scales

    def __call__(self, x):
        weight, scales = self._prepare_w8a8()
        return self._operation(x, weight, scales, self.get("bias"))


class _W8A8OutputProjection(nn.Linear):
    def to_quantized(
        self,
        group_size=None,
        bits=None,
        mode="affine",
        quantize_input=False,
        input_activations=None,
    ):
        if input_activations is not None:
            _validate_w8a8_activations(input_activations)
        if bits != 8 or mode != "affine":
            raise ValueError("W8A8 out_proj requires 8-bit affine packed weights")
        if _w8a8_operation() is None:
            return nn.QuantizedLinear.from_linear(self, group_size, bits, mode)
        return _W8A8QuantizedLinear.from_linear(self, group_size, bits, mode)

    def __call__(self, x):
        raise RuntimeError(
            "Load the configured 8-bit out_proj weights before inference"
        )


def _is_affine_quantized_linear(layer):
    # Use the module call for biases, adapters and other quantization modes.
    return (
        type(layer) is nn.QuantizedLinear
        and layer.mode == "affine"
        and layer.biases is not None
        and "bias" not in layer
    )


def _quantized_linear_args(layer):
    return (
        layer.weight,
        layer.scales,
        layer.biases,
        layer.group_size,
        layer.bits,
    )


def _quantized_linear(x, weight, scales, biases, group_size, bits):
    return mx.quantized_matmul(
        x,
        weight,
        scales=scales,
        biases=biases,
        transpose=True,
        group_size=group_size,
        bits=bits,
        mode="affine",
    )


@cache
def _make_fused_qkv(q_size, kv_size, n_heads, n_kv_heads, head_dim):
    @mx.compile
    def fused_qkv(x, projection_params):
        qkv = _quantized_linear(x, *projection_params)
        queries, keys, values = mx.split(qkv, [q_size, q_size + kv_size], axis=-1)
        batch_size, sequence_length, _ = x.shape
        queries = mx.contiguous(
            queries.reshape(batch_size, sequence_length, n_heads, head_dim).transpose(
                0, 2, 1, 3
            )
        )
        keys = mx.contiguous(
            keys.reshape(batch_size, sequence_length, n_kv_heads, head_dim).transpose(
                0, 2, 1, 3
            )
        )
        values = mx.contiguous(
            values.reshape(batch_size, sequence_length, n_kv_heads, head_dim).transpose(
                0, 2, 1, 3
            )
        )
        return queries, keys, values

    return fused_qkv


@mx.compile
def _fused_attention_output(x, output, gate_weight, out_params):
    output = mx.contiguous(output.transpose(0, 2, 1, 3))
    gate = mx.sigmoid((x @ gate_weight.T).astype(mx.float32)).astype(output.dtype)
    output = output * gate[..., None]
    return _quantized_linear(
        output.reshape(x.shape[0], x.shape[1], -1),
        *out_params,
    )


@mx.compile
def _fused_ffn(
    x, attention_output, norm_weight, gate_params, up_params, down_params, eps
):
    hidden_states = x + attention_output
    normed = mx.fast.rms_norm(hidden_states, norm_weight, eps)
    gate = _quantized_linear(normed, *gate_params)
    up = _quantized_linear(normed, *up_params)
    mlp_output = _quantized_linear(nn.gelu(gate) * up, *down_params)
    return hidden_states + mlp_output


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5

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
        output_projection = (
            _W8A8OutputProjection if _out_proj_uses_w8a8(args, layer_idx) else nn.Linear
        )
        self.out_proj = output_projection(
            q_size,
            args.hidden_size,
            bias=args.attention_bias,
        )
        self.rope = _make_rope(args, layer_type)

    def _attention_output(self, x: mx.array, output: mx.array) -> mx.array:
        if (
            mx.metal.is_available()
            and type(self.g_proj) is nn.Linear
            and "bias" not in self.g_proj
            and _is_affine_quantized_linear(self.out_proj)
        ):
            return _fused_attention_output(
                x,
                output,
                self.g_proj.weight,
                _quantized_linear_args(self.out_proj),
            )
        output = mx.contiguous(output.transpose(0, 2, 1, 3))
        gate = mx.sigmoid(self.g_proj(x).astype(mx.float32)).astype(output.dtype)
        output = output * gate[..., None]
        return self.out_proj(output.reshape(x.shape[0], x.shape[1], -1))

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: Any | None = None,
    ) -> mx.array:
        batch_size, sequence_length, _ = x.shape
        # The cache offset is the absolute token position, so RoPE stays
        # correct even after a rotating cache wraps its bounded storage.
        offset = cache.offset if cache is not None else 0
        quantized_qkv = mx.metal.is_available() and _is_affine_quantized_linear(
            self.q_k_v_proj
        )
        native_qkv = _native_qkv_rope
        use_native_qkv = (
            quantized_qkv
            and callable(native_qkv)
            and batch_size == 1
            and sequence_length == 1
            and self.q_k_v_proj.bits == 4
            and self.q_k_v_proj.group_size in (32, 64, 128)
            and type(self.rope) is nn.RoPE
            and not self.rope.traditional
            and self.rope.scale == 1.0
            and isinstance(offset, int)
        )
        if use_native_qkv:
            queries, keys, values = native_qkv(
                x,
                self.q_k_v_proj.weight,
                self.q_k_v_proj.scales,
                self.q_k_v_proj.biases,
                self.q_size,
                self.kv_size,
                self.n_heads,
                self.n_kv_heads,
                self.head_dim,
                self.rope.dims,
                self.rope.base,
                offset,
                self.q_k_v_proj.group_size,
                self.q_k_v_proj.bits,
            )
        elif quantized_qkv:
            fused_qkv = _make_fused_qkv(
                self.q_size,
                self.kv_size,
                self.n_heads,
                self.n_kv_heads,
                self.head_dim,
            )
            queries, keys, values = fused_qkv(
                x, _quantized_linear_args(self.q_k_v_proj)
            )
        else:
            qkv = self.q_k_v_proj(x)
            queries, keys, values = mx.split(
                qkv,
                [self.q_size, self.q_size + self.kv_size],
                axis=-1,
            )
            queries = queries.reshape(
                batch_size, sequence_length, self.n_heads, self.head_dim
            ).transpose(0, 2, 1, 3)
            keys = keys.reshape(
                batch_size, sequence_length, self.n_kv_heads, self.head_dim
            ).transpose(0, 2, 1, 3)
            values = values.reshape(
                batch_size, sequence_length, self.n_kv_heads, self.head_dim
            ).transpose(0, 2, 1, 3)
            queries = mx.contiguous(queries)
            keys = mx.contiguous(keys)
            values = mx.contiguous(values)

        if not use_native_qkv:
            queries = self.rope(queries, offset=offset)
            keys = self.rope(keys, offset=offset)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=cache,
            scale=self.scale,
            mask=mask,
        )
        return self._attention_output(x, output)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
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
        return self.down_proj(nn.gelu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(args, layer_idx)
        self.mlp = MLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def _ffn(self, x: mx.array, attention_output: mx.array) -> mx.array:
        layers = (
            (self.mlp.gate_proj, self.mlp.up_proj, self.mlp.down_proj)
            if type(self.mlp) is MLP
            else ()
        )
        if (
            mx.metal.is_available()
            and type(self.mlp) is MLP
            and type(self.post_attention_layernorm) is nn.RMSNorm
            and all(_is_affine_quantized_linear(layer) for layer in layers)
        ):
            gate, up, down = layers
            return _fused_ffn(
                x,
                attention_output,
                self.post_attention_layernorm.weight,
                _quantized_linear_args(gate),
                _quantized_linear_args(up),
                _quantized_linear_args(down),
                self.post_attention_layernorm.eps,
            )
        hidden_states = x + attention_output
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))

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
        return self._ffn(x, attention_output)


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

        self._first_full = next(
            (
                idx
                for idx, layer_type in enumerate(args.layer_types)
                if layer_type == "full_attention"
            ),
            None,
        )
        self._first_sliding = next(
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
        if self._first_full is not None:
            full_mask = create_attention_mask(hidden_states, cache[self._first_full])
        sliding_mask = None
        if self._first_sliding is not None:
            sliding_mask = create_attention_mask(
                hidden_states,
                cache[self._first_sliding],
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
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

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
        super().load_weights(file_or_weights, strict=strict)
        for layer in self.layers:
            projection = layer.self_attn.out_proj
            if isinstance(projection, _W8A8QuantizedLinear):
                projection._prepare_w8a8()
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
