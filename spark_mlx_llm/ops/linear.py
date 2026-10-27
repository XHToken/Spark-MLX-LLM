"""Linear configuration, dispatch, implementations and checkpoint adapters.

``dispatch_linear`` is the model's linear execution entry. ``AffineParameters``
shares pure kernels with compiled composite operators without capturing
weights. ``QuantizedProjections`` owns model checkpoint adaptation.
"""

from dataclasses import dataclass, field
from math import prod
from typing import NamedTuple

import mlx.core as mx
from mlx import nn

from .base import OperatorConfig

__all__ = [
    "AffineParameters",
    "LinearConfig",
    "QuantizedProjections",
    "W8A8Projection",
    "W8A8QuantizedLinear",
    "W8A16PerChannelLinear",
    "dispatch_linear",
]

try:
    from spark_mlx_ext import w4a16_qmv
except ImportError:
    w4a16_qmv = None

try:
    from spark_mlx_ext import w8a16_per_channel
except ImportError:
    w8a16_per_channel = None

# (N, K): measured QKV and MLP-down shapes for the 4B model.
_SPLITK_SHAPES = frozenset(((6144, 2560), (2560, 10240)))


@dataclass(kw_only=True)
class LinearConfig(OperatorConfig):
    """Select implementations for existing W8/W4 layers, never weight bit widths.

    W8 selection applies to attention out_proj and an untied W8 lm_head.
    W4 QKV/FFN layers retain their separate W4A16 arithmetic and backend.
    Unavailable extension implementations fall back to MLX affine operators.
    W8A16 also retains MLX layers unsupported by its per-channel kernel.
    """

    w8_qmv_backend: str = field(
        default="w8a16", metadata={"choices": ("auto", "native", "w8a16", "w8a8")}
    )
    w4_qmv_backend: str = field(
        default="native", metadata={"choices": ("native", "splitk")}
    )

    @classmethod
    def validate_values(cls, values):
        if "qmv_backend" in values:
            raise ValueError(
                "qmv_backend was renamed to w8_qmv_backend; update the configuration"
            )
        super().validate_values(values)


def dispatch_linear(x, projection, *, config: LinearConfig | None = None):
    """Select split-K for eligible W4 layers; otherwise call the projection.

    W8 policy is installed by QuantizedProjections at checkpoint load time.
    Prepared W8 modules apply their own input/device fallback when called.
    """
    config = config if config is not None else LinearConfig()
    if config.w4_qmv_backend == "splitk" and AffineParameters.accepts(projection):
        params = AffineParameters.from_module(projection)
        return params.select(x.shape, x.dtype, config)(x, *params)
    return projection(x)


class AffineParameters(NamedTuple):
    """Current affine tensors and metadata, passed explicitly into compiled graphs.

    ``select`` returns a pure callable with no captured weights. Composite
    operators reuse this contract instead of reaching into private helpers.
    """

    weight: mx.array
    scales: mx.array
    biases: mx.array | None
    group_size: int
    bits: int

    @classmethod
    def from_module(cls, layer):
        return cls(
            layer.weight, layer.scales, layer.biases, layer.group_size, layer.bits
        )

    def select(self, shape, dtype, config: LinearConfig):
        if config.w4_qmv_backend == "splitk" and self.supports_splitk(
            shape, dtype, *self
        ):
            return self.splitk
        return self.native

    @staticmethod
    def accepts(layer):
        # Use the module call for biases, adapters and other quantization modes.
        return (
            type(layer) is nn.QuantizedLinear
            and layer.mode == "affine"
            and layer.biases is not None
            and "bias" not in layer
        )

    @staticmethod
    def supports_splitk(shape, dtype, weight, scales, biases, group_size, bits):
        """Select only measured BF16/G32, one-row GPU geometries, without eval."""
        return (
            callable(w4a16_qmv)
            and bits == 4
            and group_size == 32
            and len(shape) >= 1
            and prod(shape[:-1]) == 1
            and weight.ndim == 2
            and (weight.shape[0], shape[-1]) in _SPLITK_SHAPES
            and weight.shape[1] * 8 == shape[-1]
            and scales.shape == (weight.shape[0], shape[-1] // 32)
            and biases is not None
            and biases.shape == scales.shape
            and dtype == mx.bfloat16
            and scales.dtype == mx.bfloat16
            and biases.dtype == mx.bfloat16
            and weight.dtype == mx.uint32
            and mx.default_device() == mx.gpu
            and mx.metal.is_available()
        )

    @staticmethod
    def native(x, weight, scales, biases, group_size, bits):
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

    @staticmethod
    def splitk(x, weight, scales, biases, group_size, bits):
        """Affine-compatible signature; select() must first choose supported W4/G32.

        The extension fixes group size and bits, so only tensors reach its API.
        """
        if not callable(w4a16_qmv):
            raise RuntimeError(  # noqa: TRY004 -- missing optional extension
                "w4_qmv_backend='splitk' requires spark-mlx-ext with the "
                "w4a16_qmv API; install or update the local extension"
            )
        return w4a16_qmv(x, weight, scales, biases)


def _get_w8a8_operation():
    try:
        from spark_mlx_ext import is_w8a8_available, w8a8_linear
    except ImportError:
        return None
    if not is_w8a8_available():
        return None
    return w8a8_linear


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


def _unpack_affine_w8(layer, name, *, allow_fallback=False):
    """Extract per-channel INT8 parameters, or allow native affine fallback."""
    n, packed_k = layer.weight.shape
    scales = layer.scales.astype(mx.float32)
    if (
        layer.mode != "affine"
        or layer.weight.dtype != mx.uint32
        or layer.biases is None
        or scales.ndim != 2
        or scales.shape != (n, packed_k * 4 // layer.group_size)
        or layer.biases.shape != scales.shape
    ):
        raise ValueError(f"{name}: QMV requires UINT32-packed affine W8 parameters")
    if not bool(
        mx.all(mx.isfinite(scales) & (scales > 0))
        & mx.all(scales == scales[:, :1])
        & mx.all(layer.biases.astype(mx.float32) == -128 * scales)
    ):
        if allow_fallback:
            return None
        raise ValueError(
            f"{name}: QMV requires repeated positive per-channel scales and "
            "affine biases=-128*scales; groupwise/asymmetric W8 is unsupported"
        )
    weight = (layer.weight ^ mx.array(0x80808080, dtype=mx.uint32)).view(mx.int8)
    return mx.contiguous(weight), mx.contiguous(scales[:, 0])


class W8A8QuantizedLinear(nn.QuantizedLinear):
    """Keep MLX affine checkpoint parameters; cache native INT8 kernel inputs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._operation = _get_w8a8_operation()
        if self._operation is None:
            raise RuntimeError("Cannot create a native W8A8 layer without W8A8 support")
        # Derived buffers are not checkpoint parameters.
        object.__setattr__(self, "_w8a8_cache", None)

    def prepare_parameters(self):
        """Validate/unpack current weights; reused buffers stay out of checkpoints."""
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
        weight, scales = self.prepare_parameters()
        return self._operation(x, weight, scales, self.get("bias"))


class _AffineOutputProjection(nn.Linear):
    """Consume stored activation metadata when explicitly overriding QMV."""

    def to_quantized(
        self,
        group_size=None,
        bits=None,
        mode="affine",
        quantize_input=False,
        input_activations=None,
    ):
        return nn.QuantizedLinear.from_linear(self, group_size, bits, mode)


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
        if _get_w8a8_operation() is None:
            return nn.QuantizedLinear.from_linear(self, group_size, bits, mode)
        return W8A8QuantizedLinear.from_linear(self, group_size, bits, mode)

    def __call__(self, x):
        raise RuntimeError(
            "Load the configured 8-bit out_proj weights before inference"
        )


class W8A16PerChannelLinear(nn.Module):
    """Single-row GPU projection with an existing affine fallback.

    ``weight`` and ``scales`` must match the corresponding affine layer:
    signed INT8 [N,K] and FP32 [N] or [N,1]. Loading derives these from the
    current checkpoint's packed weights and scales. They are explicit module
    parameters, not a hidden cache; compiled callers must pass both as inputs.
    This wrapper is for inference after checkpoint loading, not serialization
    in the original checkpoint schema. The W8 loading policy installs it without
    changing any checkpoint weight bit widths.
    """

    def __init__(self, fallback, weight, scales):
        super().__init__()
        if (
            type(fallback) is not nn.QuantizedLinear
            or fallback.mode != "affine"
            or fallback.bits != 8
            or fallback.biases is None
            or "bias" in fallback
        ):
            raise ValueError(
                "W8A16 fallback must be a bias-free affine W8 QuantizedLinear"
            )
        if (
            weight.ndim != 2
            or weight.dtype != mx.int8
            or weight.shape != (fallback.weight.shape[0], fallback.weight.shape[1] * 4)
            or scales.dtype != mx.float32
            or scales.shape not in ((weight.shape[0],), (weight.shape[0], 1))
        ):
            raise ValueError(
                "W8A16 needs matching signed INT8 [N,K] and FP32 [N] scales"
            )
        if w8a16_per_channel is None:
            raise RuntimeError("W8A16 requires spark-mlx-ext.w8a16_per_channel")
        self.fallback = fallback
        self.weight = mx.contiguous(weight)
        self.scales = mx.contiguous(scales.reshape(-1))
        self.freeze()

    def __call__(self, x):
        if (
            x.ndim >= 1
            and x.shape[-1] > 0
            and self.supports_rows(x.size // x.shape[-1], x.dtype)
        ):
            return self.matmul(x, self.weight, self.scales)
        return self.fallback(x)

    @staticmethod
    def supports_rows(num_rows, dtype):
        return (
            num_rows == 1
            and dtype in (mx.bfloat16, mx.float16)
            and mx.default_device() == mx.gpu
            and mx.metal.is_available()
        )

    @staticmethod
    def matmul(x, weight, scales):
        if w8a16_per_channel is None:
            raise RuntimeError("W8A16 requires spark-mlx-ext.w8a16_per_channel")
        return w8a16_per_channel(x, weight, scales)


class W8A8Projection(nn.Module):
    """Experimental W8A8 arithmetic with an affine CPU fallback.

    GPU prefill and decode both quantize activations per token, as in the
    experiment. The public extension API selects v9 small projections and
    the single-row head specialization; other sizes use its general kernel.
    """

    def __init__(self, fallback, weight, scales):
        super().__init__()
        try:
            from spark_mlx_ext import W8A8Linear, is_w8a8_available
        except ImportError as error:
            raise RuntimeError(
                "w8_qmv_backend='w8a8' requires spark-mlx-ext"
            ) from error
        if not is_w8a8_available():
            raise RuntimeError(
                "w8_qmv_backend='w8a8' requires Apple M5+ and macOS 26.2+"
            )
        self.fallback = fallback
        self.projection = W8A8Linear(weight, scales, fallback.get("bias"))
        self.freeze()

    def __call__(self, x):
        if (
            x.dtype in (mx.bfloat16, mx.float16, mx.float32)
            and mx.default_device() == mx.gpu
            and mx.metal.is_available()
        ):
            return self.projection(x)
        return self.fallback(x)


class QuantizedProjections:
    """Checkpoint lifecycle for W8 output projections and the untied LM head."""

    @staticmethod
    def _out_proj_uses_w8a8(args, layer_idx: int) -> bool:
        if args.w8_qmv_backend != "auto":
            return False
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

    @staticmethod
    def output_type(args, layer_idx):
        if args.w8_qmv_backend != "auto":
            return _AffineOutputProjection
        return (
            _W8A8OutputProjection
            if QuantizedProjections._out_proj_uses_w8a8(args, layer_idx)
            else nn.Linear
        )

    @staticmethod
    def head_type(args):
        return _AffineOutputProjection if args.w8_qmv_backend != "auto" else nn.Linear

    @staticmethod
    def _projections(model):
        for index, layer in enumerate(model.layers):
            yield (
                f"model.layers.{index}.self_attn.out_proj",
                layer.self_attn,
                "out_proj",
            )
        if not model.args.tie_word_embeddings:
            yield "lm_head", model, "lm_head"

    @staticmethod
    def unwrap(model):
        for _, parent, attribute in QuantizedProjections._projections(model):
            projection = getattr(parent, attribute)
            if type(projection) in (W8A16PerChannelLinear, W8A8Projection):
                setattr(parent, attribute, projection.fallback)

    @staticmethod
    def _configure(model, backend):
        """Wrap W8 out_proj and untied lm_head; W4 and embeddings stay on MLX.

        'auto' keeps the checkpoint's activation configuration; 'native' forces
        MLX affine routing at model construction. Unavailable optimized backends
        retain the affine projection, including floating-point activations for
        W8A8 fallback. W8A16 also retains native layers outside its per-channel,
        bias-free format; available W8A8 still requires compatible W8 metadata.
        Parameters are derived exclusively from the loaded affine checkpoint;
        both representations remain resident when an adapter is installed.
        """
        LinearConfig(w8_qmv_backend=backend)
        if backend in ("auto", "native"):
            return
        if backend == "w8a16" and not callable(w8a16_per_channel):
            return
        if backend == "w8a8" and _get_w8a8_operation() is None:
            return
        candidates = [
            (name, parent, attribute, getattr(parent, attribute))
            for name, parent, attribute in QuantizedProjections._projections(model)
            if type(getattr(parent, attribute)) is nn.QuantizedLinear
            and getattr(parent, attribute).bits == 8
        ]
        # Unquantized and W4-only checkpoints have no W8 projections to adapt.
        replacements = []
        for name, parent, attribute, layer in candidates:
            if backend == "w8a16" and (layer.mode != "affine" or "bias" in layer):
                continue
            parameters = _unpack_affine_w8(
                layer, name, allow_fallback=backend == "w8a16"
            )
            if parameters is None:
                continue
            weight, scales = parameters
            adapter = W8A16PerChannelLinear if backend == "w8a16" else W8A8Projection
            replacements.append((parent, attribute, adapter(layer, weight, scales)))
        # Validate everything before mutating any module.
        for parent, attribute, projection in replacements:
            setattr(parent, attribute, projection)

    @staticmethod
    def scale_paths(model):
        """FP32 operator metadata that a loader dtype override must preserve."""
        paths = set()
        for name, parent, attribute in QuantizedProjections._projections(model):
            projection = getattr(parent, attribute)
            if type(projection) is W8A16PerChannelLinear:
                paths.add(name + ".scales")
            elif type(projection) is W8A8Projection:
                paths.add(name + ".projection.scales")
        return paths

    @staticmethod
    def prepare(model):
        for layer in model.layers:
            projection = layer.self_attn.out_proj
            if isinstance(projection, W8A8QuantizedLinear):
                projection.prepare_parameters()
        QuantizedProjections._configure(model, model.args.w8_qmv_backend)
