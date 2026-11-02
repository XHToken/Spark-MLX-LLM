"""Opt-in derived W8 proposal backend; checkpoint and target stay untouched.

This changes the draft's numeric proposal distribution. Acceptance must always
use the proposal probabilities actually returned by that new draft. Original
module objects are retained for restoration; memory reduction is not claimed.
"""

import math

import mlx.core as mx
from mlx import nn


class _DenseReadout(nn.Module):
    def __init__(self, weight, bias=None):
        super().__init__()
        self.weight = weight
        if bias is not None:
            self.bias = bias

    def __call__(self, x):
        value = x @ self.weight.T
        return value + self.bias if "bias" in self else value


class DerivedDraftPerChannelW8Linear(nn.Module):
    """Derived symmetric INT8 proposal weights with FP32 per-channel scales.

    Eligible GPU M2..8 calls use the explicit public W8 multirow candidate;
    one-row calls use the existing W8A16 primitive. Unsupported shapes/devices
    retain the supplied original dense BF16 projection. This is a changed,
    shape-dependent proposal backend and is deliberately opt-in.
    """

    def __init__(self, weight, *, bias=None, fallback=None, variant="nax"):
        super().__init__()
        if (
            not isinstance(weight, mx.array)
            or weight.ndim != 2
            or weight.dtype != mx.bfloat16
            or min(weight.shape) <= 0
        ):
            raise ValueError(
                "Derived per-channel W8 requires a positive BF16 weight matrix"
            )
        if variant not in ("nax", "matrix", "wide"):
            raise ValueError("Unknown per-channel W8 multirow variant")
        scores = weight.astype(mx.float32)
        scale = mx.max(mx.abs(scores), axis=-1)
        self.scales = mx.where(scale > 0, scale / 127, 1)
        self.weight = mx.clip(
            mx.round(scores / self.scales[:, None]), -127, 127
        ).astype(mx.int8)
        if bias is not None:
            self.bias = bias
        object.__setattr__(
            self,
            "_fallback",
            fallback if fallback is not None else _DenseReadout(weight, bias),
        )
        self._variant = variant
        self.freeze()

    def __call__(self, x):
        if (
            mx.default_device() == mx.gpu
            and mx.metal.is_available()
            and x.dtype == mx.bfloat16
            and x.shape[-1] % 32 == 0
        ):
            import spark_mlx_ext as ext

            rows = math.prod(x.shape[:-1])
            if rows == 1:
                value = ext.w8a16_per_channel(x, self.weight, self.scales)
            elif 2 <= rows <= 8 and (
                self._variant != "nax"
                or self.weight.shape[0] % 32 == 0
                and x.shape[-1] % 64 == 0
            ):
                value = ext.w8a16_multirow(
                    x,
                    self.weight,
                    self.scales,
                    output_dtype=mx.bfloat16,
                    variant=self._variant,
                    simdgroups=4,
                )
            else:
                return self._fallback(x)
            return value + self.bias if "bias" in self else value
        return self._fallback(x)


class DerivedDraftW8Linear(nn.QuantizedLinear):
    """Native affine W8 layer made directly from supplied BF16 checkpoint values.

    Calling the base QuantizedLinear constructor would allocate random temporary
    weights and advance RNG state. Initializing Module and assigning all public
    quantization fields avoids both, then uses the official inherited forward.
    """

    def __init__(self, weight, *, bias=None, group_size=32):
        nn.Module.__init__(self)
        if (
            not isinstance(weight, mx.array)
            or weight.ndim != 2
            or weight.dtype != mx.bfloat16
            or min(weight.shape) <= 0
        ):
            raise ValueError("Derived draft W8 requires a BF16 weight matrix")
        if (
            type(group_size) is not int
            or group_size not in (32, 64, 128)
            or weight.shape[1] % group_size
        ):
            raise ValueError("Derived draft W8 requires group_size32/64/128 dividing K")
        self.group_size, self.bits, self.mode = group_size, 8, "affine"
        self.weight, self.scales, self.biases = mx.quantize(
            weight, group_size=group_size, bits=8, mode="affine"
        )
        if bias is not None:
            if (
                not isinstance(bias, mx.array)
                or bias.shape != (weight.shape[0],)
                or bias.dtype != mx.bfloat16
            ):
                raise ValueError("Derived draft W8 bias must be BF16[N]")
            self.bias = bias
        self.freeze()


def _parent(root, path):
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        parent = parent[int(part)] if isinstance(parent, list) else parent[part]
    return parent, parts[-1]


class DerivedDraftW8:
    """Restorable experiment backend built from original BF16 draft weights.

    Only exact dense nn.Linear modules are replaced. Embeddings, norms and the
    target are untouched. Markov and external readout changes are explicit.
    No checkpoint file is written and no initialization consumes random draws.
    """

    def __init__(
        self,
        drafter,
        *,
        group_size=32,
        quantize_body=True,
        include_markov=False,
        readout_weight=None,
        readout_projection=None,
        per_channel=False,
        operation_variant="nax",
    ):
        from .drafter import DSparkDrafter

        if type(drafter) is not DSparkDrafter:
            raise TypeError("Derived draft backend requires exact DSparkDrafter")
        if readout_weight is not None and readout_projection is not None:
            raise ValueError(
                "Choose derived readout_weight or supplied readout_projection"
            )
        if (
            readout_weight is not None or readout_projection is not None
        ) and drafter._readout_mode != "external":
            raise ValueError(
                "Derived external readout requires the external readout mode"
            )
        self._drafter = drafter
        self._replacements = []
        self._source_weights = []
        self._original_readout = drafter._external_lm_head
        self._new_readout = None
        self._restored = False
        self.description = {
            "proposal_backend": "derived_symmetric_per_channel_w8"
            if per_channel
            else "derived_affine_w8",
            "group_size": None if per_channel else group_size,
            "bits": 8,
            "include_markov": include_markov,
            "quantize_body": quantize_body,
            "readout": "unchanged",
            "changed_names": [],
            "checkpoint_modified": False,
            "target_modified": False,
            "operation_variant": operation_variant if per_channel else "native_affine",
            "dense_fallback": "original_BF16_for_unsupported_device_or_shape"
            if per_channel
            else None,
        }
        try:
            for name, module in list(drafter.named_modules()):
                if (
                    type(module) is not nn.Linear
                    or name.startswith("confidence_head.")
                    or name == "lm_head"
                ):
                    continue
                if name.startswith("markov_head.") and not include_markov:
                    continue
                if not name.startswith("markov_head.") and not quantize_body:
                    continue
                if module.weight.dtype != mx.bfloat16:
                    raise ValueError(f"{name}: expected BF16 source weight")
                derived = (
                    DerivedDraftPerChannelW8Linear(
                        module.weight,
                        bias=module.get("bias"),
                        fallback=module,
                        variant=operation_variant,
                    )
                    if per_channel
                    else DerivedDraftW8Linear(
                        module.weight, bias=module.get("bias"), group_size=group_size
                    )
                )
                mx.eval(derived.parameters())
                parent, key = _parent(drafter, name)
                self._replacements.append((parent, key, module, derived))
                self._source_weights.append((module, module.weight, module.get("bias")))
                parent[key] = derived
                self.description["changed_names"].append(name)
            if readout_weight is not None:
                if readout_weight.shape != (
                    drafter.config.draft_vocab_size,
                    drafter.config.hidden_size,
                ):
                    raise ValueError(
                        "Derived draft readout weight differs from vocabulary/hidden geometry"
                    )
                self._new_readout = (
                    DerivedDraftPerChannelW8Linear(
                        readout_weight, variant=operation_variant
                    )
                    if per_channel
                    else DerivedDraftW8Linear(readout_weight, group_size=group_size)
                )
                mx.eval(self._new_readout.parameters())
                self.description["readout"] = (
                    "derived_original_bf16_weight_per_channel_w8"
                    if per_channel
                    else "derived_original_bf16_weight_w8"
                )
            elif readout_projection is not None:
                if not callable(readout_projection):
                    raise TypeError("readout_projection must be callable")
                self._new_readout = readout_projection
                self.description["readout"] = "supplied_projection_changes_proposal"
            if self._new_readout is not None:
                drafter.bind_lm_head(self._new_readout)
        except BaseException:
            for parent, key, original, _ in reversed(self._replacements):
                parent[key] = original
            if drafter._readout_mode == "external":
                object.__setattr__(drafter, "_external_lm_head", self._original_readout)
            raise

    def activate(self):
        """Reinstall precomputed proposal tensors without quantization or RNG work."""
        if not self._restored:
            if any(
                parent[key] is not derived
                for parent, key, _, derived in self._replacements
            ):
                raise RuntimeError("Active derived draft projection changed")
            if (
                self._new_readout is not None
                and self._drafter._external_lm_head is not self._new_readout
            ):
                raise RuntimeError("Active derived draft readout changed")
            return self
        if any(
            module.weight is not weight or module.get("bias") is not bias
            for module, weight, bias in self._source_weights
        ):
            raise RuntimeError(
                "Source draft weights changed; prepare a fresh derived backend"
            )
        if any(
            parent[key] is not original
            for parent, key, original, _ in self._replacements
        ):
            raise RuntimeError(
                "Restore other draft projections before activating this backend"
            )
        if (
            self._new_readout is not None
            and self._drafter._external_lm_head is not self._original_readout
        ):
            raise RuntimeError(
                "Restore original draft readout before activating this backend"
            )
        for parent, key, _, derived in self._replacements:
            parent[key] = derived
        if self._new_readout is not None:
            self._drafter.bind_lm_head(self._new_readout)
        self._restored = False
        return self

    def restore(self):
        if self._restored:
            return
        if self._new_readout is not None:
            if self._drafter._external_lm_head is not self._new_readout:
                raise RuntimeError(
                    "Draft readout changed before derived backend restore"
                )
            object.__setattr__(
                self._drafter, "_external_lm_head", self._original_readout
            )
        for parent, key, original, derived in reversed(self._replacements):
            if parent[key] is not derived:
                raise RuntimeError(
                    "Draft projection changed before derived backend restore"
                )
            parent[key] = original
        self._restored = True

    def __call__(self):
        self.restore()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.restore()


def install_derived_draft_w8(drafter, **kwargs):
    """Prepare W8 proposal tensors outside timing and return a restore handle."""
    return DerivedDraftW8(drafter, **kwargs)
