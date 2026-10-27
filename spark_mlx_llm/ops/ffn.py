"""Residual/RMSNorm/FFN dispatch and implementations.

The MLP's LinearConfig controls W4 down-projection. Compilation is automatic
for supported standard modules; custom modules retain their own calls.
"""

from functools import cache

import mlx.core as mx
from mlx import nn

from .linear import AffineParameters

__all__ = ["dispatch_ffn"]


def dispatch_ffn(x, attention_output, norm, mlp, *, mlp_type):
    """Select one residual/norm/FFN graph while preserving custom module calls."""
    if (
        type(mlp) is not mlp_type
        or type(norm) is not nn.RMSNorm
        or not mx.metal.is_available()
    ):
        return _module_ffn(x, attention_output, norm, mlp)
    gate, up, down = mlp.gate_proj, mlp.up_proj, mlp.down_proj
    if not all(AffineParameters.accepts(layer) for layer in (gate, up, down)):
        return _module_ffn(x, attention_output, norm, mlp)

    use_splitk = False
    if (
        mlp.linear_config.w4_qmv_backend == "splitk"
        and x.shape[:-1] == (1, 1)
        and x.dtype == mx.bfloat16
        and down.weight.shape == (2560, 1280)
    ):
        # Predict metadata without building a graph. Mixed norm/projection
        # dtypes can promote BF16 residuals to FP32 before down-projection.
        down_dtype = mx.result_type(
            x,
            attention_output,
            norm.weight,
            gate.scales,
            gate.biases,
            up.scales,
            up.biases,
        )
        use_splitk = AffineParameters.supports_splitk(
            (*x.shape[:-1], up.weight.shape[0]),
            down_dtype,
            *AffineParameters.from_module(down),
        )
    compiled_ffn = _make_compiled_ffn(
        AffineParameters.splitk if use_splitk else AffineParameters.native
    )
    return compiled_ffn(
        x,
        attention_output,
        norm.weight,
        AffineParameters.from_module(gate),
        AffineParameters.from_module(up),
        AffineParameters.from_module(down),
        norm.eps,
    )


@cache
def _make_compiled_ffn(down_project=AffineParameters.native):
    """Cache a graph with the down-projection selected by dispatch_ffn.

    Gate/up always use MLX affine arithmetic. Passing the selected callable
    keeps native and split-K graphs separate without capturing model weights.
    """

    @mx.compile
    def compiled_ffn(
        x, attention_output, norm_weight, gate_params, up_params, down_params, eps
    ):
        hidden_states = x + attention_output
        normed = mx.fast.rms_norm(hidden_states, norm_weight, eps)
        gate = AffineParameters.native(normed, *gate_params)
        up = AffineParameters.native(normed, *up_params)
        mlp_output = down_project(nn.gelu(gate) * up, *down_params)
        return hidden_states + mlp_output

    return compiled_ffn


def _module_ffn(x, attention_output, norm, mlp):
    """Preserve module calls for biases, subclasses and adapters."""
    hidden_states = x + attention_output
    return hidden_states + mlp(norm(hidden_states))
