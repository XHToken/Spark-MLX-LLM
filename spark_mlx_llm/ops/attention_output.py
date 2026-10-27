"""Gated attention-output dispatch and implementations.

No independent backend flag: the output projection's type and input metadata
select the module, compiled affine or compiled per-channel W8A16 path.
"""

import mlx.core as mx
from mlx import nn

from .linear import AffineParameters, W8A16PerChannelLinear

__all__ = ["dispatch_attention_output"]


def dispatch_attention_output(x, output, gate_projection, output_projection):
    projection = output_projection
    compile_output = (
        mx.metal.is_available()
        and type(gate_projection) is nn.Linear
        and "bias" not in gate_projection
    )
    if compile_output and type(projection) is W8A16PerChannelLinear:
        if W8A16PerChannelLinear.supports_rows(x.shape[0] * x.shape[1], output.dtype):
            return _compiled_w8a16_output(
                x, output, gate_projection.weight, projection.weight, projection.scales
            )
        projection = projection.fallback
    if compile_output and AffineParameters.accepts(projection):
        return _compiled_affine_output(
            x,
            output,
            gate_projection.weight,
            AffineParameters.from_module(projection),
        )
    return _module_output(x, output, gate_projection, output_projection)


@mx.compile
def _compiled_affine_output(x, output, gate_weight, out_params):
    output = mx.contiguous(output.transpose(0, 2, 1, 3))
    gate = mx.sigmoid((x @ gate_weight.T).astype(mx.float32)).astype(output.dtype)
    output = output * gate[..., None]
    return AffineParameters.native(
        output.reshape(x.shape[0], x.shape[1], -1),
        *out_params,
    )


@mx.compile
def _compiled_w8a16_output(x, output, gate_weight, weight, scales):
    """Preserve the experimental graph boundary and explicit tensor inputs."""
    output = mx.contiguous(output.transpose(0, 2, 1, 3))
    gate = mx.sigmoid((x @ gate_weight.T).astype(mx.float32)).astype(output.dtype)
    output = output * gate[..., None]
    return W8A16PerChannelLinear.matmul(
        output.reshape(x.shape[0], x.shape[1], -1), weight, scales
    )


def _module_output(x, output, gate_projection, output_projection):
    output = mx.contiguous(output.transpose(0, 2, 1, 3))
    gate = mx.sigmoid(gate_projection(x).astype(mx.float32)).astype(output.dtype)
    output = output * gate[..., None]
    return output_projection(output.reshape(x.shape[0], x.shape[1], -1))
