"""Public DSpark metadata, sampling, acceptance, and generation interfaces."""

from importlib import import_module

from .acceptance import (
    AcceptanceResult,
    greedy_accept,
    sampled_accept,
)
from .checkpoint import (
    CheckpointInfo,
    TensorInfo,
    inspect_checkpoint,
    read_safetensors_header,
    validate_draft_shapes,
)
from .config import DraftLayerSpec, DSparkConfig, spark_layer_specs
from .draft_policy import prefix_survival_probabilities, verification_length
from .metrics import DraftLengthStats, GenerationResult

__all__ = [
    "AcceptanceResult",
    "CheckpointInfo",
    "DSparkConfig",
    "DSparkDrafter",
    "DraftLayerSpec",
    "DraftLengthStats",
    "DraftPrediction",
    "GenerationResult",
    "SamplingOptions",
    "TensorInfo",
    "TokenEvent",
    "generate",
    "greedy_accept",
    "inspect_checkpoint",
    "load_pair",
    "prefix_survival_probabilities",
    "read_safetensors_header",
    "sample_draft",
    "sampled_accept",
    "spark_layer_specs",
    "stream_generate",
    "validate_draft_shapes",
    "verification_length",
]

_RUNTIME_EXPORTS = {
    "DSparkDrafter": "drafter",
    "load_pair": "loader",
    "generate": "generation",
    "stream_generate": "generation",
    "TokenEvent": "generation",
    "SamplingOptions": "sampling",
    "DraftPrediction": "sampling",
    "sample_draft": "sampling",
}


def __getattr__(name):
    if name not in _RUNTIME_EXPORTS:
        raise AttributeError(name)
    module = import_module(f"{__name__}.{_RUNTIME_EXPORTS[name]}")
    return getattr(module, name)
