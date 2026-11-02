"""Strict loading of a local merged Spark target and DSpark checkpoint."""

import json
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_map_with_path
from mlx_lm.tokenizer_utils import load as load_tokenizer

from .checkpoint import DRAFT_PREFIX, inspect_checkpoint, validate_draft_shapes
from .drafter import DSparkDrafter
from .target import SparkTarget


@dataclass(frozen=True)
class ModelPair:
    target: SparkTarget
    drafter: DSparkDrafter
    tokenizer: object
    config: dict
    generation_config: dict


def _quantize_structure(model, weights, quantization):
    def predicate(path, module):
        if path + ".scales" not in weights:
            return False
        return quantization.get(path, True)

    nn.quantize(
        model,
        group_size=quantization.get("group_size", 64),
        bits=quantization.get("bits", 4),
        mode=quantization.get("mode", "affine"),
        class_predicate=predicate,
    )


def load_pair(path, *, dtype=None, lazy=False, tokenizer=True, model_config=None):
    from spark_mlx_llm.model import Model, ModelArgs
    from spark_mlx_llm.ops.linear import QuantizedProjections

    path = Path(path).expanduser().resolve()
    raw = json.loads((path / "config.json").read_text())
    target_config = {"w8_qmv_backend": "native", **raw, **(model_config or {})}
    target_model = Model(ModelArgs.from_dict(target_config))
    checkpoint = inspect_checkpoint(path)
    drafter = DSparkDrafter.from_spark_config(
        raw,
        readout_mode="linear"
        if checkpoint.summary()["draft_has_lm_head"]
        else "external",
    )
    validate_draft_shapes(drafter.config, checkpoint)
    if checkpoint.summary()["unclassified_tensors"]:
        raise ValueError("Merged checkpoint contains unclassified tensors")
    target_weights, draft_weights = {}, {}
    for shard in sorted({t.shard for t in checkpoint.tensors}):
        for name, array in mx.load(str(shard)).items():
            if name.startswith(DRAFT_PREFIX):
                draft_weights[name.removeprefix(DRAFT_PREFIX)] = array
            else:
                target_weights[name] = array
    target_weights = target_model.sanitize(target_weights)
    if raw.get("quantization"):
        _quantize_structure(target_model, target_weights, raw["quantization"])
    target_model.load_weights(list(target_weights.items()), strict=True)
    drafter.load_weights(list(draft_weights.items()), strict=True)
    if dtype is not None:
        dtypes = {"bfloat16": mx.bfloat16, "float32": mx.float32}
        if dtype not in dtypes:
            raise ValueError("dtype must be bfloat16 or float32")
        scale_paths = QuantizedProjections.scale_paths(target_model)
        target_model.update(
            tree_map_with_path(
                lambda name, value: (
                    value.astype(dtypes[dtype])
                    if name not in scale_paths
                    and mx.issubdtype(value.dtype, mx.floating)
                    else value
                ),
                target_model.parameters(),
            )
        )
        drafter.set_dtype(dtypes[dtype])
    target = SparkTarget(target_model, drafter.config.target_layer_ids)
    if drafter.lm_head is None:
        drafter.bind_lm_head(target.project)
    if not lazy:
        mx.eval(target_model.parameters(), drafter.parameters())
    tok = (
        load_tokenizer(
            path,
            tokenizer_config_extra={"trust_remote_code": False},
            eos_token_ids=raw.get("eos_token_id"),
        )
        if tokenizer
        else None
    )
    generation_path = path / "generation_config.json"
    generation_config = (
        json.loads(generation_path.read_text()) if generation_path.exists() else {}
    )
    return ModelPair(target, drafter, tok, raw, generation_config)
