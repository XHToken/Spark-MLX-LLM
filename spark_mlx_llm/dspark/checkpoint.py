"""Read Safetensors metadata without loading tensor payloads."""

import json
import math
import struct
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .config import DSparkConfig

_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "U16": 2,
    "I16": 2,
    "U32": 4,
    "I32": 4,
    "U64": 8,
    "I64": 8,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
    "F16": 2,
    "BF16": 2,
    "F32": 4,
    "F64": 8,
}
DRAFT_PREFIX = "draft_model."


@dataclass(frozen=True)
class TensorInfo:
    """Tensor metadata; data offsets are relative to the shard's payload start."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    shard: Path
    data_offsets: tuple[int, int]

    @property
    def num_parameters(self) -> int:
        return math.prod(self.shape)

    @property
    def nbytes(self) -> int:
        return self.data_offsets[1] - self.data_offsets[0]


@dataclass(frozen=True)
class CheckpointInfo:
    path: Path
    tensors: tuple[TensorInfo, ...]

    @property
    def target_tensors(self) -> dict[str, TensorInfo]:
        return {
            tensor.name: tensor
            for tensor in self.tensors
            if tensor.name.startswith(("model.", "lm_head."))
        }

    @property
    def draft_tensors(self) -> dict[str, TensorInfo]:
        return {
            tensor.name.removeprefix(DRAFT_PREFIX): tensor
            for tensor in self.tensors
            if tensor.name.startswith(DRAFT_PREFIX)
        }

    def summary(self) -> dict:
        target, draft = self.target_tensors, self.draft_tensors
        known = set(target) | {DRAFT_PREFIX + name for name in draft}
        return {
            "path": str(self.path),
            "num_shards": len({tensor.shard for tensor in self.tensors}),
            "num_tensors": len(self.tensors),
            "num_parameters": sum(t.num_parameters for t in self.tensors),
            "weight_bytes": sum(t.nbytes for t in self.tensors),
            "dtypes": dict(Counter(t.dtype for t in self.tensors)),
            "target_parameters": sum(t.num_parameters for t in target.values()),
            "draft_parameters": sum(t.num_parameters for t in draft.values()),
            "draft_has_lm_head": "lm_head.weight" in draft,
            "unclassified_tensors": sorted(
                tensor.name for tensor in self.tensors if tensor.name not in known
            ),
        }


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def read_safetensors_header(
    path: str | Path, *, max_header_bytes: int = 16 * 1024 * 1024
) -> tuple[TensorInfo, ...]:
    path = Path(path)
    file_size = path.stat().st_size
    with path.open("rb") as stream:
        size_bytes = stream.read(8)
        if len(size_bytes) != 8:
            raise ValueError(f"{path}: missing Safetensors header length")
        header_size = struct.unpack("<Q", size_bytes)[0]
        if not 0 < header_size <= min(max_header_bytes, file_size - 8):
            raise ValueError(f"{path}: invalid Safetensors header length {header_size}")
        header = json.loads(stream.read(header_size), object_pairs_hook=_unique_object)
    if not isinstance(header, dict):
        raise TypeError(f"{path}: Safetensors header must be a JSON object")
    payload_size = file_size - 8 - header_size
    tensors = []
    for name, info in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(info, dict):
            raise TypeError(f"{path}: invalid metadata for tensor {name!r}")
        shape, dtype, offsets = (
            info.get("shape"),
            info.get("dtype"),
            info.get("data_offsets"),
        )
        if not isinstance(shape, list) or any(
            type(dim) is not int or dim < 0 for dim in shape
        ):
            raise ValueError(f"{path}: invalid shape for tensor {name!r}")
        if not isinstance(dtype, str) or dtype not in _DTYPE_BYTES:
            raise ValueError(
                f"{path}: unsupported dtype for tensor {name!r}: {dtype!r}"
            )
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(offset) is not int for offset in offsets)
            or not 0 <= offsets[0] <= offsets[1] <= payload_size
        ):
            raise ValueError(f"{path}: invalid data offsets for tensor {name!r}")
        tensor = TensorInfo(name, tuple(shape), dtype, path, tuple(offsets))
        if tensor.nbytes != tensor.num_parameters * _DTYPE_BYTES[dtype]:
            raise ValueError(f"{path}: shape and byte count differ for tensor {name!r}")
        tensors.append(tensor)
    end = 0
    for tensor in sorted(tensors, key=lambda t: t.data_offsets):
        if tensor.data_offsets[0] != end:
            raise ValueError(f"{path}: tensor data has gaps or overlaps")
        end = tensor.data_offsets[1]
    if end != payload_size:
        raise ValueError(f"{path}: tensor data does not cover the payload")
    return tuple(tensors)


def inspect_checkpoint(path: str | Path) -> CheckpointInfo:
    """Inspect indexed shards, or model*.safetensors when no index is present."""
    path = Path(path).resolve()
    index_path = path / "model.safetensors.index.json"
    index = None
    if index_path.exists():
        with index_path.open(encoding="utf-8") as stream:
            index = json.load(stream, object_pairs_hook=_unique_object)
        if not isinstance(index, dict):
            raise ValueError("The checkpoint index must be a JSON object")
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("The checkpoint index needs a nonempty weight_map")
        if any(not isinstance(shard, str) for shard in weight_map.values()):
            raise ValueError("weight_map shard paths must be strings")
        shards = [(path / name).resolve() for name in sorted(set(weight_map.values()))]
        if any(not shard.is_relative_to(path) for shard in shards):
            raise ValueError(
                "Index shard paths must stay inside the checkpoint directory"
            )
    else:
        shards = sorted(path.glob("model*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No Safetensors model shards found in {path}")
    tensors = {}
    actual_map = {}
    for shard in shards:
        for tensor in read_safetensors_header(shard):
            if tensor.name in tensors:
                raise ValueError(f"Tensor occurs in multiple shards: {tensor.name}")
            tensors[tensor.name] = tensor
            actual_map[tensor.name] = shard.relative_to(path).as_posix()
    result = CheckpointInfo(path, tuple(tensors.values()))
    if index is not None:
        if actual_map != weight_map:
            raise ValueError("Checkpoint weight_map does not match the shard headers")
        metadata = index.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("Checkpoint index metadata must be a JSON object")
        for name, value in (
            ("total_parameters", sum(t.num_parameters for t in result.tensors)),
            ("total_size", sum(t.nbytes for t in result.tensors)),
        ):
            if name in metadata and metadata[name] != value:
                raise ValueError(
                    f"Checkpoint metadata {name} does not match the tensors"
                )
    return result


def validate_draft_shapes(config: DSparkConfig, checkpoint: CheckpointInfo) -> None:
    """Validate Spark draft tensor names and shapes; leave readout binding unresolved."""
    h, intermediate = config.hidden_size, config.intermediate_size
    shapes = {
        "embed_tokens.weight": (config.vocab_size, h),
        "fc.weight": (h, config.context_input_size),
        "hidden_norm.weight": (h,),
        "norm.weight": (h,),
    }
    if config.fc_norm:
        for i in range(len(config.target_layer_ids)):
            shapes[f"fc_norm.{i}.weight"] = (h,)
    for i in range(config.num_draft_layers):
        prefix = f"layers.{i}."
        shapes.update(
            {
                prefix + "input_layernorm.weight": (h,),
                prefix + "post_attention_layernorm.weight": (h,),
                prefix + "self_attn.q_k_v_proj.weight": (
                    config.q_size + 2 * config.kv_size,
                    h,
                ),
                prefix + "self_attn.g_proj.weight": (config.num_attention_heads, h),
                prefix + "self_attn.out_proj.weight": (h, config.q_size),
                prefix + "mlp.gate_proj.weight": (intermediate, h),
                prefix + "mlp.up_proj.weight": (intermediate, h),
                prefix + "mlp.down_proj.weight": (h, intermediate),
            }
        )
        if config.attention_bias:
            shapes[prefix + "self_attn.q_k_v_proj.bias"] = (
                config.q_size + 2 * config.kv_size,
            )
            shapes[prefix + "self_attn.g_proj.bias"] = (config.num_attention_heads,)
            shapes[prefix + "self_attn.out_proj.bias"] = (h,)
        if config.mlp_bias:
            for name, size in (
                ("gate_proj", intermediate),
                ("up_proj", intermediate),
                ("down_proj", h),
            ):
                shapes[prefix + f"mlp.{name}.bias"] = (size,)
    if config.markov_rank:
        shapes["markov_head.markov_w1.weight"] = (config.vocab_size, config.markov_rank)
        shapes["markov_head.markov_w2.weight"] = (
            config.draft_vocab_size,
            config.markov_rank,
        )
    if config.enable_confidence_head:
        shapes["confidence_head.proj.weight"] = (1, config.confidence_input_size)
        shapes["confidence_head.proj.bias"] = (1,)
    draft = checkpoint.draft_tensors
    if "lm_head.weight" in draft:
        shapes["lm_head.weight"] = (config.draft_vocab_size, h)
    missing = sorted(shapes.keys() - draft.keys())
    unexpected = sorted(draft.keys() - shapes.keys())
    wrong_shapes = {
        name: {"expected": shape, "actual": draft[name].shape}
        for name, shape in shapes.items()
        if name in draft and draft[name].shape != shape
    }
    if missing or unexpected or wrong_shapes:
        raise ValueError(
            f"Draft tensor mismatch: missing={missing}, unexpected={unexpected}, "
            f"wrong_shapes={wrong_shapes}"
        )
