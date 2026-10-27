"""Convert local Spark BF16 or compressed-tensors INT4 checkpoints to MLX.

Dimensions come from config.json. By default QKV/MLP use asymmetric INT4/G32,
out_proj and lm_head use symmetric per-channel INT8 with dynamic per-token INT8
activations (w8a8 backend), and gates/embedding/norms remain BF16. Draft tensors
are copied byte-for-byte. Use --quantization none to keep a BF16 target.

Requires NumPy only; conversion is streamed on CPU without loading the model.
"""

from __future__ import annotations

import copy
import ctypes
import hashlib
import json
import math
import os
import re
import shutil
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any, Literal

import numpy as np

__all__ = ["convert_to_mlx"]

SIZES = {"BF16": 2, "F16": 2, "F32": 4, "I8": 1, "I32": 4, "U32": 4, "I64": 8}

ASSETS = (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
    "chat_template.jinja",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def inventory(source, *, include_draft_shards=False):
    index = source / "model.safetensors.index.json"
    mapping = json.loads(index.read_text())["weight_map"] if index.exists() else None
    names = set(mapping.values()) if mapping else {"model.safetensors"}
    if include_draft_shards:
        if mapping is None:
            names = {p.name for p in source.glob("model*.safetensors")}
        else:
            names.update(p.name for p in source.glob("model-draft*.safetensors"))
    require(bool(names), "No Safetensors model shards found")
    require(
        all(Path(n).name == n and n.endswith(".safetensors") for n in names),
        "Invalid shard names",
    )
    require(
        names == {p.name for p in source.glob("*.safetensors")},
        "Missing or unreferenced Safetensors shards",
    )
    tensors = {}
    for name in sorted(names):
        path = source / name
        with path.open("rb") as stream:
            prefix = stream.read(8)
            require(len(prefix) == 8, f"Truncated header: {path}")
            size = struct.unpack("<Q", prefix)[0]
            require(2 <= size <= 64 * 1024 * 1024, f"Invalid header size: {path}")
            header = json.loads(stream.read(size))
        end = 0
        spans = []
        for key, desc in header.items():
            if key == "__metadata__":
                continue
            dtype, shape, offsets = (
                desc[k] for k in ("dtype", "shape", "data_offsets")
            )
            require(key not in tensors, f"Duplicate tensor: {key}")
            require(
                dtype in SIZES
                and shape
                and all(type(n) is int and n > 0 for n in shape),
                f"Invalid dtype/shape: {key}",
            )
            require(
                len(offsets) == 2
                and all(type(n) is int for n in offsets)
                and 0 <= offsets[0] < offsets[1]
                and offsets[1] - offsets[0] == math.prod(shape) * SIZES[dtype],
                f"Invalid offsets: {key}",
            )
            require(
                mapping is None
                or mapping.get(key) == name
                or (
                    include_draft_shards
                    and key.startswith("draft_model.")
                    and name.startswith("model-draft")
                    and key not in mapping
                ),
                f"Shard index mismatch: {key}",
            )
            tensors[key] = {
                "path": path,
                "offset": 8 + size + offsets[0],
                "dtype": dtype,
                "shape": shape,
                "bytes": offsets[1] - offsets[0],
            }
            spans.append(offsets)
        for start, stop in sorted(spans):
            require(start == end, f"Non-contiguous tensor payload: {path}")
            end = stop
        require(
            8 + size + end == path.stat().st_size, f"Truncated/trailing payload: {path}"
        )
    indexed = set(tensors)
    if include_draft_shards and mapping is not None:
        indexed -= {
            k for k in tensors if k.startswith("draft_model.") and k not in mapping
        }
    require(mapping is None or set(mapping) == indexed, "Incomplete shard index")
    return tensors, names | ({index.name} if index.exists() else set())


def read(desc, first=0, stop=None):
    stop = desc["shape"][0] if stop is None else stop
    stride = desc["bytes"] // desc["shape"][0]
    with desc["path"].open("rb") as stream:
        stream.seek(desc["offset"] + first * stride)
        data = stream.read((stop - first) * stride)
    require(len(data) == (stop - first) * stride, "Source changed or was truncated")
    return data


def floats(desc, first, stop):
    raw = read(desc, first, stop)
    if desc["dtype"] == "BF16":
        values = (np.frombuffer(raw, "<u2").astype(np.uint32) << 16).view(np.float32)
    else:
        require(desc["dtype"] in ("F16", "F32"), "Expected floating-point scales")
        values = np.frombuffer(raw, "<f2" if desc["dtype"] == "F16" else "<f4")
    return values.astype(np.float32).reshape(stop - first, *desc["shape"][1:])


def bf16(values):
    words = np.asarray(values, dtype="<f4").view(np.uint32)
    encoded = ((words + 0x7FFF + ((words >> 16) & 1)) >> 16).astype("<u2")
    decoded = (encoded.astype(np.uint32) << 16).view(np.float32)
    require(bool(np.isfinite(decoded).all()), "BF16 coefficient overflow")
    return encoded, decoded


def matches(selector, module):
    return (
        selector == "Linear"
        or selector == module
        or (
            selector.startswith("re:")
            and re.fullmatch(selector[3:], module) is not None
        )
    )


def shapes(config):
    keys = (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "vocab_size",
    )
    require(
        all(type(config.get(k)) is int and config[k] > 0 for k in keys),
        "Invalid Spark dimensions",
    )
    hidden, inter, layers, heads, kv, dim, vocab = (config[k] for k in keys)
    require(heads % kv == 0, "Invalid GQA head counts")
    require(
        not config.get("attention_bias") and not config.get("mlp_bias"),
        "Linear biases are unsupported",
    )
    plain = {"model.embedding.weight": [vocab, hidden], "model.norm.weight": [hidden]}
    linear = {}
    for i in range(layers):
        prefix = f"model.layers.{i}"
        linear.update(
            {
                f"{prefix}.self_attn.q_k_v_proj": [(heads + 2 * kv) * dim, hidden],
                f"{prefix}.self_attn.out_proj": [hidden, heads * dim],
                f"{prefix}.self_attn.g_proj": [heads, hidden],
                f"{prefix}.mlp.gate_proj": [inter, hidden],
                f"{prefix}.mlp.up_proj": [inter, hidden],
                f"{prefix}.mlp.down_proj": [hidden, inter],
            }
        )
        plain.update(
            {
                f"{prefix}.input_layernorm.weight": [hidden],
                f"{prefix}.post_attention_layernorm.weight": [hidden],
            }
        )
    if not config.get("tie_word_embeddings", True):
        linear["lm_head"] = [vocab, hidden]
    return linear, plain


def _plan_compressed(config, tensors):
    require(config.get("model_type") == "spark2_5", "Expected model_type=spark2_5")
    require("quantization" not in config, "Source is already in MLX format")
    quant = config.get("quantization_config", {})
    require(
        quant.get("quant_method") == "compressed-tensors"
        and quant.get("format")
        in ("mixed-precision", "pack-quantized", "int-quantized"),
        "Expected compressed-tensors integer quantization",
    )
    require(
        not quant.get("kv_cache_scheme")
        and not quant.get("sparsity_config")
        and not quant.get("transform_config"),
        "KV quantization/sparsity/transforms unsupported",
    )
    groups = quant.get("config_groups", {})
    require(bool(groups), "Missing config_groups")
    linear, plain = shapes(config)
    modules, output, consumed = {}, {}, set()

    def expect(key, dtypes, shape):
        require(key in tensors, f"Missing tensor: {key}")
        desc = tensors[key]
        require(
            desc["dtype"] in dtypes and desc["shape"] == shape,
            f"Tensor dtype/shape mismatch: {key}",
        )
        consumed.add(key)
        return desc

    for module, shape in linear.items():
        selected = [
            g
            for g in groups.values()
            if any(matches(s, module) for s in g.get("targets", []))
        ]
        if any(matches(s, module) for s in quant.get("ignore", [])):
            selected = []
        require(len(selected) <= 1, f"Conflicting quantization groups: {module}")
        if not selected:
            plain[module + ".weight"] = shape
            continue
        group = selected[0]
        weights = group["weights"]
        bits, strategy = weights.get("num_bits"), weights.get("strategy")
        require(
            weights.get("type") == "int"
            and not weights.get("dynamic")
            and weights.get("actorder") in (None, "static")
            and not weights.get("block_structure"),
            f"Unsupported weights: {module}",
        )
        require(
            group.get("output_activations") is None,
            f"Output activation quantization unsupported: {module}",
        )
        acts = group.get("input_activations")
        if acts is not None:
            expected = {
                "num_bits": 8,
                "type": "int",
                "strategy": "token",
                "symmetric": True,
                "dynamic": True,
            }
            require(
                bits == 8 and all(acts.get(k) == v for k, v in expected.items()),
                f"Only dynamic symmetric per-token INT8 inputs supported: {module}",
            )
        out_dim, in_dim = shape
        if strategy == "group" and bits == 4:
            group_size = weights.get("group_size")
            require(
                group.get("format") in (None, "pack-quantized")
                and group_size in (32, 64, 128)
                and in_dim % group_size == 0,
                f"Expected packed INT4/G32/G64/G128: {module}",
            )
            expect(module + ".weight_packed", {"I32", "U32"}, [out_dim, in_dim // 8])
            scale_shape = [out_dim, in_dim // group_size]
            original = expect(module + ".weight_shape", {"I64"}, [2])
            require(
                list(struct.unpack("<qq", read(original))) == shape,
                f"Declared weight shape mismatch: {module}",
            )
            if module + ".weight_zero_point" in tensors:
                expect(
                    module + ".weight_zero_point",
                    {"I32", "U32"},
                    [(out_dim + 7) // 8, scale_shape[1]],
                )
            else:
                require(
                    weights.get("symmetric") is True, f"Missing zero points: {module}"
                )
            kind = "packed-int4"
        else:
            require(
                bits == 8
                and strategy == "channel"
                and weights.get("symmetric") is True
                and weights.get("group_size") is None
                and group.get("format") in (None, "int-quantized")
                and in_dim % 32 == 0,
                f"Expected symmetric per-channel INT8: {module}",
            )
            group_size, scale_shape, kind = 32, [out_dim, 1], "channel-int8"
            expect(module + ".weight", {"I8"}, shape)
        expect(module + ".weight_scale", {"BF16", "F16", "F32"}, scale_shape)
        modules[module] = {
            "bits": bits,
            "group_size": group_size,
            "shape": shape,
            "kind": kind,
        }
        for suffix, dtype, width in (
            ("weight", "U32", in_dim // (32 // bits)),
            ("scales", "BF16", in_dim // group_size),
            ("biases", "BF16", in_dim // group_size),
        ):
            output[f"{module}.{suffix}"] = {"dtype": dtype, "shape": [out_dim, width]}
    require(modules, "No quantized Spark modules found")
    for key, shape in plain.items():
        expect(key, {"BF16"}, shape)
        output[key] = {"dtype": "BF16", "shape": shape}
    # Static GPTQ group assignment needs no permutation here. Any serialized
    # g_idx/permutation (or other unhandled tensor) must fail rather than be lost.
    require(
        consumed == set(tensors),
        f"Unexpected tensors: {sorted(set(tensors) - consumed)[:8]}",
    )
    for name, group in groups.items():
        require(
            any(any(matches(s, m) for s in group.get("targets", [])) for m in modules),
            f"Quantization group matches no converted module: {name}",
        )
    return modules, output


def _packed_int4_rows(module, scheme, tensors, first, stop):
    scales = floats(tensors[module + ".weight_scale"], first, stop)
    require(
        bool(np.isfinite(scales).all() and (scales > 0).all()),
        f"Invalid scales: {module}",
    )
    packed = np.frombuffer(read(tensors[module + ".weight_packed"], first, stop), "<u4")
    packed = packed.reshape(stop - first, -1)
    zero = tensors.get(module + ".weight_zero_point")
    if zero is None:
        zeros = np.full(scales.shape, 8, dtype=np.uint32)
    else:
        start, end = first // 8, (stop + 7) // 8
        raw = np.frombuffer(read(zero, start, end), "<u4").reshape(end - start, -1)
        shifts = np.arange(8, dtype=np.uint32)[None, :, None] * 4
        zeros = ((raw[:, None, :] >> shifts) & 15).reshape(-1, scales.shape[1])
        zeros = zeros[first - start * 8 : stop - start * 8]
    scale_data, rounded_scales = bf16(scales)
    require(bool((rounded_scales > 0).all()), f"BF16 scale underflow: {module}")
    offsets = -zeros.astype(np.float32) * scales
    bias_data, rounded_biases = bf16(offsets)
    stats = {
        "max_abs_scale_rounding_error": float(np.max(np.abs(rounded_scales - scales))),
        "max_abs_bias_rounding_error": float(np.max(np.abs(rounded_biases - offsets))),
    }
    return {
        "weight": packed.tobytes(),
        "scales": scale_data.tobytes(),
        "biases": bias_data.tobytes(),
    }, stats


def publish(partial, output):
    # Use a no-replace rename so another process cannot lose its output.
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        result = libc.renamex_np(os.fsencode(partial), os.fsencode(output), 4)
    elif sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        result = libc.renameat2(
            -100, os.fsencode(partial), -100, os.fsencode(output), 1
        )
    else:
        raise NotImplementedError(
            "Atomic publication requires macOS or Linux renameat2"
        )
    if result:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(output))


ACTIVATIONS = {
    "num_bits": 8,
    "type": "int",
    "symmetric": True,
    "strategy": "token",
    "dynamic": True,
}
W4_SUFFIXES = ("self_attn.q_k_v_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
DRAFT_TODO = (
    "TODO: add draft quantization; draft tensors are currently copied unchanged."
)


def _is_w4(module):
    return module.endswith(W4_SUFFIXES)


def _dtype(module):
    return "F32" if module == "lm_head" else "BF16"


def _packed(codes, bits):
    codes = np.asarray(codes, dtype=np.uint32)
    shifts = np.arange(32 // bits, dtype=np.uint32) * bits
    groups = codes.reshape(*codes.shape[:-1], -1, 32 // bits)
    return np.bitwise_or.reduce(groups << shifts, axis=-1).astype("<u4")


def _coefficient(values, dtype):
    require(bool(np.isfinite(values).all()), "Non-finite quantization coefficients")
    if dtype == "BF16":
        return bf16(values)
    values = np.asarray(values, dtype="<f4")
    return values, values


def _plan(config, tensors, quantization):
    require(config.get("model_type") == "spark2_5", "Expected model_type=spark2_5")
    require(not config.get("quantization"), "Source is already in MLX format")
    compressed = bool(config.get("quantization_config"))
    require(
        not compressed or quantization == "mixed",
        "--quantization none requires a BF16 source (INT4 dequantization is unsupported)",
    )
    draft = {k: v for k, v in tensors.items() if k.startswith("draft_model.")}
    # TODO: implement draft quantization after its execution policy is supported.
    require(
        all(v["dtype"] in ("BF16", "F16", "F32") for v in draft.values()),
        "Draft must be floating point; quantized draft checkpoints are unsupported",
    )
    skipped = sorted(k for k in tensors if "rotary_emb.inv_freq" in k)
    target = {k: v for k, v in tensors.items() if k not in draft and k not in skipped}
    if config.get("tie_word_embeddings", True) and "lm_head.weight" in target:
        # Spark's loader uses the embedding when the head is tied.
        target.pop("lm_head.weight")
        skipped.append("lm_head.weight")
    effective = copy.deepcopy(config)
    untie = quantization == "mixed" and config.get("tie_word_embeddings", True)
    if untie:
        effective["tie_word_embeddings"] = False
        if not compressed:
            target["lm_head.weight"] = target["model.embedding.weight"]
    if compressed:
        modules, planned = _plan_compressed(config, target)
        if untie:
            target["lm_head.weight"] = target["model.embedding.weight"]
        # Some 1.7B INT4 sources keep a tied BF16 head. Quantize this head only;
        # retain every existing compressed integer code in the transformer.
        if "lm_head" not in modules:
            head = target.get("lm_head.weight")
            require(
                head is not None and head["dtype"] == "BF16",
                "Missing BF16 lm_head or tied embedding",
            )
            modules["lm_head"] = {
                "bits": 8,
                "group_size": 32,
                "shape": head["shape"],
                "kind": "bf16-int8",
            }
        for module, scheme in modules.items():
            require(
                (_is_w4(module) and scheme["bits"] == 4 and scheme["group_size"] == 32)
                or (
                    (module == "lm_head" or module.endswith("self_attn.out_proj"))
                    and scheme["kind"] in ("channel-int8", "bf16-int8")
                ),
                f"Source quantization does not match the Spark W4/W8 profile: {module}",
            )
        linear, _ = shapes(effective)
        require(
            set(modules) == {m for m in linear if not m.endswith("self_attn.g_proj")},
            "Source must quantize QKV/MLP to INT4 and out_proj/lm_head to INT8",
        )
    else:
        linear, plain = shapes(effective)
        expected = {**plain, **{m + ".weight": s for m, s in linear.items()}}
        require(
            set(target) == set(expected),
            f"BF16 tensor mismatch: missing={sorted(set(expected) - set(target))}, "
            f"unexpected={sorted(set(target) - set(expected))}",
        )
        for key, shape in expected.items():
            require(
                target[key]["dtype"] == "BF16" and target[key]["shape"] == shape,
                f"BF16 dtype/shape mismatch: {key}",
            )
        planned = {
            k: {"dtype": v["dtype"], "shape": v["shape"]} for k, v in target.items()
        }
        modules = {}
        if quantization == "mixed":
            for module, shape in linear.items():
                if module.endswith("self_attn.g_proj"):
                    continue
                require(
                    shape[1] % 32 == 0,
                    f"Input dimension must be divisible by group size 32: {module}",
                )
                modules[module] = {
                    "bits": 4 if _is_w4(module) else 8,
                    "group_size": 32,
                    "shape": shape,
                    "kind": "bf16-int4" if _is_w4(module) else "bf16-int8",
                }
                planned.pop(module + ".weight")
    for module, scheme in modules.items():
        n, k = scheme["shape"]
        planned[module + ".weight"] = {
            "dtype": "U32",
            "shape": [n, k // (32 // scheme["bits"])],
        }
        for suffix in ("scales", "biases"):
            planned[module + "." + suffix] = {
                "dtype": _dtype(module),
                "shape": [n, k // 32],
            }
    planned.update(
        {k: {"dtype": v["dtype"], "shape": v["shape"]} for k, v in draft.items()}
    )
    return effective, target, modules, planned, skipped, untie, compressed


def _rows(module, scheme, tensors, first, stop):
    kind = scheme["kind"]
    if kind == "packed-int4":
        # Preserve source integer codes and unpack zero points along output rows.
        return _packed_int4_rows(module, scheme, tensors, first, stop)
    dtype = _dtype(module)
    if kind == "channel-int8":
        scales = floats(tensors[module + ".weight_scale"], first, stop)
        codes = np.frombuffer(read(tensors[module + ".weight"], first, stop), "i1")
        codes = codes.reshape(stop - first, -1).astype(np.int32) + 128
    else:
        values = floats(tensors[module + ".weight"], first, stop)
        require(bool(np.isfinite(values).all()), f"Non-finite weights: {module}")
        if scheme["bits"] == 4:
            grouped = values.reshape(stop - first, -1, 32)
            lower = np.minimum(grouped.min(axis=-1), 0)
            upper = np.maximum(grouped.max(axis=-1), 0)
            scales = (upper - lower) / np.float32(15)
        else:
            scales = np.max(np.abs(values), axis=-1, keepdims=True) / np.float32(127)
        # All-zero groups/rows have zero codes and a nonzero representable scale.
        scales = np.where(scales == 0, np.float32(1), scales)
    require(
        bool(np.isfinite(scales).all() and (scales > 0).all()),
        f"Invalid scales: {module}",
    )
    scale_data, rounded = _coefficient(scales, dtype)
    require(bool((rounded > 0).all()), f"Scale underflow: {module}")
    if kind == "bf16-int4":
        zeros = np.clip(np.rint(-lower / rounded), 0, 15)
        codes = np.clip(np.rint(grouped / rounded[..., None]) + zeros[..., None], 0, 15)
        codes = codes.reshape(values.shape)
        offsets = -zeros * rounded
    else:
        if kind == "bf16-int8":
            codes = np.clip(np.rint(values / rounded), -127, 127).astype(np.int32) + 128
        # Derive offsets from the stored scale to preserve exact W8 symmetry.
        offsets = -np.float32(128) * rounded
        scale_data = np.repeat(scale_data, scheme["shape"][1] // 32, axis=1)
    bias_data, decoded_bias = _coefficient(offsets, dtype)
    if scheme["bits"] == 8:
        bias_data = np.repeat(bias_data, scheme["shape"][1] // 32, axis=1)
    stats = {
        "max_abs_scale_rounding_error": float(np.max(np.abs(rounded - scales))),
        "max_abs_bias_rounding_error": float(
            np.max(
                np.abs(
                    decoded_bias
                    - (-128 * scales if kind == "channel-int8" else offsets)
                )
            )
        ),
    }
    return {
        "weight": _packed(codes, scheme["bits"]).tobytes(),
        "scales": scale_data.tobytes(),
        "biases": bias_data.tobytes(),
    }, stats


def _write(path, planned, modules, tensors, chunk_rows):
    header, offset = {"__metadata__": {"format": "mlx"}}, 0
    for key, desc in sorted(planned.items()):
        size = math.prod(desc["shape"]) * SIZES[desc["dtype"]]
        header[key] = {**desc, "data_offsets": [offset, offset + size]}
        offset += size
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    base, stats = 8 + len(raw), {}
    converted_keys = {
        m + "." + s for m in modules for s in ("weight", "scales", "biases")
    }
    with path.open("xb") as stream:
        stream.write(struct.pack("<Q", len(raw)))
        stream.write(raw)
        stream.truncate(base + offset)
        for key in sorted(planned.keys() - converted_keys):
            desc = tensors[key]
            stream.seek(base + header[key]["data_offsets"][0])
            stride = desc["bytes"] // desc["shape"][0]
            step = max(1, (8 * 1024 * 1024) // stride)
            for first in range(0, desc["shape"][0], step):
                stream.write(read(desc, first, min(first + step, desc["shape"][0])))
        for i, (module, scheme) in enumerate(modules.items(), 1):
            stats[module] = {}
            for first in range(0, scheme["shape"][0], chunk_rows):
                stop = min(first + chunk_rows, scheme["shape"][0])
                payload, errors = _rows(module, scheme, tensors, first, stop)
                for suffix, data in payload.items():
                    desc = header[module + "." + suffix]
                    stride = math.prod(desc["shape"][1:]) * SIZES[desc["dtype"]]
                    require(
                        len(data) == (stop - first) * stride,
                        f"Converted payload size mismatch: {module}.{suffix}",
                    )
                    stream.seek(base + desc["data_offsets"][0] + first * stride)
                    stream.write(data)
                for key, value in errors.items():
                    stats[module][key] = max(stats[module].get(key, 0.0), value)
            print(
                f"Converted {i}/{len(modules)}: {module}", file=sys.stderr, flush=True
            )
        stream.flush()
        os.fsync(stream.fileno())
    return stats


def convert_to_mlx(
    source: str | Path,
    output: str | Path | None = None,
    *,
    quantization: Literal["mixed", "none"] = "mixed",
    w8_backend: Literal["w8a8", "w8a16", "native"] = "w8a8",
    inspect_only: bool = False,
    chunk_rows: int = 256,
) -> dict[str, Any]:
    """Convert a local BF16 or compressed-tensors Spark checkpoint to MLX.

    Model dimensions are read from the source configuration. Conversion runs
    on CPU in bounded chunks and publishes the output only after validation.
    Draft weights are copied unchanged and are never quantized.

    Args:
        source: Path to a local model directory containing ``config.json``,
            single-file or sharded Safetensors weights, and tokenizer files.
            BF16 and compressed-tensors INT4 checkpoints are supported,
            including checkpoints with ``draft_model.*`` tensors.
        output: Destination directory. It must not exist or be a symlink, and
            neither source nor output may contain the other. Defaults to None,
            which is allowed only when ``inspect_only=True``. If supplied during
            inspection, the same destination path checks still apply.
        quantization: Target weight quantization policy. ``"mixed"`` uses
            asymmetric INT4 with group size 32 for QKV/MLP and symmetric
            per-channel INT8 for out_proj/lm_head, retaining existing compressed
            integer codes when present. A tied lm_head is quantized separately
            from the BF16 embedding. ``"none"`` retains BF16 target weights and
            requires a BF16 source. Defaults to ``"mixed"``.
        w8_backend: Inference policy written into the output configuration for
            W8 layers. ``"w8a8"`` dynamically quantizes activations per token;
            the optimized path requires the W8A8 Metal extension and M5+.
            ``"w8a16"`` uses the optimized floating-activation backend, while
            ``"native"`` uses MLX affine operators without the W8A8 extension.
            Unavailable optimized backends fall back to MLX affine operators
            with floating-point activations. This policy does not change CPU
            conversion execution and is only applied when ``quantization="mixed"``.
            Defaults to ``"w8a8"``.
        inspect_only: Whether to validate the configuration and tensor layout
            without writing output. Inspection does not scan all weight values
            or compute coefficient rounding errors. Defaults to False.
        chunk_rows: Maximum number of quantized weight rows processed per CPU
            chunk. Must be a positive integer. Smaller values reduce temporary
            memory usage. Defaults to 256.

    Returns:
        A JSON-serializable conversion report containing ``status``
        (``"inspected"`` or ``"complete"``), resolved source/output paths,
        ``source_format``, ``quantized_module_counts``, ``lm_head_untied``,
        and draft metadata. Completed conversions additionally include
        ``coefficient_rounding`` with per-module scale/bias rounding errors,
        ``source_quantization_config``, and ``converter_sha256``. The complete
        report is also saved as ``conversion_manifest.json`` in the output.

    Raises:
        ValueError: An argument, model configuration, tensor layout, or weight
            coefficient is invalid or unsupported, or the source files change
            during conversion.
        KeyError: Required configuration or tensor metadata fields are missing.
        FileNotFoundError: A required configuration or tensor file is missing.
        FileExistsError: The destination already exists, including when another
            process creates it before the converted model is published.
        OSError: Reading source files, writing output, or publishing the
            destination fails for another filesystem-related reason.
        NotImplementedError: The platform does not support the required atomic
            directory publication operation.
    """
    source = Path(source).expanduser().resolve()
    require(source.is_dir(), f"Missing source directory: {source}")
    require(quantization in ("mixed", "none"), "Expected quantization=mixed or none")
    require(w8_backend in ("w8a8", "w8a16", "native"), "Invalid W8 backend")
    require(type(chunk_rows) is int and chunk_rows > 0, "chunk_rows must be positive")
    require(inspect_only or output is not None, "--output is required unless --inspect")
    if output is not None:
        output = Path(output).expanduser().absolute()
        require(not output.is_symlink(), "Output must not be a symlink")
        output = output.resolve()
        require(
            source != output
            and source not in output.parents
            and output not in source.parents,
            "Source and output directories must be disjoint",
        )
        if output.exists():
            raise FileExistsError(f"Output already exists: {output}")
    config = json.loads((source / "config.json").read_text())
    tensors, files = inventory(source, include_draft_shards=True)
    effective, target, modules, planned, skipped, untie, compressed = _plan(
        config, tensors, quantization
    )
    assets = [n for n in ASSETS if (source / n).is_file()]
    require(
        "tokenizer_config.json" in assets
        and ("tokenizer.json" in assets or "tokenizer.model" in assets),
        "Missing tokenizer files",
    )
    storage = {"bits": 4, "group_size": 32, "mode": "affine"}
    storage.update(
        {
            m: {"bits": s["bits"], "group_size": 32, "mode": "affine"}
            for m, s in modules.items()
        }
    )
    execution = copy.deepcopy(storage)
    for module, scheme in modules.items():
        if scheme["bits"] == 8 and w8_backend == "w8a8":
            execution[module]["input_activations"] = copy.deepcopy(ACTIVATIONS)
    for key in ("auto_map", "model_file", "compression_config", "quantization_config"):
        effective.pop(key, None)
    effective.update(
        dtype="bfloat16", torch_dtype="bfloat16", quantize_activations=False
    )
    if modules:
        effective.update(
            quantization=storage,
            quantization_config=execution,
            w8_qmv_backend=w8_backend,
        )
    result = {
        "status": "inspected" if inspect_only else "complete",
        "source": str(source),
        "output": str(output) if output else None,
        "source_format": "compressed-tensors" if compressed else "bf16",
        "quantization": quantization,
        "w8_backend": w8_backend if modules else None,
        "quantized_module_counts": {
            str(b): sum(s["bits"] == b for s in modules.values()) for b in (4, 8)
        },
        "weight_codes_requantized": bool(modules) and not compressed,
        "newly_quantized_modules": [
            m for m, s in modules.items() if s["kind"].startswith("bf16-")
        ],
        "lm_head_untied": untie,
        "draft_tensor_count": sum(k.startswith("draft_model.") for k in planned),
        "draft_quantized": False,
        "draft_todo": DRAFT_TODO,
        "skipped_tensors": skipped,
        "output_tensor_bytes": sum(
            math.prod(d["shape"]) * SIZES[d["dtype"]] for d in planned.values()
        ),
        "coefficient_dtypes": {
            "qkv_mlp": "bfloat16",
            "out_proj": "bfloat16",
            "lm_head": "float32",
        }
        if modules
        else {},
    }
    if inspect_only:
        return result
    effective["conversion"] = {
        k: result[k]
        for k in (
            "source_format",
            "quantization",
            "weight_codes_requantized",
            "lm_head_untied",
            "draft_quantized",
            "draft_todo",
            "coefficient_dtypes",
        )
    }
    tracked = [source / n for n in files | set(assets) | {"config.json"}]
    signatures = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in tracked}
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.", suffix=".partial", dir=output.parent
        )
    )
    try:
        main_plan = {
            k: v for k, v in planned.items() if not k.startswith("draft_model.")
        }
        draft_plan = {k: v for k, v in planned.items() if k.startswith("draft_model.")}
        result["coefficient_rounding"] = _write(
            partial / "model.safetensors", main_plan, modules, target, chunk_rows
        )
        mapping = {k: "model.safetensors" for k in main_plan}
        if draft_plan:
            _write(
                partial / "model-draft.safetensors", draft_plan, {}, tensors, chunk_rows
            )
            mapping.update({k: "model-draft.safetensors" for k in draft_plan})
        (partial / "config.json").write_text(json.dumps(effective, indent=2) + "\n")
        for name in assets:
            shutil.copyfile(source / name, partial / name)
        index = {
            "metadata": {"total_size": result["output_tensor_bytes"]},
            "weight_map": dict(sorted(mapping.items())),
        }
        (partial / "model.safetensors.index.json").write_text(
            json.dumps(index, indent=2) + "\n"
        )
        saved, _ = inventory(partial, include_draft_shards=True)
        require(
            set(saved) == set(planned)
            and all(
                saved[k]["dtype"] == d["dtype"] and saved[k]["shape"] == d["shape"]
                for k, d in planned.items()
            ),
            "Written tensor inventory mismatch",
        )
        require(
            all(
                (p.stat().st_size, p.stat().st_mtime_ns) == signatures[p]
                for p in tracked
            ),
            "Source changed during conversion",
        )
        result["source_quantization_config"] = config.get("quantization_config")
        result["converter_sha256"] = hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest()
        (partial / "conversion_manifest.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        publish(partial, output)
    finally:
        if partial.exists():
            shutil.rmtree(partial)
    return result


def main(argv=None, *, prog=None):
    """Run the conversion CLI; retained for module invocation compatibility."""
    from .commands.convert import main as command_main

    return command_main(argv, prog=prog)


if __name__ == "__main__":
    main()
