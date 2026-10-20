#!/usr/bin/env python3
"""Convert Spark compressed-tensors W4/W8 checkpoints to MLX affine Safetensors.

IMPORTANT: Every W8A8 layer becomes W8A16 (BF16 activations), including lm_head.
Integer weight codes are retained, not re-quantized. All affine scales/biases
are stored in BF16; FP32 source scales and affine offsets may be rounded.
This changes inference arithmetic and does not reproduce the source W8A8 model.

Supports Spark2_5 packed groupwise INT4 and symmetric per-channel INT8, including
static GPTQ actorder with no serialized group-index/permutation tensors. Runs on
CPU with NumPy, accepts single/sharded Safetensors, and never overwrites output.
"""

from __future__ import annotations

import argparse
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

import numpy as np

NOTICE = (
    "All source W8A8 layers are converted to W8A16 with BF16 activations, "
    "including attention out_proj and lm_head. Activation quantization is removed. "
    "Integer weight codes are preserved; affine scales and biases are rounded "
    "to BF16. Source W8A8 inference is not numerically reproduced."
)
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


def inventory(source):
    index = source / "model.safetensors.index.json"
    mapping = json.loads(index.read_text())["weight_map"] if index.exists() else None
    names = set(mapping.values()) if mapping else {"model.safetensors"}
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
                mapping is None or mapping.get(key) == name,
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
    require(mapping is None or set(mapping) == set(tensors), "Incomplete shard index")
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


def plan(config, tensors):
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
    modules, output, consumed, changes = {}, {}, set(), []

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
            changes.append(
                {
                    "module": module,
                    "source": "W8A8",
                    "target": "W8A16",
                    "target_activation_dtype": "bfloat16",
                }
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
    quantization = {"bits": 4, "group_size": 32, "mode": "affine"}
    quantization.update(
        {
            m: {"bits": s["bits"], "group_size": s["group_size"], "mode": "affine"}
            for m, s in modules.items()
        }
    )
    return modules, output, quantization, changes


def module_rows(module, scheme, tensors, first, stop):
    scales = floats(tensors[module + ".weight_scale"], first, stop)
    require(
        bool(np.isfinite(scales).all() and (scales > 0).all()),
        f"Invalid scales: {module}",
    )
    if scheme["kind"] == "channel-int8":
        signed = np.frombuffer(read(tensors[module + ".weight"], first, stop), "i1")
        unsigned = signed.view(np.uint8) ^ np.uint8(128)
        packed = unsigned.reshape(stop - first, -1).copy().view("<u4")
        scales = np.repeat(scales, scheme["shape"][1] // 32, axis=1)
        zeros = np.full(scales.shape, 128, dtype=np.uint32)
    else:
        packed = np.frombuffer(
            read(tensors[module + ".weight_packed"], first, stop), "<u4"
        )
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


def write_weights(path, output, modules, tensors, chunk_rows):
    header, offset = {"__metadata__": {"format": "mlx"}}, 0
    for key, desc in sorted(output.items()):
        size = math.prod(desc["shape"]) * SIZES[desc["dtype"]]
        header[key] = {**desc, "data_offsets": [offset, offset + size]}
        offset += size
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    data_start, stats = 8 + len(raw), {}
    with path.open("xb") as target:
        target.write(struct.pack("<Q", len(raw)))
        target.write(raw)
        target.truncate(data_start + offset)
        for key in sorted(set(output) & set(tensors)):
            if any(key == m + ".weight" for m in modules):
                continue
            desc = tensors[key]
            target.seek(data_start + header[key]["data_offsets"][0])
            stride = desc["bytes"] // desc["shape"][0]
            step = max(1, (8 * 1024 * 1024) // stride)
            for first in range(0, desc["shape"][0], step):
                target.write(read(desc, first, min(first + step, desc["shape"][0])))
        for i, (module, scheme) in enumerate(modules.items(), 1):
            stats[module] = {
                "max_abs_scale_rounding_error": 0.0,
                "max_abs_bias_rounding_error": 0.0,
            }
            for first in range(0, scheme["shape"][0], chunk_rows):
                stop = min(first + chunk_rows, scheme["shape"][0])
                payload, errors = module_rows(module, scheme, tensors, first, stop)
                for suffix, data in payload.items():
                    desc = header[f"{module}.{suffix}"]
                    stride = desc["shape"][1] * SIZES[desc["dtype"]]
                    target.seek(data_start + desc["data_offsets"][0] + first * stride)
                    target.write(data)
                for key, value in errors.items():
                    stats[module][key] = max(stats[module][key], value)
            print(
                f"Converted {i}/{len(modules)}: {module}", file=sys.stderr, flush=True
            )
        target.flush()
        os.fsync(target.fileno())
    return stats


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


def convert(source, output=None, *, inspect_only=False, chunk_rows=256):
    source = Path(source).expanduser().resolve()
    require(source.is_dir(), f"Missing source directory: {source}")
    require(type(chunk_rows) is int and chunk_rows > 0, "chunk_rows must be positive")
    require(
        inspect_only or output is not None,
        "--output is required unless --inspect is used",
    )
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
    tensors, files = inventory(source)
    modules, planned, quantization, changes = plan(config, tensors)
    assets = [name for name in ASSETS if (source / name).is_file()]
    require(
        "tokenizer_config.json" in assets
        and ("tokenizer.json" in assets or "tokenizer.model" in assets),
        "Missing tokenizer files",
    )
    counts = {
        str(bits): sum(s["bits"] == bits for s in modules.values()) for bits in (4, 8)
    }
    result = {
        "status": "inspected" if inspect_only else "complete",
        "source": str(source),
        "output": str(output) if output else None,
        "notice": NOTICE,
        "quantized_module_counts": counts,
        "activation_changes": changes,
        "coefficient_dtype": "bfloat16",
        "weight_codes_requantized": False,
        "output_tensor_bytes": sum(
            math.prod(d["shape"]) * SIZES[d["dtype"]] for d in planned.values()
        ),
    }
    if inspect_only:
        return result
    paths = [source / name for name in files | set(assets) | {"config.json"}]
    signatures = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in paths}
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.", suffix=".partial", dir=output.parent
        )
    )
    try:
        stats = write_weights(
            partial / "model.safetensors", planned, modules, tensors, chunk_rows
        )
        target_config = copy.deepcopy(config)
        for key in ("auto_map", "model_file", "compression_config"):
            target_config.pop(key, None)
        target_config.update(
            quantization=quantization,
            quantization_config=copy.deepcopy(quantization),
            quantize_activations=False,
            dtype="bfloat16",
            torch_dtype="bfloat16",
        )
        target_config["conversion"] = {
            "source_format": "compressed-tensors",
            "target_format": "mlx/affine",
            "notice": NOTICE,
            "activation_quantization": "disabled; W8A8 -> W8A16 (BF16)",
            "coefficient_dtype": "bfloat16",
        }
        (partial / "config.json").write_text(json.dumps(target_config, indent=2) + "\n")
        for name in assets:
            shutil.copyfile(source / name, partial / name)
        index = {
            "metadata": {"total_size": result["output_tensor_bytes"]},
            "weight_map": {k: "model.safetensors" for k in sorted(planned)},
        }
        (partial / "model.safetensors.index.json").write_text(
            json.dumps(index, indent=2) + "\n"
        )
        saved, _ = inventory(partial)
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
                (p.stat().st_size, p.stat().st_mtime_ns) == signatures[p] for p in paths
            ),
            "Source changed during conversion",
        )
        result.update(
            source_quantization_config=config["quantization_config"],
            coefficient_rounding=stats,
            converter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        )
        (partial / "conversion_manifest.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        (partial / "CONVERSION_NOTES.txt").write_text(NOTICE + "\n")
        publish(partial, output)
    finally:
        if partial.exists():
            shutil.rmtree(partial)
    return result


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--source", required=True, type=Path, help="Local compressed-tensors directory"
    )
    parser.add_argument("--output", type=Path, help="New MLX directory; must not exist")
    parser.add_argument(
        "--inspect", action="store_true", help="Validate metadata only; write nothing"
    )
    parser.add_argument(
        "--chunk-rows", type=int, default=256, help="Rows processed on CPU at once"
    )
    args = parser.parse_args()
    if not args.inspect and args.output is None:
        parser.error("--output is required unless --inspect is used")
    print(NOTICE, file=sys.stderr, flush=True)
    try:
        result = convert(
            args.source,
            args.output,
            inspect_only=args.inspect,
            chunk_rows=args.chunk_rows,
        )
    except (ValueError, OSError, KeyError, NotImplementedError) as error:
        parser.exit(1, f"Conversion failed: {error}\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k
                not in (
                    "source_quantization_config",
                    "coefficient_rounding",
                    "activation_changes",
                )
            },
            indent=2,
        )
    )
    print(f"W8A8 -> W8A16 layers: {len(result['activation_changes'])}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
