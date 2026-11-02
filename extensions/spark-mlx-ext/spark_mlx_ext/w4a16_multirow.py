"""Small-multirow affine W4/G32 projections for speculative verification.

Schedules preserve packed weight codes and BF16 scale/bias values. Accumulation
is FP32 with one BF16 output rounding. Explicit experimental two-term matrix
variants and changed reduction order need independent numerical validation.
"""

import math
from functools import lru_cache
from string import Template

import mlx.core as mx

from ._ext import is_w8a8_available
from ._validation import gpu_stream
from ._w4a16_multirow_sources import (
    HEADER,
    MERGE,
    MMA,
    MMA2,
    NAX,
    NAX4,
    NAX4_PACK,
    NAX_HEADER,
    P2,
    SPLIT_P2,
    WIDE,
)


@lru_cache(maxsize=128)
def _wide_kernel(m, n, k, kl, sg, p2=False):
    return mx.fast.metal_kernel(
        name=f"spark_w4_multi_{'p2' if p2 else 'wide'}_m{m}_n{n}_k{k}_l{kl}_s{sg}",
        input_names=["x", "weight", "scales", "biases"],
        output_names=["out"],
        header=HEADER,
        source=Template(P2 if p2 else WIDE).substitute(M=m, N=n, K=k, KL=kl, SG=sg),
        ensure_row_contiguous=True,
        compile_options={"math_mode": "safe"},
    )


@lru_cache(maxsize=128)
def _mma_kernel(m, n, k, splits, sg, two_part=False, interleaved=False):
    return mx.fast.metal_kernel(
        name=f"spark_w4_multi_mma{'2' if two_part else ''}_m{m}_n{n}_k{k}_p{splits}_s{sg}_i{int(interleaved)}",
        input_names=["x", "weight", "scales", "biases"],
        output_names=["partials"],
        header=HEADER,
        source=Template(MMA2 if two_part else MMA).substitute(
            M=m,
            N=n,
            K=k,
            SPLITS=splits,
            SG=sg,
            OT="bfloat16_t" if splits == 1 else "float",
            INTERLEAVED="true" if interleaved else "false",
        ),
        ensure_row_contiguous=True,
        compile_options={"math_mode": "safe"},
    )


@lru_cache(maxsize=128)
def _nax_kernel(m, n, k, splits, sg, two_part=False, interleaved=False):
    return mx.fast.metal_kernel(
        name=f"spark_w4_multi_nax{'2' if two_part else ''}_m{m}_n{n}_k{k}_p{splits}_s{sg}_i{int(interleaved)}",
        input_names=["x", "weight", "scales", "biases"],
        output_names=["partials"],
        header=NAX_HEADER,
        source=Template(NAX).substitute(
            M=m,
            N=n,
            K=k,
            SG=sg,
            SPLITS=splits,
            OT="bfloat16_t" if splits == 1 else "float",
            INTERLEAVED="true" if interleaved else "false",
            TWO_PART="true" if two_part else "false",
            BTYPE="bfloat16_t" if two_part else "float",
        ),
        ensure_row_contiguous=True,
        compile_options={"math_mode": "safe"},
    )


@lru_cache(maxsize=128)
def _nax4_kernel(
    m, n, k, splits, sg, tiled=False, static_layout=False, multiply_mode=False,
    static_slice=False, contiguous_k=False,
):
    return mx.fast.metal_kernel(
        name=f"spark_w4_multi_nax4_m{m}_n{n}_k{k}_p{splits}_s{sg}_t{int(tiled)}_l{int(static_layout)}_z{int(multiply_mode)}_e{int(static_slice)}_c{int(contiguous_k)}",
        input_names=["x", "weight", "scales", "biases", "group_sums"],
        output_names=["partials"],
        header=NAX_HEADER,
        source=Template(NAX4).substitute(
            M=m,
            N=n,
            K=k,
            SG=sg,
            SPLITS=splits,
            OT="bfloat16_t" if splits == 1 else "float",
            TILED="true" if tiled else "false",
            STATIC_LAYOUT="true" if static_layout else "false",
            MULTIPLY_MODE="true" if multiply_mode else "false",
            MODE="multiply" if multiply_mode else "multiply_accumulate",
            SLICE="slice<32, 32>" if static_slice else "slice",
            GROUP_BEGIN="split * (K / 32) / SPLITS" if contiguous_k else "split",
            GROUP_END="(split + 1) * (K / 32) / SPLITS" if contiguous_k else "K / 32",
            GROUP_STEP="1" if contiguous_k else "SPLITS",
        ),
        ensure_row_contiguous=True,
        compile_options={"math_mode": "safe"},
    )


@lru_cache(maxsize=128)
def _split_p2_kernel(m, n, k, splits):
    return mx.fast.metal_kernel(
        name=f"spark_w4_multi_splitp2_m{m}_n{n}_k{k}_p{splits}",
        input_names=["x", "weight", "scales", "biases"],
        output_names=["out"],
        header=HEADER,
        source=Template(SPLIT_P2).substitute(M=m, N=n, K=k, SPLITS=splits),
        ensure_row_contiguous=True,
        compile_options={"math_mode": "safe"},
    )


def prepare_w4a16_multirow(weight, scales, biases):
    """Lazily reorder exact packed bytes/metadata into eight-output tiles.

    Returns same-shaped arrays whose byte layout is consumed only with
    ``layout='interleaved8'``. Call once outside timed generation. This does
    not quantize, modify input arrays, evaluate work, or cache parameters.
    """
    if any(not isinstance(v, mx.array) for v in (weight, scales, biases)):
        raise TypeError("weight, scales and biases must be MLX arrays")
    if (
        weight.ndim != 2
        or weight.dtype != mx.uint32
        or weight.shape[0] <= 0
        or weight.shape[0] % 8
    ):
        raise ValueError(
            "Interleaved W4 preparation needs UINT32 weight [N,K/8], N divisible by8"
        )
    n, p = weight.shape
    if (
        p <= 0
        or p % 4
        or scales.shape != (n, p // 4)
        or biases.shape != scales.shape
        or scales.dtype != mx.bfloat16
        or biases.dtype != mx.bfloat16
    ):
        raise ValueError("Interleaved W4 preparation needs BF16 metadata [N,K/32]")
    return tuple(
        mx.contiguous(v.reshape(n // 8, 8, v.shape[1]).transpose(0, 2, 1)).reshape(
            v.shape
        )
        for v in (weight, scales, biases)
    )


@lru_cache(maxsize=32)
def _nax4_pack_kernel(n, k):
    return mx.fast.metal_kernel(
        name=f"spark_w4_pack_nax4_n{n}_k{k}",
        input_names=["weight"],
        output_names=["out"],
        header=HEADER,
        source=Template(NAX4_PACK).substitute(N=n, K=k),
        ensure_row_contiguous=True,
        compile_options={"math_mode": "safe"},
    )


def prepare_w4a16_nax4(weight, scales, biases):
    """Lazily reorder UInt4 codes into [N/256,K/32,32K,256N] blocks.

    Same-shaped output arrays require layout='nax_tiled256'. Preparation does
    not change values or inputs and must occur outside steady-state timing.
    K and N nibble axes are transposed by an exact GPU packing kernel, enabling
    MPP's required 256-element leading alignment without padding parameter bytes.
    """
    if any(not isinstance(v, mx.array) for v in (weight, scales, biases)):
        raise TypeError("weight, scales and biases must be MLX arrays")
    if (
        weight.ndim != 2
        or weight.dtype != mx.uint32
        or weight.shape[0] <= 0
        or weight.shape[0] % 256
    ):
        raise ValueError("NAX4 packing needs UINT32[N,K/8], N divisible by256")
    n, p = weight.shape
    if (
        p <= 0
        or p % 4
        or scales.shape != (n, p // 4)
        or biases.shape != scales.shape
        or scales.dtype != mx.bfloat16
        or biases.dtype != mx.bfloat16
    ):
        raise ValueError("NAX4 packing needs BF16 metadata [N,K/32]")
    gpu_stream(None, operator="NAX4 packing")
    packed = _nax4_pack_kernel(n, p * 8)(
        inputs=[weight],
        output_shapes=[weight.shape],
        output_dtypes=[mx.uint32],
        grid=(n * p, 1, 1),
        threadgroup=(256, 1, 1),
    )[0]
    meta = tuple(
        mx.contiguous(v.reshape(n // 256, 256, p // 4).transpose(0, 2, 1)).reshape(
            v.shape
        )
        for v in (scales, biases)
    )
    return (packed, *meta)


@lru_cache(maxsize=64)
def _merge_kernel(size, splits):
    return mx.fast.metal_kernel(
        name=f"spark_w4_multi_merge_n{size}_p{splits}",
        input_names=["partials"],
        output_names=["out"],
        header=HEADER,
        source=Template(MERGE).substitute(SIZE=size, SPLITS=splits),
        ensure_row_contiguous=True,
        compile_options={"math_mode": "safe"},
    )


def w4a16_multirow(
    x,
    weight,
    scales,
    biases,
    *,
    algorithm="wide",
    k_lanes=8,
    splits=4,
    simdgroups=2,
    layout="native",
    static_layout=False,
    multiply_mode=False,
    static_slice=False,
    contiguous_k=False,
    stream=None,
):
    """Return BF16 affine W4/G32 matmul for 2..8 activation rows.

    Leading dimensions of x may have product M in [2,8], K is a positive
    multiple of 32, native UINT32 weight is [N,K/8], and BF16 scale/bias are
    [N,K/32]. N is positive. Input layouts are accepted through lazy contiguous
    conversion. Inputs are never modified; no eval or synchronization occurs.
    CPU execution and unsupported descriptors raise instead of falling back.

    ``wide`` decodes each weight chunk once for all M rows. ``p2`` factors the
    affine dot into scaled unsigned-code and activation-sum terms. ``mma``
    constructs FP32 8x8 fragments directly from the same codes/scales. ``mma2``
    explicitly represents decoded weights as high+low BF16 terms and performs
    two BF16 matrix products into FP32 accumulators. This decomposition is
    exact for many ordinary W4 affine descriptors but not guaranteed for all
    possible BF16 metadata; it remains an experimental arithmetic variant.
    ``nax`` uses M5 MPP 16x32x16 cooperative tensors with BF16 inputs and FP32
    reconstructed weights. ``nax2`` uses two BF16 weight terms instead. Both
    NAX descriptors explicitly disable relaxed precision; no TF32 switch is
    enabled. Matrix schedules can consume once-prepared interleaved8 byte
    layout from prepare_w4a16_multirow; native arrays must never be passed
    with that layout flag, or prepared arrays with layout='native'.
    ``nax4`` instead multiplies exact BF16 x by the original UInt4 codes in
    M8xN32xK32 MPP tiles, then applies each G32 affine scale/bias in FP32 using
    FP32 activation group sums. It preserves all codes and metadata, but
    algebraically reassociates the affine dot. It accepts native layout with
    N divisible by32 or exact once-prepared nax_tiled256 layout with N divisible
    by256. Group sum construction is included in the lazy call. Experimental
    ``multiply_mode=True`` selects MPP's overwrite product for each G32 group
    instead of initializing C to zero before multiply-accumulate. The default
    retains the validated multiply-accumulate schedule.
    ``static_slice=True`` declares full 32x32 weight slices to MPP without
    hardcoding cooperative-thread layouts. ``contiguous_k=True`` partitions
    G32 groups into contiguous intervals rather than interleaved split-K;
    uneven intervals and empty partitions are supported. Both are opt-in
    nax4 schedules; contiguous_k changes FP32 reduction order.
    All schedules require separate numerical and model validation.
    """
    values = (x, weight, scales, biases)
    if any(not isinstance(v, mx.array) for v in values):
        raise TypeError("x, weight, scales and biases must be MLX arrays")
    if (
        x.dtype != mx.bfloat16
        or scales.dtype != mx.bfloat16
        or biases.dtype != mx.bfloat16
    ):
        raise ValueError("W4 multirow requires BF16 x, scales and biases")
    if weight.dtype != mx.uint32:
        raise ValueError("W4 multirow requires UINT32-packed weights")
    if x.ndim < 2 or not 2 <= math.prod(x.shape[:-1]) <= 8:
        raise ValueError("W4 multirow requires 2..8 activation rows")
    m, k = math.prod(x.shape[:-1]), x.shape[-1]
    if k <= 0 or k % 32:
        raise ValueError("W4 multirow requires positive K divisible by 32")
    if weight.ndim != 2 or weight.shape[0] <= 0 or weight.shape[1] != k // 8:
        raise ValueError("W4 multirow requires positive weight [N,K/8]")
    n = weight.shape[0]
    if scales.shape != (n, k // 32) or biases.shape != scales.shape:
        raise ValueError("W4 multirow requires metadata [N,K/32]")
    if algorithm not in (
        "wide",
        "mma",
        "mma2",
        "p2",
        "nax",
        "nax2",
        "nax4",
        "split_p2",
    ):
        raise ValueError("Unsupported W4 multirow algorithm")
    for option_name, value in (("static_slice", static_slice), ("contiguous_k", contiguous_k)):
        if type(value) is not bool:
            raise ValueError(f"{option_name} must be bool")
        if value and algorithm != "nax4":
            raise ValueError(f"{option_name} requires algorithm=nax4")
    if type(multiply_mode) is not bool:
        raise ValueError("multiply_mode must be bool")
    if multiply_mode and algorithm != "nax4":
        raise ValueError("multiply_mode requires algorithm=nax4")
    if algorithm == "split_p2" and (
        k % 512 or n % 4 or layout != "native" or splits > 8
    ):
        raise ValueError(
            "split_p2 requires K divisible by512, N divisible by4, native layout, splits<=8"
        )
    if algorithm == "nax4" and (n % 32 or layout not in ("native", "nax_tiled256")):
        raise ValueError(
            "nax4 requires native/nax_tiled256 layout and N divisible by32"
        )
    if layout not in ("native", "interleaved8", "nax_tiled256"):
        raise ValueError("layout must be native, interleaved8 or nax_tiled256")
    if layout == "nax_tiled256" and algorithm != "nax4":
        raise ValueError("nax_tiled256 requires algorithm=nax4")
    if layout == "nax_tiled256" and n % 256:
        raise ValueError("nax_tiled256 requires N divisible by256")
    if layout == "interleaved8" and (n % 8 or algorithm in ("wide", "p2")):
        raise ValueError("Interleaved8 requires N divisible by8 and a matrix algorithm")
    if type(k_lanes) is not int or k_lanes not in (4, 8, 16, 32):
        raise ValueError("k_lanes must be 4, 8, 16 or 32")
    if type(simdgroups) is not int or simdgroups not in (1, 2, 4, 8):
        raise ValueError("simdgroups must be 1, 2, 4 or 8")
    if type(splits) is not int or splits not in (1, 2, 4, 8, 16):
        raise ValueError("splits must be 1, 2, 4, 8 or 16")
    stream = gpu_stream(stream, operator="W4 multirow")
    if type(static_layout) is not bool:
        raise ValueError("static_layout must be bool")
    if static_layout and not mx.device_info().get("architecture", "").startswith(
        "applegpu_g17"
    ):
        raise ValueError("Static NAX layout is validated only on Apple GPU g17")
    if algorithm.startswith("nax") and not is_w8a8_available():
        raise ValueError("NAX W4 schedules require Apple M5+ and macOS26.2+")
    shape = (*x.shape[:-1], n)
    if algorithm == "split_p2":
        return _split_p2_kernel(m, n, k, splits)(
            inputs=list(values),
            output_shapes=[shape],
            output_dtypes=[mx.bfloat16],
            grid=(n // 4 * 32 * splits, 1, 1),
            threadgroup=(32 * splits, 1, 1),
            stream=stream,
        )[0]
    if algorithm in ("wide", "p2"):
        rows = 32 // k_lanes * simdgroups
        return _wide_kernel(m, n, k, k_lanes, simdgroups, algorithm == "p2")(
            inputs=list(values),
            output_shapes=[shape],
            output_dtypes=[mx.bfloat16],
            grid=(((n + rows - 1) // rows) * 32 * simdgroups, 1, 1),
            threadgroup=(32 * simdgroups, 1, 1),
            stream=stream,
        )[0]
    matrix_kernel = _nax_kernel if algorithm.startswith("nax") else _mma_kernel
    matrix_rows = 32 if algorithm.startswith("nax") else 8
    if algorithm == "nax4":
        matrix_callable = _nax4_kernel(
            m,
            n,
            k,
            splits,
            simdgroups,
            layout == "nax_tiled256",
            static_layout,
            multiply_mode,
            static_slice,
            contiguous_k,
        )
        sums = mx.sum(x.reshape(m, k // 32, 32).astype(mx.float32), axis=-1)
        matrix_inputs = [*values, sums]
    else:
        matrix_callable = matrix_kernel(
            m,
            n,
            k,
            splits,
            simdgroups,
            algorithm in ("mma2", "nax2"),
            layout == "interleaved8",
        )
        matrix_inputs = list(values)
    partials = matrix_callable(
        inputs=matrix_inputs,
        output_shapes=[shape if splits == 1 else (splits, m, n)],
        output_dtypes=[mx.bfloat16 if splits == 1 else mx.float32],
        grid=(
            ((n + matrix_rows * simdgroups - 1) // (matrix_rows * simdgroups))
            * 32
            * simdgroups,
            splits,
            1,
        ),
        threadgroup=(32 * simdgroups, 1, 1),
        stream=stream,
    )[0]
    if splits == 1:
        return partials
    return _merge_kernel(m * n, splits)(
        inputs=[partials],
        output_shapes=[shape],
        output_dtypes=[mx.bfloat16],
        grid=(m * n, 1, 1),
        threadgroup=(256, 1, 1),
        stream=stream,
    )[0]
