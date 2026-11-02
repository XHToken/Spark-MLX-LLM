"""Optional draft FFN, row-padding and immutable Markov-cache helpers.

The FFN and padding paths preserve source parameters and ordinary fallbacks.
The Markov helper prepares separate native-M1 bias rows without changing the
model parameter tree; its handle restores or reactivates the prepared lookup.
"""

import math

import mlx.core as mx
from mlx import nn


@mx.compile
def _compiled_dense_residual_ffn(
    hidden, norm_weight, gate_weight, up_weight, down_weight, eps
):
    normalized = mx.fast.rms_norm(hidden, norm_weight, eps)
    gate = normalized @ gate_weight.T
    up = normalized @ up_weight.T
    projected = (nn.gelu(gate) * up) @ down_weight.T
    return hidden + projected


def draft_residual_ffn(hidden, norm, mlp, *, enabled=True):
    """Run BF16 norm/MLP/residual as one compiled pure graph when eligible.

    This does not quantize weights or replace exact GELU with an approximation.
    Compiler fusion can change BF16 intermediate rounding, so callers must
    validate real draft probabilities before enabling it in measured generation.
    """
    from .drafter import SparkDraftMLP

    projections = tuple(
        getattr(mlp, name, None) for name in ("gate_proj", "up_proj", "down_proj")
    )
    if (
        enabled
        and mx.default_device() == mx.gpu
        and mx.metal.is_available()
        and hidden.dtype == mx.bfloat16
        and type(mlp) is SparkDraftMLP
        and type(norm) is nn.RMSNorm
        and norm.weight.dtype == mx.bfloat16
        and all(
            type(p) is nn.Linear and "bias" not in p and p.weight.dtype == mx.bfloat16
            for p in projections
        )
    ):
        gate, up, down = projections
        return _compiled_dense_residual_ffn(
            hidden, norm.weight, gate.weight, up.weight, down.weight, norm.eps
        )
    return hidden + mlp(norm(hidden))


class DraftLinearPadding:
    """Temporary experiment patch scoped to existing dense draft module IDs.

    nn.Linear.__call__ is temporarily wrapped because Python special methods
    are looked up on the class. Every unselected module immediately calls the
    original implementation, including target linears and quantized linears.
    Parameters/modules retain their identities and types. Restore in a finally
    block or use the returned handle as a context manager. Nested handles must
    be restored in reverse installation order.
    """

    def __init__(self, drafter, *, source_rows=7, padded_rows=16):
        from .drafter import DSparkDrafter

        if type(drafter) is not DSparkDrafter:
            raise TypeError("Draft padding requires an exact DSparkDrafter instance")
        if (
            type(source_rows) is not int
            or type(padded_rows) is not int
            or source_rows <= 1
            or padded_rows <= source_rows
        ):
            raise ValueError("Draft padding requires 1 < source_rows < padded_rows")
        named = [
            (name, module)
            for name, module in drafter.named_modules()
            if type(module) is nn.Linear
        ]
        selected = {id(module): name for name, module in named}
        self.module_names = [name for name, _ in named]
        self.source_rows, self.padded_rows = source_rows, padded_rows
        self.stats = {"selected_calls": 0, "padded_calls": 0, "padded_modules": {}}
        self._original = nn.Linear.__call__
        self._restored = False

        def call(projection, x):
            name = selected.get(id(projection))
            if name is not None:
                self.stats["selected_calls"] += 1
                if (
                    x.ndim >= 2
                    and math.prod(x.shape[:-1]) == source_rows
                    and x.dtype == projection.weight.dtype == mx.bfloat16
                ):
                    self.stats["padded_calls"] += 1
                    self.stats["padded_modules"][name] = (
                        self.stats["padded_modules"].get(name, 0) + 1
                    )
                    flattened = x.reshape(source_rows, x.shape[-1])
                    padded = mx.pad(flattened, [(0, padded_rows - source_rows), (0, 0)])
                    result = self._original(projection, padded)[:source_rows]
                    return result.reshape(*x.shape[:-1], result.shape[-1])
            return self._original(projection, x)

        self._wrapper = call
        nn.Linear.__call__ = call

    def restore(self):
        if self._restored:
            return
        if nn.Linear.__call__ is not self._wrapper:
            raise RuntimeError(
                "Restore draft padding handles in reverse installation order"
            )
        nn.Linear.__call__ = self._original
        self._restored = True

    def __call__(self):
        self.restore()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.restore()


def install_draft_linear_padding(drafter, *, source_rows=7, padded_rows=16):
    """Install optional pure-row draft padding and return a restorable handle.

    The external embedding readout is not a registered draft nn.Linear and is
    deliberately not patched. A one-row Markov projection remains unchanged.
    Compiled explicit-weight FFN helpers bypass nn.Linear.__call__, so benchmark
    this experiment separately before combining it with such helpers.
    """
    return DraftLinearPadding(drafter, source_rows=source_rows, padded_rows=padded_rows)


class MarkovBiasCache:
    """Optional fixed-weight Markov cache for the generic first C token IDs.

    Cache entries are computed individually through native M1 step_bias, so hit
    values retain that exact stored BF16 result. No dataset or generated text
    selects cache IDs. The device kernel executes its fallback only on misses.
    Checkpoint parameters must remain immutable while this handle is installed;
    reassigning weight objects automatically uses the native method instead.
    In-place weight mutation is unsupported until the handle is restored.
    """

    def __init__(self, drafter, *, cache_size=512):
        from .drafter import DSparkDrafter, VanillaMarkovHead

        if (
            type(drafter) is not DSparkDrafter
            or type(drafter.markov_head) is not VanillaMarkovHead
        ):
            raise TypeError(
                "Markov cache requires an exact DSparkDrafter with vanilla Markov head"
            )
        head = drafter.markov_head
        original = head.step_bias
        if (
            getattr(original, "__func__", None) is not VanillaMarkovHead.step_bias
            or getattr(head.prev_embeddings, "__func__", None)
            is not VanillaMarkovHead.prev_embeddings
            or type(head.markov_w1) is not nn.Embedding
            or type(head.markov_w2) is not nn.Linear
        ):
            raise ValueError(
                "Markov cache preparation requires native Markov methods and modules"
            )
        embedding, projection = head.markov_w1.weight, head.markov_w2.weight
        if (
            embedding.dtype != mx.bfloat16
            or projection.dtype != mx.bfloat16
            or embedding.shape[1] % 32
            or "bias" in head.markov_w2
        ):
            raise ValueError(
                "Markov cache requires immutable BF16 bias-free weights with rank divisible by32"
            )
        if type(cache_size) is not int or not 0 < cache_size <= embedding.shape[0]:
            raise ValueError("Markov cache_size must be positive and fit vocabulary")
        rows = []
        for token in range(cache_size):
            value = original(mx.array([token], mx.int32))
            mx.eval(value)
            rows.append(value)
        self.cache = mx.concatenate(rows, axis=0)
        self.lookup = mx.where(
            mx.arange(embedding.shape[0]) < cache_size,
            mx.arange(embedding.shape[0]),
            -1,
        ).astype(mx.int32)
        mx.eval(self.cache, self.lookup)
        self.cache_size = cache_size
        self.cache_nbytes = self.cache.nbytes + self.lookup.nbytes
        self.embedding, self.projection = embedding, projection
        self._embedding_module, self._projection_module = head.markov_w1, head.markov_w2
        self._head, self._original = head, original
        self._had_instance = "step_bias" in head.__dict__
        self._instance = head.__dict__.get("step_bias")
        self._had_cache_attribute = "_runtime_markov_cache" in head.__dict__
        self._cache_attribute = head.__dict__.get("_runtime_markov_cache")
        self._restored = True
        self._wrapper = None
        self.activate()

    def activate(self):
        """Rebind a prepared immutable cache without rebuilding any tensor."""
        import spark_mlx_ext as ext

        if not self._restored:
            if self._head.step_bias is not self._wrapper:
                raise RuntimeError("Active Markov step_bias changed")
            return self
        head = self._head
        embedding, projection = self.embedding, self.projection
        original = self._original
        current = head.step_bias
        if (
            head.markov_w1 is not self._embedding_module
            or head.markov_w2 is not self._projection_module
            or head.markov_w1.weight is not embedding
            or head.markov_w2.weight is not projection
            or "bias" in head.markov_w2
        ):
            raise RuntimeError("Source Markov weights changed; prepare a fresh cache")
        if (
            getattr(current, "__func__", None)
            is not getattr(original, "__func__", None)
            or getattr(current, "__self__", None) is not head
            or getattr(head.prev_embeddings, "__func__", None)
            is not type(head).prev_embeddings
        ):
            raise RuntimeError(
                "Restore other Markov methods before activating this cache"
            )
        if head.__dict__.get("_runtime_markov_cache") is not self._cache_attribute:
            raise RuntimeError("Restore other runtime Markov cache before activation")

        def step_bias(ids):
            if (
                head.markov_w1 is not self._embedding_module
                or head.markov_w2 is not self._projection_module
                or head.markov_w1.weight is not embedding
                or head.markov_w2.weight is not projection
                or "bias" in head.markov_w2
                or getattr(head.prev_embeddings, "__func__", None)
                is not type(head).prev_embeddings
            ):
                return original(ids)
            return ext.cached_markov_bias(
                ids.astype(mx.int32), embedding, projection, self.cache, self.lookup
            )

        self._wrapper = step_bias
        object.__setattr__(head, "step_bias", step_bias)
        object.__setattr__(head, "_runtime_markov_cache", self)
        self._restored = False
        return self

    def restore(self):
        if self._restored:
            return
        if self._head.step_bias is not self._wrapper:
            raise RuntimeError("Markov step_bias changed before cache restore")
        if self._had_instance:
            object.__setattr__(self._head, "step_bias", self._instance)
        else:
            object.__delattr__(self._head, "step_bias")
        if self._had_cache_attribute:
            object.__setattr__(
                self._head, "_runtime_markov_cache", self._cache_attribute
            )
        else:
            object.__delattr__(self._head, "_runtime_markov_cache")
        self._restored = True
        # Break the wrapper->handle reference cycle so large derived tables can
        # be reclaimed promptly when the caller releases a restored handle.
        self._wrapper = None

    def __call__(self):
        self.restore()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.restore()


def install_markov_bias_cache(drafter, *, cache_size=512):
    """Prepare generic native-M1 entries and install a restorable device lookup.

    Preparation allocates about cache_size*vocab_size*2 bytes, temporarily twice
    that during concatenation. It must run outside timed generation, under the
    caller's shared device lock. The original draft parameter tree is unchanged.
    """
    return MarkovBiasCache(drafter, cache_size=cache_size)


def _draft_context_projection(context, projection, q_size, kv_size):
    """Project only K/V rows when the original projection has exact known semantics."""
    weight = getattr(projection, "weight", None)
    if (
        context.dtype != mx.bfloat16
        or not isinstance(weight, mx.array)
        or weight.ndim != 2
        or weight.shape[0] != q_size + 2 * kv_size
    ):
        return None
    if type(projection) is nn.Linear:
        if projection.weight.dtype != mx.bfloat16 or (
            "bias" in projection and projection.bias.dtype != mx.bfloat16
        ):
            return None
        weight = projection.weight[q_size:]
        if "bias" in projection:
            return mx.addmm(projection.bias[q_size:], context, weight.T)
        return context @ weight.T

    from .draft_quantization import DerivedDraftPerChannelW8Linear

    if not (
        type(projection) is DerivedDraftPerChannelW8Linear
        and mx.default_device() == mx.gpu
        and mx.metal.is_available()
        and context.shape[-1] % 32 == 0
    ):
        return None
    rows = math.prod(context.shape[:-1])
    # Real five-layer replay regressed at M1: retain the ordinary W8 call.
    # BF16 dense M1 remains eligible above, where row slicing reduced latency.
    if not (
        2 <= rows <= 8
        and (
            projection._variant != "nax"
            or 2 * kv_size % 32 == 0
            and context.shape[-1] % 64 == 0
        )
    ):
        return None
    import spark_mlx_ext as ext

    weight, scales = projection.weight[q_size:], projection.scales[q_size:]
    value = ext.w8a16_multirow(
        context,
        weight,
        scales,
        output_dtype=mx.bfloat16,
        variant=projection._variant,
        simdgroups=4,
    )
    return value + projection.bias[q_size:] if "bias" in projection else value


class DraftContextKV:
    """Restorable KV-only context projection for exact draft attention modules.

    Q rows are unused by context commits. This helper slices the current dense
    BF16 or eligible derived per-channel W8 parameters before projection. Module
    objects, parameter arrays, head layout and RoPE offsets are preserved.
    Unsupported/custom projections keep their ordinary full projection call.
    GPU algorithm selection may change BF16 rounding; validate actual proposals
    before enabling this experiment in a measured generation candidate.
    """

    def __init__(self, drafter):
        from .drafter import DSparkDrafter, SparkDraftAttention

        if type(drafter) is not DSparkDrafter:
            raise TypeError("Context KV projection requires exact DSparkDrafter")
        self._drafter = drafter
        self._entries = []
        self.stats = {"kv_only_calls": 0, "fallback_calls": 0}
        self._restored = True
        for layer in drafter.layers:
            attention = layer.self_attn
            original = attention.project_context
            if (
                type(attention) is not SparkDraftAttention
                or getattr(original, "__func__", None)
                is not SparkDraftAttention.project_context
            ):
                raise ValueError(
                    "Context KV projection requires native draft attention"
                )
            stats = self.stats

            def project_context(
                context,
                *,
                context_offset,
                attention=attention,
                original=original,
                stats=stats,
            ):
                projected = None
                if attention.spec is not None:
                    projected = _draft_context_projection(
                        context,
                        attention.q_k_v_proj,
                        attention.q_size,
                        attention.kv_size,
                    )
                if projected is None:
                    stats["fallback_calls"] += 1
                    return original(context, context_offset=context_offset)
                stats["kv_only_calls"] += 1
                context_k, context_v = mx.split(projected, [attention.kv_size], axis=-1)
                context_k = attention.rope(
                    attention._heads(context_k, attention.num_kv_heads),
                    offset=context_offset,
                )
                return context_k, attention._heads(context_v, attention.num_kv_heads)

            self._entries.append(
                (
                    attention,
                    original,
                    "project_context" in attention.__dict__,
                    attention.__dict__.get("project_context"),
                    project_context,
                )
            )
        self.activate()

    def activate(self):
        if not self._restored:
            if any(
                attention.project_context is not wrapper
                for attention, _, _, _, wrapper in self._entries
            ):
                raise RuntimeError("Active draft context projection changed")
            return self
        if len(self._drafter.layers) != len(self._entries) or any(
            layer.self_attn is not entry[0]
            for layer, entry in zip(self._drafter.layers, self._entries)
        ):
            raise RuntimeError(
                "Draft attention modules changed before context activation"
            )
        for attention, original, _, _, _ in self._entries:
            current = attention.project_context
            if (
                getattr(current, "__func__", None) is not original.__func__
                or getattr(current, "__self__", None) is not attention
            ):
                raise RuntimeError(
                    "Restore other draft context methods before activation"
                )
        for attention, _, _, _, wrapper in self._entries:
            object.__setattr__(attention, "project_context", wrapper)
        self._restored = False
        return self

    def restore(self):
        if self._restored:
            return
        if any(
            attention.project_context is not wrapper
            for attention, _, _, _, wrapper in self._entries
        ):
            raise RuntimeError("Draft context projection changed before restore")
        for attention, _, had_instance, instance, _ in reversed(self._entries):
            if had_instance:
                object.__setattr__(attention, "project_context", instance)
            else:
                object.__delattr__(attention, "project_context")
        self._restored = True

    def __call__(self):
        self.restore()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.restore()


def install_draft_context_kv(drafter):
    """Skip unused context Q rows through an optional restorable projection handle."""
    return DraftContextKV(drafter)
