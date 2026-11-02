"""Probability transforms and greedy/probabilistic token selection."""

import math
from dataclasses import dataclass, replace
from functools import cache, partial

import mlx.core as mx


@dataclass(frozen=True)
class SamplingOptions:
    """Sampling policy for either model; use separate instances to tune each."""

    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0

    def __post_init__(self):
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        if not math.isfinite(self.top_p) or not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if type(self.top_k) is not int or self.top_k < 0:
            raise ValueError("top_k must be a nonnegative integer")

    def with_overrides(self, *, temperature=None, top_p=None, top_k=None):
        """Create an independent policy, inheriting fields whose override is None."""
        return replace(
            self,
            **{
                name: value
                for name, value in (
                    ("temperature", temperature),
                    ("top_p", top_p),
                    ("top_k", top_k),
                )
                if value is not None
            },
        )


@dataclass(frozen=True)
class DraftPrediction:
    """Selected draft tokens, sigmoid confidence, and optional proposal distributions."""

    tokens: mx.array
    confidence: mx.array | None
    proposal_probs: mx.array | None = None


def probabilities(logits, options):
    if options.temperature <= 0:
        raise ValueError("Probability sampling requires temperature > 0")
    scores = logits.astype(mx.float32) / options.temperature
    if 0 < options.top_k < scores.shape[-1]:
        cutoff = mx.min(mx.topk(scores, options.top_k, axis=-1), axis=-1, keepdims=True)
        scores = mx.where(scores >= cutoff, scores, -float("inf"))
    probs = mx.softmax(scores, axis=-1)
    if options.top_p < 1:
        order = mx.argsort(probs, axis=-1)[..., ::-1]
        ranked = mx.take_along_axis(probs, order, axis=-1)
        keep = mx.cumsum(ranked, axis=-1) - ranked < options.top_p
        ranked = mx.where(keep, ranked, 0)
        # ``order`` is a permutation, so scatter restores vocabulary order
        # directly. Sorting the permutation again adds a second full-vocabulary
        # sort without changing the resulting distribution.
        probs = mx.put_along_axis(mx.zeros_like(probs), order, ranked, axis=-1)
        probs = probs / mx.sum(probs, axis=-1, keepdims=True)
    return probs


def sample_probabilities(probs):
    return mx.random.categorical(mx.log(probs)).astype(mx.int32)


def sample_logits(logits, options):
    if options.temperature == 0:
        return mx.argmax(logits, axis=-1).astype(mx.int32)
    return sample_probabilities(probabilities(logits, options))


def sample_draft(
    drafter,
    block_hidden: mx.array,
    anchor_ids: mx.array,
    options: SamplingOptions,
    count: int | None = None,
    *,
    require_proposal_probs: bool = False,
    require_confidence: bool = True,
    compile_chain: bool = False,
) -> DraftPrediction:
    """Select a draft chain, conditioning Markov/confidence on actual prior tokens.

    A zero temperature selects argmax tokens. Set require_proposal_probs to retain
    its one-hot proposal distribution when the target uses probability sampling.
    Positive temperatures retain the filtered proposal distributions for rejection
    correction. An omitted count uses all trained prediction slots. Set
    require_confidence=False when no confidence scheduling is requested;
    token selection and proposal probabilities are unaffected. compile_chain
    opts eligible vanilla Markov sampling into a pure device graph with explicit
    weights and RNG state; unsupported/custom drafters retain ordinary calls.
    """
    if drafter.config.draft_vocab_size != drafter.config.vocab_size:
        raise ValueError(
            "Draft sampling needs a full vocabulary or a draft-to-target map"
        )
    hidden = drafter.prediction_hidden(block_hidden)
    if count is None:
        count = hidden.shape[1]
    if type(count) is not int or not 1 <= count <= hidden.shape[1]:
        raise ValueError("count is outside the draft prediction slots")
    if anchor_ids.shape != (hidden.shape[0],) or not mx.issubdtype(
        anchor_ids.dtype, mx.integer
    ):
        raise ValueError(
            f"Token IDs must be an integer array with shape {(hidden.shape[0],)}"
        )
    hidden = hidden[:, :count]
    base = drafter.compute_logits(hidden)
    if compile_chain and not require_confidence and options.temperature > 0:
        from mlx import nn

        from .draft_quantization import DerivedDraftPerChannelW8Linear
        from .drafter import DSparkDrafter, VanillaMarkovHead

        markov = getattr(drafter, "markov_head", None)
        runtime_cache = getattr(markov, "_runtime_markov_cache", None)
        if (
            type(drafter) is DSparkDrafter
            and type(markov) is VanillaMarkovHead
            and runtime_cache is not None
            and markov.step_bias is runtime_cache._wrapper
            and markov.markov_w1 is runtime_cache._embedding_module
            and markov.markov_w2 is runtime_cache._projection_module
            and markov.markov_w1.weight is runtime_cache.embedding
            and markov.markov_w2.weight is runtime_cache.projection
            and "bias" not in markov.markov_w2
            and getattr(markov.prev_embeddings, "__func__", None)
            is VanillaMarkovHead.prev_embeddings
            and getattr(drafter.add_markov_bias, "__func__", None)
            is DSparkDrafter.add_markov_bias
        ):
            tokens, proposals = _compiled_cached_markov_chain(options)(
                base,
                anchor_ids,
                runtime_cache.embedding,
                runtime_cache.projection,
                runtime_cache.cache,
                runtime_cache.lookup,
            )
            return DraftPrediction(
                tokens=tokens, confidence=None, proposal_probs=proposals
            )
        if (
            type(drafter) is DSparkDrafter
            and type(markov) is VanillaMarkovHead
            and type(markov.markov_w1) is nn.Embedding
            and type(markov.markov_w2) is DerivedDraftPerChannelW8Linear
            and "bias" not in markov.markov_w2
            and markov.markov_w1.weight.dtype == mx.bfloat16
            and markov.markov_w1.weight.shape[1] % 32 == 0
            and hidden.shape[0] == 1
            and mx.default_device() == mx.gpu
            and mx.metal.is_available()
            and getattr(markov.step_bias, "__func__", None)
            is VanillaMarkovHead.step_bias
            and getattr(markov.prev_embeddings, "__func__", None)
            is VanillaMarkovHead.prev_embeddings
            and getattr(drafter.add_markov_bias, "__func__", None)
            is DSparkDrafter.add_markov_bias
        ):
            tokens, proposals = _compiled_per_channel_markov_chain(options)(
                base,
                anchor_ids,
                markov.markov_w1.weight,
                markov.markov_w2.weight,
                markov.markov_w2.scales,
            )
            return DraftPrediction(
                tokens=tokens, confidence=None, proposal_probs=proposals
            )
        if (
            type(drafter) is DSparkDrafter
            and type(markov) is VanillaMarkovHead
            and type(markov.markov_w1) is nn.Embedding
            and type(markov.markov_w2) is nn.Linear
            and "bias" not in markov.markov_w2
            and getattr(markov.step_bias, "__func__", None)
            is VanillaMarkovHead.step_bias
            and getattr(markov.prev_embeddings, "__func__", None)
            is VanillaMarkovHead.prev_embeddings
            and getattr(drafter.add_markov_bias, "__func__", None)
            is DSparkDrafter.add_markov_bias
        ):
            tokens, proposals = _compiled_markov_chain(options)(
                base,
                anchor_ids,
                markov.markov_w1.weight,
                markov.markov_w2.weight,
            )
            return DraftPrediction(
                tokens=tokens, confidence=None, proposal_probs=proposals
            )
    previous = anchor_ids
    tokens, proposals, confidences = [], [], []
    for i in range(count):
        logits = drafter.add_markov_bias(base[:, i], previous)
        if require_confidence and drafter.confidence_head is not None:
            confidences.append(
                mx.sigmoid(drafter.confidence_logits(hidden[:, i], previous))
            )
        if options.temperature == 0:
            previous = mx.argmax(logits, axis=-1).astype(mx.int32)
            if require_proposal_probs:
                proposals.append(
                    (mx.arange(logits.shape[-1]) == previous[..., None]).astype(
                        mx.float32
                    )
                )
        else:
            proposal = probabilities(logits, options)
            previous = sample_probabilities(proposal)
            proposals.append(proposal)
        tokens.append(previous)
    return DraftPrediction(
        tokens=mx.stack(tokens, axis=1),
        confidence=mx.stack(confidences, axis=1) if confidences else None,
        proposal_probs=mx.stack(proposals, axis=1) if proposals else None,
    )


@cache
def _compiled_markov_chain(options):
    """Vanilla conditional Markov proposals with explicit weights and RNG state."""

    @partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
    def chain(base, anchors, markov_embeddings, markov_projection):
        previous = anchors
        tokens, proposals = [], []
        for i in range(base.shape[1]):
            bias = mx.take(markov_embeddings, previous, axis=0) @ markov_projection.T
            proposal = probabilities(base[:, i] + bias, options)
            previous = sample_probabilities(proposal)
            tokens.append(previous)
            proposals.append(proposal)
        return mx.stack(tokens, axis=1), mx.stack(proposals, axis=1)

    return chain


@cache
def _compiled_cached_markov_chain(options):
    """Conditional cached-bias sampling with explicit derived tensors and RNG."""
    import spark_mlx_ext as ext

    @partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
    def chain(base, anchors, embedding, projection, bias_cache, lookup):
        previous = anchors
        tokens, proposals = [], []
        for i in range(base.shape[1]):
            bias = ext.cached_markov_bias(
                previous.astype(mx.int32), embedding, projection, bias_cache, lookup
            )
            proposal = probabilities(base[:, i] + bias, options)
            previous = sample_probabilities(proposal)
            tokens.append(previous)
            proposals.append(proposal)
        return mx.stack(tokens, axis=1), mx.stack(proposals, axis=1)

    return chain


@cache
def _compiled_per_channel_markov_chain(options):
    """Unchanged per-channel W8 M1 proposals with explicit tensors and RNG."""
    import spark_mlx_ext as ext

    @partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
    def chain(base, anchors, embedding, projection, scales):
        previous = anchors
        tokens, proposals = [], []
        for i in range(base.shape[1]):
            previous_embedding = mx.take(embedding, previous, axis=0)
            bias = ext.w8a16_per_channel(previous_embedding, projection, scales)
            proposal = probabilities(base[:, i] + bias, options)
            previous = sample_probabilities(proposal)
            tokens.append(previous)
            proposals.append(proposal)
        return mx.stack(tokens, axis=1), mx.stack(proposals, axis=1)

    return chain
