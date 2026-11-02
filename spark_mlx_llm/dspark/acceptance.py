"""Accept verified draft prefixes and select correction or bonus tokens.

Greedy acceptance uses only the standard library.
Probability-based acceptance loads MLX sampling utilities when called.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache, partial


@dataclass(frozen=True)
class AcceptanceResult:
    tokens: tuple[int, ...]
    num_drafted: int
    num_accepted: int

    @property
    def num_rejected(self) -> int:
        return self.num_drafted - self.num_accepted


def greedy_accept(
    draft_tokens: Sequence[int], target_tokens: Sequence[int]
) -> AcceptanceResult:
    """Accept a matching prefix and emit the target correction or bonus token.

    Target predictions come from one forward over [anchor, *draft_tokens].
    This rule is for greedy decoding; sampled decoding needs rejection sampling.
    """
    draft, target = tuple(draft_tokens), tuple(target_tokens)
    if len(target) != len(draft) + 1:
        raise ValueError("Need one target prediction per draft token plus a bonus row")
    if any(type(token) is not int or token < 0 for token in (*draft, *target)):
        raise ValueError("Token IDs must be nonnegative integers")
    accepted = 0
    while accepted < len(draft) and draft[accepted] == target[accepted]:
        accepted += 1
    return AcceptanceResult(target[: accepted + 1], len(draft), accepted)


def sampled_accept(
    draft_tokens, draft_probs, target_logits, options, *, compact_target=False
) -> AcceptanceResult:
    """Accept a sampled prefix and emit a residual correction or target bonus.

    options is the target policy. draft_probs must describe the actual draft
    policy (including filtering, or a one-hot distribution for greedy proposals).
    The two policies need not share temperature, top-p, or top-k.
    compact_target opts nonempty verification into keeping a boolean nucleus
    mask and materializing only the correction row's normalized target vector.
    It preserves the ordinary sorting/tie policy and explicit random state.
    """
    import mlx.core as mx

    if isinstance(draft_tokens, mx.array):
        if draft_tokens.ndim != 1 or not mx.issubdtype(draft_tokens.dtype, mx.integer):
            raise ValueError("Draft token IDs must be a one-dimensional integer array")
        ids = draft_tokens.astype(mx.int32)
        count = ids.shape[0]
    else:
        count = len(draft_tokens)
        ids = mx.array(draft_tokens, dtype=mx.int32)
    if target_logits.ndim != 2 or target_logits.shape[0] != count + 1:
        raise ValueError(
            "Need one target distribution per draft token and one bonus row"
        )
    if count and (
        draft_probs is None or draft_probs.shape != (count, target_logits.shape[1])
    ):
        raise ValueError("Draft proposal shape differs from the verified block")
    if not count:
        # Give the compiled function a consistently typed array instead of None.
        draft_probs = mx.zeros((0, target_logits.shape[1]), dtype=mx.float32)
    # Prefix selection, residual construction and correction sampling stay on
    # device. Reading this one packet is the only host synchronization. Include
    # the original draft IDs so callers need not materialize them beforehand.
    packet = _sampled_accept_kernel(options, bool(compact_target))(
        ids, draft_probs, target_logits
    ).tolist()
    prefix, correction, invalid_q, invalid_mass = packet[:4]
    if invalid_q:
        raise ValueError(
            "A sampled draft token must have positive proposal probability"
        )
    if invalid_mass:
        raise RuntimeError("Rejected proposal has no residual target mass")
    return AcceptanceResult(
        tuple(packet[4 : 4 + prefix]) + (correction,), count, prefix
    )


def _compact_target_components(target_logits, options):
    """Keep target softmax/mask/mass without a dense normalized block.

    Sorting and cumulative thresholding match sampling.probabilities exactly.
    The boolean scatter restores vocabulary order before the reduction, so
    normalization retains the original vocabulary-order summation contract.
    """
    import mlx.core as mx

    if options.temperature <= 0:
        raise ValueError("Probability sampling requires temperature > 0")
    scores = target_logits.astype(mx.float32) / options.temperature
    if 0 < options.top_k < scores.shape[-1]:
        cutoff = mx.min(mx.topk(scores, options.top_k, axis=-1), axis=-1, keepdims=True)
        scores = mx.where(scores >= cutoff, scores, -float("inf"))
    raw = mx.softmax(scores, axis=-1)
    if options.top_p == 1:
        return raw, None, None
    order = mx.argsort(raw, axis=-1)[..., ::-1]
    ranked = mx.take_along_axis(raw, order, axis=-1)
    keep = mx.cumsum(ranked, axis=-1) - ranked < options.top_p
    mask = mx.put_along_axis(mx.zeros(raw.shape, dtype=mx.bool_), order, keep, axis=-1)
    mass = mx.sum(mx.where(mask, raw, 0), axis=-1)
    return raw, mask, mass


@cache
def _sampled_accept_kernel(options, compact_target=False):
    """Cache the device graph by sampling policy, including explicit RNG state.

    Imports remain lazy so greedy_accept stays usable without MLX. Errors are
    returned as flags and raised by sampled_accept after the single packet read.
    """
    import mlx.core as mx

    from .sampling import probabilities, sample_probabilities

    @partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
    def kernel(ids, proposal, target_logits):
        count = ids.shape[0]
        compact = compact_target and count > 0
        if compact:
            raw, mask, target_mass = _compact_target_components(target_logits, options)
        else:
            target = probabilities(target_logits, options)
        if count:
            rows = mx.arange(count)
            q = proposal[rows, ids]
            invalid_q = mx.any(q <= 0)
            safe_q = mx.where(q > 0, q, 1)
            if compact:
                p = raw[rows, ids]
                if mask is not None:
                    p = mx.where(mask[rows, ids], p, 0) / target_mass[:count]
            else:
                p = target[rows, ids]
            accepted = mx.random.uniform(shape=(count,)) < mx.minimum(1, p / safe_q)
            prefix = mx.sum(mx.cumprod(accepted.astype(mx.int32)))
            if compact:
                selected_target = mx.take(raw, prefix, axis=0)
                if mask is not None:
                    selected_target = mx.where(
                        mx.take(mask, prefix, axis=0), selected_target, 0
                    ) / mx.take(target_mass, prefix, axis=0)
            else:
                selected_target = mx.take(target, prefix, axis=0)
            selected_proposal = mx.take(proposal, mx.minimum(prefix, count - 1), axis=0)
            residual = mx.maximum(selected_target - selected_proposal, 0)
            mass = mx.sum(residual)
            rejected = prefix < count
            invalid_mass = rejected & (mass <= 0)
            # The fallback only protects malformed inputs from undefined
            # categorical(-inf, ...); invalid_mass is still raised on the host.
            normalized = residual / mx.where(mass > 0, mass, 1)
            correction_probs = mx.where(
                rejected & (mass > 0), normalized, selected_target
            )
        else:
            prefix = mx.array(0, dtype=mx.int32)
            invalid_q = invalid_mass = mx.array(False)
            correction_probs = target[0]
        correction = sample_probabilities(correction_probs)
        header = mx.stack([prefix, correction, invalid_q, invalid_mass]).astype(
            mx.int32
        )
        return mx.concatenate([header, ids])

    return kernel
