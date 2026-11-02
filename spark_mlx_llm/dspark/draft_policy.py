"""Choose the draft prefix to verify using confidence and a token budget.

This policy runs before target verification and does not decide actual acceptance.
It uses only the standard library.
"""

import math
from collections.abc import Iterable


def _probability(value: float, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError(f"{name} must be a finite probability in [0, 1]")
    return float(value)


def prefix_survival_probabilities(confidences: Iterable[float]) -> tuple[float, ...]:
    """Convert conditional per-token acceptance probabilities to prefix survival."""
    survival = 1.0
    result = []
    for confidence in confidences:
        survival *= _probability(confidence, "confidence")
        result.append(survival)
    return tuple(result)


def verification_length(
    confidences: Iterable[float],
    threshold: float,
    *,
    max_draft_tokens: int | None = None,
) -> int:
    """Admit a prefix until its survival falls below threshold, allowing zero.

    This is a single-request threshold policy, not the paper's global cost scheduler.
    The caller must supply probabilities, not confidence-head logits.
    """
    threshold = _probability(threshold, "threshold")
    if max_draft_tokens is not None and (
        type(max_draft_tokens) is not int or max_draft_tokens < 0
    ):
        raise ValueError("max_draft_tokens must be a nonnegative integer")
    length = 0
    survival = 1.0
    confidences = iter(confidences)
    while max_draft_tokens is None or length < max_draft_tokens:
        try:
            confidence = next(confidences)
        except StopIteration:
            break
        survival *= _probability(confidence, "confidence")
        if survival < threshold:
            break
        length += 1
    return length
