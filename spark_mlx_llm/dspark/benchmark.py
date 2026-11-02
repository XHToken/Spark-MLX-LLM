"""Paired baseline/speculative measurements with greedy output checks."""

import statistics

from .generation import generate


def benchmark(pair, prompt, *, trials=3, warmup_tokens=8, **options):
    if type(trials) is not int or trials <= 0:
        raise ValueError("trials must be a positive integer")
    if type(warmup_tokens) is not int or warmup_tokens < 0:
        raise ValueError("warmup_tokens must be a nonnegative integer")
    options = dict(options)
    options.pop("mode", None)
    if warmup_tokens:
        warmup = {**options, "max_new_tokens": warmup_tokens}
        for mode in ("baseline", "dspark"):
            generate(pair, prompt, mode=mode, **warmup)
    baseline, speculative = [], []
    for trial in range(trials):
        modes = ("baseline", "dspark") if trial % 2 == 0 else ("dspark", "baseline")
        results = {mode: generate(pair, prompt, mode=mode, **options) for mode in modes}
        baseline.append(results["baseline"])
        speculative.append(results["dspark"])
    sampled = (
        options.get("sampling") is not None and options["sampling"].temperature > 0
    )
    equal = (
        None
        if sampled
        else all(a.token_ids == b.token_ids for a, b in zip(baseline, speculative))
    )
    base_rate = statistics.median(r.tokens_per_second for r in baseline)
    spec_rate = statistics.median(r.tokens_per_second for r in speculative)
    ratio = spec_rate / base_rate if base_rate else None
    return {
        "trials": trials,
        "greedy_verification": options.get("greedy_verification", "batched"),
        "greedy_match": equal,
        "baseline_tokens_per_second": base_rate,
        "dspark_tokens_per_second": spec_rate,
        "speedup": ratio if equal is not False else None,
        "measured_speed_ratio": ratio,
        "baseline": [r.to_dict() for r in baseline],
        "dspark": [r.to_dict() for r in speculative],
    }
