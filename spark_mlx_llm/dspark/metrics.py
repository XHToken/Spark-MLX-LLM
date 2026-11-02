"""Generation results, metric definitions, and draft-length counters.

These data types use only the standard library and perform no model execution.
"""

from dataclasses import asdict, dataclass, field


@dataclass
class PhaseTimings:
    """Additive host-wall timings for requests that actually recorded them.

    Split phases use existing evaluation boundaries; they are not GPU kernel
    timings. Cache commits can enqueue work executed by a subsequent phase.
    Missing historical timings must remain missing rather than become zeros.
    """

    requests: int = 0
    output_tokens: int = 0
    committed_tokens: int = 0
    rounds: int = 0
    target_steps: int = 0
    generation_seconds: float = 0.0
    draft_seconds: float = 0.0
    target_forward_seconds: float = 0.0
    acceptance_seconds: float = 0.0
    target_commit_seconds: float = 0.0
    context_seconds: float = 0.0
    other_seconds: float = 0.0

    @classmethod
    def from_dict(cls, values):
        return cls(**{k: values[k] for k in cls.__dataclass_fields__})

    @classmethod
    def from_result(cls, result):
        # Sequential verification interleaves forward/accept/commit operations.
        if result.target_forward_seconds is None:
            return None
        parts = {
            name: getattr(result, name)
            for name in (
                "draft_seconds", "target_forward_seconds", "acceptance_seconds",
                "target_commit_seconds", "context_seconds",
            )
        }
        return cls(
            requests=1,
            output_tokens=len(result.token_ids),
            committed_tokens=max(0, len(result.token_ids) - 1),
            rounds=result.rounds,
            target_steps=result.target_steps,
            generation_seconds=result.generation_seconds,
            other_seconds=result.generation_seconds - sum(parts.values()),
            **parts,
        )

    def add(self, other):
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))

    def to_dict(self):
        values = asdict(self)
        phases = (
            "draft_seconds", "target_forward_seconds", "acceptance_seconds",
            "target_commit_seconds", "context_seconds", "other_seconds",
        )
        values["phase_percent"] = {
            name: 100 * getattr(self, name) / self.generation_seconds
            if self.generation_seconds else 0.0
            for name in phases
        }
        values["phase_ms_per_round"] = {
            name: 1000 * getattr(self, name) / self.rounds if self.rounds else 0.0
            for name in phases
        }
        values["ms_per_round"] = (
            1000 * self.generation_seconds / self.rounds if self.rounds else 0.0
        )
        values["ms_per_committed_token"] = (
            1000 * self.generation_seconds / self.committed_tokens
            if self.committed_tokens else 0.0
        )
        values["committed_tokens_per_round"] = (
            self.committed_tokens / self.rounds if self.rounds else 0.0
        )
        return values


@dataclass
class DraftLengthStats:
    rounds: int = 0
    drafted_tokens: int = 0
    accepted_tokens: int = 0

    @classmethod
    def from_dict(cls, values):
        return cls(**{name: values[name] for name in cls.__dataclass_fields__})

    def add(self, other):
        self.rounds += other.rounds
        self.drafted_tokens += other.drafted_tokens
        self.accepted_tokens += other.accepted_tokens

    @property
    def draft_acceptance_rate(self):
        return (
            self.accepted_tokens / self.drafted_tokens if self.drafted_tokens else 0.0
        )

    def to_dict(self):
        return {**asdict(self), "draft_acceptance_rate": self.draft_acceptance_rate}


@dataclass(frozen=True)
class GenerationResult:
    token_ids: tuple[int, ...]
    text: str
    prompt_tokens: int
    finish_reason: str
    prefill_seconds: float
    generation_seconds: float
    draft_seconds: float
    verify_seconds: float
    context_seconds: float
    target_steps: int
    drafted_tokens: int
    accepted_tokens: int
    peak_memory_bytes: int
    rounds: int = 0
    draft_length_stats: dict[int, DraftLengthStats] = field(default_factory=dict)
    target_forward_seconds: float | None = None
    acceptance_seconds: float | None = None
    target_commit_seconds: float = 0.0

    @property
    def tokens_per_second(self):
        return (
            len(self.token_ids) / self.generation_seconds
            if self.generation_seconds
            else 0.0
        )

    @property
    def draft_acceptance_rate(self):
        """Fraction of submitted draft tokens accepted, excluding target tokens."""
        return (
            self.accepted_tokens / self.drafted_tokens if self.drafted_tokens else 0.0
        )

    @property
    def mean_acceptance_length(self):
        """Actual tokens emitted per round, including target tokens and EOS."""
        return max(0, len(self.token_ids) - 1) / self.rounds if self.rounds else 0.0

    @property
    def mean_draft_length(self):
        """Submitted draft tokens per round, including rounds with no draft."""
        return self.drafted_tokens / self.rounds if self.rounds else 0.0

    def to_dict(self):
        return {
            **asdict(self),
            "tokens_per_second": self.tokens_per_second,
            "draft_acceptance_rate": self.draft_acceptance_rate,
            "mean_acceptance_length": self.mean_acceptance_length,
            "mean_draft_length": self.mean_draft_length,
            "draft_length_stats": {
                length: stats.to_dict()
                for length, stats in sorted(self.draft_length_stats.items())
            },
        }
