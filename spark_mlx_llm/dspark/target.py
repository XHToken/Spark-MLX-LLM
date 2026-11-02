"""Run the existing Spark target and expose logits and selected layer features."""

from dataclasses import dataclass

import mlx.core as mx
from mlx_lm.models.base import create_attention_mask

from .cache import TargetState


@dataclass(frozen=True)
class TargetOutput:
    logits: mx.array
    taps: tuple[mx.array, ...]


class SparkTarget:
    """Target forward adapter; the generation loop controls acceptance and commits."""

    def __init__(self, model, tap_ids):
        self.model = model
        self.tap_ids = tuple(tap_ids)
        if len(set(self.tap_ids)) != len(self.tap_ids) or any(
            type(i) is not int or not 0 <= i < len(model.layers) for i in self.tap_ids
        ):
            raise ValueError("Invalid target layer tap IDs")

    def make_state(self):
        return TargetState(
            self.model.args.sliding_window if layer.is_sliding else None
            for layer in self.model.layers
        )

    def project(self, hidden):
        if self.model.args.tie_word_embeddings:
            return self.model.model.embedding.as_linear(hidden)
        return self.model.lm_head(hidden)

    def run(self, tokens, state, *, capture=True, last_only=False):
        """Forward one transaction. The caller commits only its accepted input rows."""
        if tokens.ndim != 2 or tokens.shape[0] != 1 or tokens.shape[1] == 0:
            raise ValueError("Target tokens must have shape [1, nonempty_length]")
        if len(state.layers) != len(self.model.layers):
            raise ValueError("Target cache layer count differs from the model")
        if any(cache.pending for cache in state.layers):
            raise RuntimeError("Finish target verification before another forward")
        _ = state.offset
        inner = self.model.model
        hidden = inner.embedding(tokens)
        full_mask = sliding_mask = None
        for layer, cache in zip(inner.layers, state.layers):
            if layer.is_sliding:
                if sliding_mask is None:
                    sliding_mask = create_attention_mask(
                        hidden, cache, window_size=self.model.args.sliding_window
                    )
            elif full_mask is None:
                full_mask = create_attention_mask(hidden, cache)
        captured = {}
        wanted = set(self.tap_ids) if capture else set()
        try:
            for i, (layer, cache) in enumerate(zip(inner.layers, state.layers)):
                hidden = layer(
                    hidden, sliding_mask if layer.is_sliding else full_mask, cache
                )
                if i in wanted:
                    captured[i] = hidden
            normalized = inner.norm(hidden)
            logits = self.project(normalized[:, -1:] if last_only else normalized)
        except Exception:
            state.abort()
            raise
        return TargetOutput(
            logits, tuple(captured[i] for i in self.tap_ids) if capture else ()
        )
