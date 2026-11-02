"""KV storage with an explicit commit boundary for speculative forwards."""

import mlx.core as mx


def cache_arrays(caches):
    """Collect materializable K/V arrays from a sequence of layer caches."""
    return [array for cache in caches for array in cache.state if array is not None]


class TransactionalKVCache:
    """Keep all rows from the current forward until its accepted length is known."""

    def __init__(self, window_size: int | None = None):
        if window_size is not None and (
            type(window_size) is not int or window_size <= 0
        ):
            raise ValueError("window_size must be a positive integer")
        self.window_size = window_size
        self.offset = 0
        self.pending = 0
        self._keys = None
        self._values = None
        self._length = 0

    @property
    def start(self):
        return self.offset - self._length

    @property
    def keys(self):
        return None if self._keys is None else self._keys[..., : self._length, :]

    @property
    def values(self):
        return None if self._values is None else self._values[..., : self._length, :]

    @property
    def state(self):
        return self.keys, self.values

    @property
    def nbytes(self):
        if self._keys is None:
            return 0
        return self._keys.nbytes + self._values.nbytes

    def size(self):
        return self._length

    def update_and_fetch(self, keys, values):
        if self.pending:
            raise RuntimeError(
                "Finish the pending KV transaction before another forward"
            )
        count = keys.shape[2]
        if count == 0 or values.shape[:3] != keys.shape[:3]:
            raise ValueError("KV updates need matching, nonempty sequence dimensions")
        if self._keys is not None and (
            keys.shape[:2] != self._keys.shape[:2]
            or keys.shape[3] != self._keys.shape[3]
            or values.shape[3] != self._values.shape[3]
        ):
            raise ValueError("KV update geometry changed")
        if self.window_size is not None:
            self._keys = (
                keys
                if self._keys is None
                else mx.concatenate([self.keys, keys], axis=2)
            )
            self._values = (
                values
                if self._values is None
                else mx.concatenate([self.values, values], axis=2)
            )
        else:
            needed = self._length + count
            if self._keys is None or needed > self._keys.shape[2]:
                capacity = ((needed + 255) // 256) * 256
                next_keys = mx.zeros(
                    (*keys.shape[:2], capacity, keys.shape[3]), dtype=keys.dtype
                )
                next_values = mx.zeros(
                    (*values.shape[:2], capacity, values.shape[3]), dtype=values.dtype
                )
                if self._length:
                    next_keys[..., : self._length, :] = self.keys
                    next_values[..., : self._length, :] = self.values
                self._keys, self._values = next_keys, next_values
            self._keys[..., self._length : needed, :] = keys
            self._values[..., self._length : needed, :] = values
        self._length += count
        self.offset += count
        self.pending = count
        if self.window_size is not None:
            # Keep rollback history in storage; expose only rows the first query can use.
            first = max(0, self._length - count - self.window_size + 1)
            return self.keys[..., first:, :], self.values[..., first:, :]
        return self.keys, self.values

    def finish(self, accepted: int | None = None):
        if accepted is None:
            accepted = self.pending
        if type(accepted) is not int or not 0 <= accepted <= self.pending:
            raise ValueError("Accepted rows must be inside the pending KV update")
        rejected = self.pending - accepted
        self._length -= rejected
        self.offset -= rejected
        self.pending = 0
        if self.window_size is not None and self._keys is not None:
            keep = min(self.window_size, self._length)
            first = self._length - keep
            self._keys = self._keys[..., first : self._length, :]
            self._values = self._values[..., first : self._length, :]
            self._length = keep

    def make_mask(self, count, *, window_size=None, return_array=False):
        if self.pending:
            raise RuntimeError("Cannot build a mask for an unfinished KV transaction")
        window = window_size if window_size is not None else self.window_size
        if window is None and not return_array:
            return None if count == 1 else "causal"
        first = self.start
        if self.window_size is not None:
            first = max(first, self.offset - self.window_size + 1)
        if not return_array and (
            count == 1 or (window is not None and self.offset + count - first <= window)
        ):
            return None if count == 1 else "causal"
        queries = self.offset + mx.arange(count)[:, None]
        keys = first + mx.arange(self.offset + count - first)[None, :]
        allowed = keys <= queries
        if window is not None:
            allowed = allowed & (keys > queries - window)
        return allowed


class TargetState:
    def __init__(self, windows):
        self.layers = [TransactionalKVCache(window) for window in windows]

    @property
    def offset(self):
        offsets = {cache.offset for cache in self.layers}
        if len(offsets) != 1:
            raise RuntimeError("Target layer cache offsets diverged")
        return self.layers[0].offset

    def finish(self, accepted=None):
        if len({cache.pending for cache in self.layers}) != 1:
            raise RuntimeError("Target layer transactions diverged")
        for cache in self.layers:
            cache.finish(accepted)

    def abort(self):
        for cache in self.layers:
            if cache.pending:
                cache.finish(0)

    def arrays(self):
        return cache_arrays(self.layers)


class DraftKVCache(TransactionalKVCache):
    """Append only committed target features, already projected and rotated."""

    def append(self, keys, values):
        self.update_and_fetch(keys, values)
        self.finish()
