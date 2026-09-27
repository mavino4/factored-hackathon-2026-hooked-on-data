"""Per-user request rate limit, in process. With N replicas the effective limit is N x the setting."""

import time


class RateLimiter:
    """Token bucket: ``per_minute`` requests with bursts up to the same size."""

    def __init__(self, per_minute: int, clock=time.monotonic, prune_every: int = 1000):
        self.capacity = float(per_minute)
        self.refill_per_s = per_minute / 60.0
        self._clock = clock
        self._buckets: dict[str, tuple[float, float]] = {}
        self._prune_every = prune_every
        self._calls = 0

    def allow(self, key: str) -> bool:
        now = self._clock()
        self._calls += 1
        if self._calls % self._prune_every == 0:
            self._prune(now)
        tokens, last = self._buckets.get(key, (self.capacity, now))
        tokens = min(self.capacity, tokens + (now - last) * self.refill_per_s)
        if tokens < 1:
            self._buckets[key] = (tokens, now)
            return False
        self._buckets[key] = (tokens - 1, now)
        return True

    def _prune(self, now: float) -> None:
        # A bucket idle long enough to be full again is identical to a missing one.
        full_after = self.capacity / self.refill_per_s
        self._buckets = {k: v for k, v in self._buckets.items() if now - v[1] < full_after}

    def __len__(self) -> int:
        return len(self._buckets)
