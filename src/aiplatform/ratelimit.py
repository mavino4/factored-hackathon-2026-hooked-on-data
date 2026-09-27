"""Per-user limits, in process. Good for a single replica; move to shared storage when scaling out."""

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


class DailyTokenQuota:
    def __init__(self, tokens_per_day: int, clock=time.time):
        self.limit = tokens_per_day
        self._clock = clock
        self._day = self._today()
        self._used: dict[str, int] = {}

    def _today(self) -> int:
        return int(self._clock() // 86_400)

    def _roll(self) -> None:
        if (today := self._today()) != self._day:  # new UTC day: forget yesterday
            self._day = today
            self._used = {}

    def exceeded(self, user_id: str) -> bool:
        self._roll()
        return self._used.get(user_id, 0) >= self.limit

    def charge(self, user_id: str, tokens: int) -> None:
        self._roll()
        self._used[user_id] = self._used.get(user_id, 0) + tokens
