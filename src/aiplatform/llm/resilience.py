"""Error classification, backoff and a per-provider circuit breaker."""

import random
import time
from enum import Enum

import anthropic


class Disposition(Enum):
    RETRY = "retry"  # transient: retry on the same provider, then fail over
    FAILOVER = "failover"  # provider-specific (auth, missing model): go to the next provider
    FATAL = "fatal"  # the request itself is wrong: fails the same everywhere


def classify(exc: BaseException) -> Disposition:
    if isinstance(exc, anthropic.APIConnectionError):  # includes APITimeoutError
        return Disposition.RETRY
    if isinstance(exc, anthropic.APIStatusError):
        status = exc.status_code
        if status in (408, 409, 429) or status >= 500:  # 529 = overloaded
            return Disposition.RETRY
        if status in (401, 403, 404):
            return Disposition.FAILOVER
    return Disposition.FATAL


def retry_after_seconds(exc: BaseException) -> float | None:
    response = getattr(exc, "response", None)
    if response is None:
        return None
    try:
        return float(response.headers.get("retry-after", ""))
    except ValueError:
        return None


def backoff_delay(attempt: int, *, base: float = 0.5, cap: float = 20.0,
                  retry_after: float | None = None) -> float:
    """Exponential backoff with full jitter; never earlier than the server's retry-after."""
    delay = random.uniform(0, min(cap, base * 2**attempt))
    if retry_after is not None:
        delay = max(delay, min(retry_after, cap))
    return delay


class CircuitBreaker:
    """Opens after ``threshold`` consecutive failures. After ``cooldown_s`` it lets
    exactly one probe request through (half-open); its outcome closes or re-opens it."""

    def __init__(self, threshold: int = 5, cooldown_s: float = 30.0, clock=time.monotonic):
        self.threshold = threshold
        self.cooldown_s = cooldown_s
        self._clock = clock
        self._failures = 0
        self._opened_at: float | None = None
        self._probing = False

    def available(self) -> bool:
        """Could a request go through now? Does not claim the half-open probe."""
        if self._opened_at is None:
            return True
        return not self._probing and self._clock() - self._opened_at >= self.cooldown_s

    def acquire(self) -> bool:
        """Claim permission for one request; call ``release()`` when it finishes."""
        if self._opened_at is None:
            return True
        if not self.available():
            return False
        self._probing = True
        return True

    def release(self) -> None:
        self._probing = False

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self.threshold:
            self._opened_at = self._clock()
