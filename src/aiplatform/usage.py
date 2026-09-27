"""Token usage records and the per-user daily token quota built on them."""

import time
from dataclasses import dataclass
from typing import Protocol

from anthropic.types import Message


@dataclass(frozen=True)
class UsageEvent:
    user_id: str
    conversation_id: str | None
    route: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int

    @property
    def total_tokens(self) -> int:
        return (self.input_tokens + self.output_tokens
                + self.cache_read_tokens + self.cache_write_tokens)

    @classmethod
    def from_message(cls, message: Message, *, user_id: str, conversation_id: str | None,
                     route: str, provider: str) -> "UsageEvent":
        u = message.usage
        return cls(user_id=user_id, conversation_id=conversation_id, route=route,
                   provider=provider, model=message.model,
                   input_tokens=u.input_tokens, output_tokens=u.output_tokens,
                   cache_read_tokens=u.cache_read_input_tokens or 0,
                   cache_write_tokens=u.cache_creation_input_tokens or 0)


class UsageStore(Protocol):
    async def record(self, event: UsageEvent) -> None: ...
    async def tokens_used_today(self, user_id: str) -> int: ...

    async def daily_summary(self, days: int) -> list[dict] | None:
        """Per day/route/provider/model totals, or None if this store keeps no history."""
        ...


def utc_day_start(now: float) -> float:
    return now - now % 86_400


class InMemoryUsageStore:
    """Per-user totals for the current UTC day only (bounded memory)."""

    def __init__(self, clock=time.time):
        self._clock = clock
        self._day = utc_day_start(clock())
        self._totals: dict[str, int] = {}

    def _roll(self) -> None:
        if (day := utc_day_start(self._clock())) != self._day:
            self._day = day
            self._totals = {}

    async def record(self, event: UsageEvent) -> None:
        self._roll()
        self._totals[event.user_id] = self._totals.get(event.user_id, 0) + event.total_tokens

    async def tokens_used_today(self, user_id: str) -> int:
        self._roll()
        return self._totals.get(user_id, 0)

    async def daily_summary(self, days: int) -> list[dict] | None:
        return None  # only today's per-user totals are kept in memory


class TokenQuota:
    def __init__(self, store: UsageStore, tokens_per_day: int):
        self.store = store
        self.limit = tokens_per_day

    async def exceeded(self, user_id: str) -> bool:
        return await self.store.tokens_used_today(user_id) >= self.limit
