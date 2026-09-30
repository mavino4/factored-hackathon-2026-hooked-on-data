"""Human handoffs: a conversation handed from the bot to a human advisor.

Lifecycle:
    offered ──accept──> open ──close──> closed      (the bot answers again)
       └────decline──> declined
    (customer asks for a person) ─────> open

While a handoff is open the bot does not answer; operators reply through the admin
API. Status changes are atomic (a compare-and-set on the current status), so two
replicas can't both accept or close the same handoff.
"""

import uuid
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Literal, Protocol

Status = Literal["offered", "open", "declined", "closed"]
Reason = Literal["customer_request", "insistence"]


class HandoffNotFound(Exception):
    """Unknown handoff, someone else's, or not in the expected status."""


@dataclass(frozen=True)
class Handoff:
    id: str
    conversation_id: str
    user_id: str
    status: Status
    reason: Reason
    summary: str
    # Position in the conversation when the handoff was created (to count turns since).
    message_index: int
    created_at: datetime
    closed_at: datetime | None = None


class HandoffStore(Protocol):
    async def create(self, *, conversation_id: str, user_id: str, status: Status,
                     reason: Reason, summary: str, message_index: int) -> Handoff: ...

    async def latest(self, conversation_id: str) -> Handoff | None:
        """The conversation's most recent handoff, if any."""
        ...

    async def get(self, handoff_id: str) -> Handoff: ...

    async def list(self, status: Status | None = None, limit: int = 100) -> list[Handoff]: ...

    async def transition(self, handoff_id: str, *, expected: Status, to: Status,
                         conversation_id: str | None = None,
                         user_id: str | None = None) -> Handoff:
        """Move a handoff from ``expected`` to ``to`` exactly once, optionally checking its
        owner. Raises HandoffNotFound otherwise."""
        ...


class InMemoryHandoffStore:
    def __init__(self) -> None:
        self._items: dict[str, Handoff] = {}

    async def create(self, *, conversation_id: str, user_id: str, status: Status,
                     reason: Reason, summary: str, message_index: int) -> Handoff:
        handoff = Handoff(id=str(uuid.uuid4()), conversation_id=conversation_id,
                          user_id=user_id, status=status, reason=reason, summary=summary,
                          message_index=message_index, created_at=datetime.now(UTC))
        self._items[handoff.id] = handoff
        return handoff

    async def latest(self, conversation_id: str) -> Handoff | None:
        mine = [h for h in self._items.values() if h.conversation_id == conversation_id]
        return max(mine, key=lambda h: h.created_at, default=None)

    async def get(self, handoff_id: str) -> Handoff:
        if handoff_id not in self._items:
            raise HandoffNotFound(handoff_id)
        return self._items[handoff_id]

    async def list(self, status: Status | None = None, limit: int = 100) -> list[Handoff]:
        items = [h for h in self._items.values() if status is None or h.status == status]
        return sorted(items, key=lambda h: h.created_at)[:limit]

    async def transition(self, handoff_id: str, *, expected: Status, to: Status,
                         conversation_id: str | None = None,
                         user_id: str | None = None) -> Handoff:
        h = self._items.get(handoff_id)
        if (h is None or h.status != expected
                or (conversation_id is not None and h.conversation_id != conversation_id)
                or (user_id is not None and h.user_id != user_id)):
            raise HandoffNotFound(handoff_id)
        closed = datetime.now(UTC) if to in ("closed", "declined") else None
        self._items[handoff_id] = replace(h, status=to, closed_at=closed)
        return self._items[handoff_id]
