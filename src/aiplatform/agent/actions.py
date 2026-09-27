"""Pending actions: irreversible tool calls waiting for the user's approval."""

import uuid
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

Status = Literal["pending", "approved", "rejected"]


class ActionNotPending(Exception):
    """Unknown action, someone else's, or already decided."""


@dataclass(frozen=True)
class PendingAction:
    id: str
    conversation_id: str
    user_id: str
    tool_use_id: str
    tool_name: str
    input: dict[str, Any]
    status: Status
    created_at: datetime
    decided_at: datetime | None = None


class ActionStore(Protocol):
    async def create(self, *, conversation_id: str, user_id: str, tool_use_id: str,
                     tool_name: str, input: dict[str, Any]) -> PendingAction: ...

    async def list_pending(self, conversation_id: str, user_id: str) -> list[PendingAction]: ...

    async def decide(self, action_id: str, *, user_id: str, conversation_id: str,
                     approve: bool) -> PendingAction:
        """Atomically move a pending action to approved/rejected, exactly once.
        Raises ActionNotPending otherwise, so a tool can never run twice."""
        ...


class InMemoryActionStore:
    def __init__(self) -> None:
        self._items: dict[str, PendingAction] = {}

    async def create(self, *, conversation_id: str, user_id: str, tool_use_id: str,
                     tool_name: str, input: dict[str, Any]) -> PendingAction:
        action = PendingAction(id=str(uuid.uuid4()), conversation_id=conversation_id,
                               user_id=user_id, tool_use_id=tool_use_id, tool_name=tool_name,
                               input=input, status="pending", created_at=datetime.now(UTC))
        self._items[action.id] = action
        return action

    async def list_pending(self, conversation_id: str, user_id: str) -> list[PendingAction]:
        return [a for a in self._items.values()
                if a.conversation_id == conversation_id and a.user_id == user_id
                and a.status == "pending"]

    async def decide(self, action_id: str, *, user_id: str, conversation_id: str,
                     approve: bool) -> PendingAction:
        action = self._items.get(action_id)
        if (action is None or action.user_id != user_id
                or action.conversation_id != conversation_id or action.status != "pending"):
            raise ActionNotPending(action_id)
        decided = replace(action, status="approved" if approve else "rejected",
                          decided_at=datetime.now(UTC))
        self._items[action_id] = decided
        return decided
