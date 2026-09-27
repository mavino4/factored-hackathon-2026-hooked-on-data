"""Conversation storage. History is append-only: earlier turns are never edited,
which keeps prompt caches valid and is required by newer Claude models."""

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal, Protocol

Kind = Literal["chat", "agent"]

TITLE_CHARS = 80


class ConversationNotFound(Exception):
    pass


@dataclass
class Conversation:
    id: str
    user_id: str
    kind: Kind
    created_at: datetime
    updated_at: datetime
    title: str | None = None
    messages: list[dict] = field(default_factory=list)


def title_from(message: dict) -> str | None:
    content = message["content"]
    if message["role"] == "user" and isinstance(content, str):
        return content.strip()[:TITLE_CHARS] or None
    return None


class ConversationRepository(Protocol):
    async def create(self, user_id: str, kind: Kind) -> Conversation: ...

    async def get(self, conversation_id: str, user_id: str) -> Conversation:
        """Load a conversation with its messages. Raises ConversationNotFound
        if it doesn't exist or belongs to another user."""
        ...

    async def append(self, conv: Conversation, message: dict) -> None:
        """Append one message at position ``len(conv.messages)`` and add it to
        ``conv.messages``. Raises ConversationBusy if another writer got there first."""
        ...

    async def list(self, user_id: str, limit: int = 50) -> list[Conversation]:
        """The user's conversations, most recently updated first, without messages."""
        ...


class InMemoryConversationRepository:
    """Single-process store for tests and local development without a database."""

    def __init__(self) -> None:
        self._items: dict[str, Conversation] = {}

    async def create(self, user_id: str, kind: Kind) -> Conversation:
        now = datetime.now(UTC)
        conv = Conversation(id=str(uuid.uuid4()), user_id=user_id, kind=kind,
                            created_at=now, updated_at=now)
        self._items[conv.id] = conv
        return conv

    async def get(self, conversation_id: str, user_id: str) -> Conversation:
        conv = self._items.get(conversation_id)
        if conv is None or conv.user_id != user_id:  # never reveal other users' conversations
            raise ConversationNotFound(conversation_id)
        return conv

    async def append(self, conv: Conversation, message: dict) -> None:
        stored = self._items[conv.id]
        if conv.title is None:
            conv.title = title_from(message)
        conv.updated_at = datetime.now(UTC)
        if stored is not conv:
            stored.messages.append(message)
            stored.title, stored.updated_at = conv.title, conv.updated_at
        conv.messages.append(message)

    async def list(self, user_id: str, limit: int = 50) -> list[Conversation]:
        mine = [c for c in self._items.values() if c.user_id == user_id]
        mine.sort(key=lambda c: c.updated_at, reverse=True)
        return [Conversation(id=c.id, user_id=c.user_id, kind=c.kind, created_at=c.created_at,
                             updated_at=c.updated_at, title=c.title) for c in mine[:limit]]
