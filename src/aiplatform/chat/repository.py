"""Conversation storage. History is append-only: earlier turns are never edited,
which keeps prompt caches valid and is required by newer Claude models."""

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal, Protocol

Kind = Literal["chat", "agent"]


class ConversationNotFound(Exception):
    pass


@dataclass
class Conversation:
    id: str
    user_id: str
    kind: Kind
    created_at: datetime
    messages: list[dict] = field(default_factory=list)


class ConversationRepository(Protocol):
    def create(self, user_id: str, kind: Kind) -> Conversation: ...
    def get(self, conversation_id: str, user_id: str) -> Conversation: ...
    def append(self, conversation_id: str, message: dict) -> None: ...


class InMemoryConversationRepository:
    """Single-process store for local development and v1 demos.

    Swap for a Postgres implementation before running more than one replica.
    """

    def __init__(self) -> None:
        self._items: dict[str, Conversation] = {}

    def create(self, user_id: str, kind: Kind) -> Conversation:
        conv = Conversation(id=str(uuid.uuid4()), user_id=user_id, kind=kind,
                            created_at=datetime.now(UTC))
        self._items[conv.id] = conv
        return conv

    def get(self, conversation_id: str, user_id: str) -> Conversation:
        conv = self._items.get(conversation_id)
        if conv is None or conv.user_id != user_id:  # never reveal other users' conversations
            raise ConversationNotFound(conversation_id)
        return conv

    def append(self, conversation_id: str, message: dict) -> None:
        self._items[conversation_id].messages.append(message)
