"""One reply at a time per conversation, so concurrent requests can't interleave history."""

from collections.abc import Iterator
from contextlib import contextmanager


class ConversationBusy(Exception):
    pass


class InFlight:
    """In-process guard (single event loop). The Postgres ``seq`` key covers multiple replicas."""

    def __init__(self) -> None:
        self._busy: set[str] = set()

    def is_busy(self, conversation_id: str) -> bool:
        return conversation_id in self._busy

    @contextmanager
    def hold(self, conversation_id: str) -> Iterator[None]:
        if conversation_id in self._busy:
            raise ConversationBusy(conversation_id)
        self._busy.add(conversation_id)
        try:
            yield
        finally:
            self._busy.discard(conversation_id)
