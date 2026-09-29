"""Chat turns: append the user message, stream the model reply, persist it.

The reply itself runs as a LangGraph graph (``chat/graph.py``)."""

from collections.abc import AsyncIterator

from aiplatform.chat.graph import build_chat_graph
from aiplatform.chat.history import (  # noqa: F401  (re-exported for callers)
    MAX_HISTORY_CHARS,
    ConversationTooLong,
    assistant_turn,
    check_history_size,
)
from aiplatform.chat.inflight import InFlight
from aiplatform.chat.prompts import reply_language
from aiplatform.chat.repository import Conversation, ConversationRepository
from aiplatform.graph_stream import stream_graph
from aiplatform.llm.gateway import AIGateway, Completed, TextDelta
from aiplatform.usage import UsageStore


class NothingToRegenerate(Exception):
    pass


class ChatService:
    def __init__(self, gateway: AIGateway, repo: ConversationRepository,
                 usage: UsageStore, inflight: InFlight):
        self._gateway = gateway
        self._repo = repo
        self._usage = usage
        self._inflight = inflight
        self._graph = build_chat_graph(gateway=gateway, repo=repo, usage=usage)

    async def send(self, user_id: str, conversation_id: str, text: str,
                   language: str | None = None) -> AsyncIterator[TextDelta | Completed]:
        conv = await self._repo.get(conversation_id, user_id)
        with self._inflight.hold(conv.id):
            check_history_size(conv.messages)
            await self._repo.append(conv, {"role": "user", "content": text})
            async for event in self._reply(user_id, conv, language):
                yield event

    async def regenerate(self, user_id: str, conversation_id: str,
                         language: str | None = None) -> AsyncIterator[TextDelta | Completed]:
        """Answer the last user message again (e.g. after an interrupted stream),
        without adding a duplicate user message."""
        conv = await self._repo.get(conversation_id, user_id)
        with self._inflight.hold(conv.id):
            if not conv.messages or conv.messages[-1]["role"] != "user":
                raise NothingToRegenerate()
            async for event in self._reply(user_id, conv, language):
                yield event

    async def _reply(self, user_id: str, conv: Conversation,
                     language: str | None) -> AsyncIterator[TextDelta | Completed]:
        state = {"user_id": user_id, "conv": conv, "suffix": reply_language(language)}
        async for event in stream_graph(self._graph, state, {}):
            yield event
