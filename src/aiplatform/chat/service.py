"""Chat turns: append the user message, stream the model reply, persist it."""

from collections.abc import AsyncIterator

from aiplatform.chat.inflight import InFlight
from aiplatform.chat.prompts import CHAT_SYSTEM_PROMPT, reply_language
from aiplatform.chat.repository import Conversation, ConversationRepository
from aiplatform.llm.gateway import AIGateway, Completed, TextDelta
from aiplatform.llm.models import ROUTES
from aiplatform.usage import UsageEvent, UsageStore

# Claude Haiku 4.5 has a 200K-token context window; leave room for the reply.
MAX_HISTORY_CHARS = 150_000 * 4  # rough chars-per-token estimate


class ConversationTooLong(Exception):
    pass


class NothingToRegenerate(Exception):
    pass


def check_history_size(messages: list[dict]) -> None:
    if sum(len(str(m["content"])) for m in messages) > MAX_HISTORY_CHARS:
        raise ConversationTooLong()


def assistant_turn(message) -> dict:
    # Keep the full content blocks, not only the text.
    return {"role": "assistant", "content": [block.to_dict() for block in message.content]}


class ChatService:
    def __init__(self, gateway: AIGateway, repo: ConversationRepository,
                 usage: UsageStore, inflight: InFlight):
        self._gateway = gateway
        self._repo = repo
        self._usage = usage
        self._inflight = inflight

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
        route = ROUTES["chat"]
        async for event in self._gateway.stream(route, system=CHAT_SYSTEM_PROMPT,
                                                messages=conv.messages, conversation_id=conv.id,
                                                system_suffix=reply_language(language)):
            if isinstance(event, Completed):
                message = event.message
                await self._usage.record(UsageEvent.from_message(
                    message, user_id=user_id, conversation_id=conv.id,
                    route=route.name, provider=event.provider))
                if message.stop_reason != "refusal" and message.content:
                    await self._repo.append(conv, assistant_turn(message))
            yield event
