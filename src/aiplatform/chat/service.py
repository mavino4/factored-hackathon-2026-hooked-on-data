"""Chat turns: append the user message, stream the model reply, persist it."""

from collections.abc import AsyncIterator

from aiplatform.chat.inflight import InFlight
from aiplatform.chat.prompts import CHAT_SYSTEM_PROMPT
from aiplatform.chat.repository import Conversation, ConversationRepository
from aiplatform.llm.gateway import AIGateway, Completed, TextDelta
from aiplatform.llm.models import ROUTES
from aiplatform.ratelimit import DailyTokenQuota

# Claude Haiku 4.5 has a 200K-token context window; leave room for the reply.
MAX_HISTORY_CHARS = 150_000 * 4  # rough chars-per-token estimate


class ConversationTooLong(Exception):
    pass


class NothingToRegenerate(Exception):
    pass


def check_history_size(messages: list[dict]) -> None:
    if sum(len(str(m["content"])) for m in messages) > MAX_HISTORY_CHARS:
        raise ConversationTooLong()


def usage_tokens(message) -> int:
    u = message.usage
    return (u.input_tokens + u.output_tokens + (u.cache_read_input_tokens or 0)
            + (u.cache_creation_input_tokens or 0))


def assistant_turn(message) -> dict:
    # Keep the full content blocks, not only the text.
    return {"role": "assistant", "content": [block.to_dict() for block in message.content]}


class ChatService:
    def __init__(self, gateway: AIGateway, repo: ConversationRepository,
                 quota: DailyTokenQuota, inflight: InFlight):
        self._gateway = gateway
        self._repo = repo
        self._quota = quota
        self._inflight = inflight

    async def send(self, user_id: str, conversation_id: str,
                   text: str) -> AsyncIterator[TextDelta | Completed]:
        conv = self._repo.get(conversation_id, user_id)
        with self._inflight.hold(conv.id):
            check_history_size(conv.messages)
            self._repo.append(conv.id, {"role": "user", "content": text})
            async for event in self._reply(user_id, conv):
                yield event

    async def regenerate(self, user_id: str,
                         conversation_id: str) -> AsyncIterator[TextDelta | Completed]:
        """Answer the last user message again (e.g. after an interrupted stream),
        without adding a duplicate user message."""
        conv = self._repo.get(conversation_id, user_id)
        with self._inflight.hold(conv.id):
            if not conv.messages or conv.messages[-1]["role"] != "user":
                raise NothingToRegenerate()
            async for event in self._reply(user_id, conv):
                yield event

    async def _reply(self, user_id: str, conv: Conversation) -> AsyncIterator[TextDelta | Completed]:
        async for event in self._gateway.stream(ROUTES["chat"], system=CHAT_SYSTEM_PROMPT,
                                                messages=conv.messages, conversation_id=conv.id):
            if isinstance(event, Completed):
                message = event.message
                self._quota.charge(user_id, usage_tokens(message))
                if message.stop_reason != "refusal" and message.content:
                    self._repo.append(conv.id, assistant_turn(message))
            yield event
