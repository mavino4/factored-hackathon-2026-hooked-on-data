"""A chat turn as a one-node LangGraph graph: stream the model reply, record usage,
store the reply. The user message is added (or checked, on regenerate) before the run."""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from aiplatform.chat.history import assistant_turn
from aiplatform.chat.prompts import CHAT_SYSTEM_PROMPT
from aiplatform.chat.repository import Conversation, ConversationRepository
from aiplatform.graph_stream import emitter
from aiplatform.llm.gateway import AIGateway, Completed
from aiplatform.llm.models import ROUTES
from aiplatform.usage import UsageEvent, UsageStore


class ChatState(TypedDict):
    user_id: str
    conv: Conversation
    suffix: str | None  # per-request system text (reply language)


def build_chat_graph(*, gateway: AIGateway, repo: ConversationRepository, usage: UsageStore):
    route = ROUTES["chat"]

    async def reply(state: ChatState) -> dict:
        emit, conv = emitter(), state["conv"]
        async for event in gateway.stream(route, system=CHAT_SYSTEM_PROMPT,
                                          messages=conv.messages, conversation_id=conv.id,
                                          system_suffix=state["suffix"]):
            if isinstance(event, Completed):
                message = event.message
                await usage.record(UsageEvent.from_message(
                    message, user_id=state["user_id"], conversation_id=conv.id,
                    route=route.name, provider=event.provider))
                if message.stop_reason != "refusal" and message.content:
                    await repo.append(conv, assistant_turn(message))
            await emit(event)
        return {}

    graph = StateGraph(ChatState)
    graph.add_node("reply", reply)
    graph.add_edge(START, "reply")
    graph.add_edge("reply", END)
    return graph.compile(name="chat")
