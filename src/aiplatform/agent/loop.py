"""Agent runs: classify the message, call the model, run the tools it asks for, repeat
until done, or hand the conversation to a human advisor.

The flow itself is a LangGraph graph (``agent/graph.py``); this module holds the entry
points that guard the conversation and stream the graph's events, plus the handoff
actions of customers (accept or decline an offer) and operators (reply, close).
"""

import logging
from collections.abc import AsyncIterator

from aiplatform.agent.actions import ActionStore
from aiplatform.agent.events import (  # noqa: F401  (re-exported for callers)
    AgentDone,
    AgentEvent,
    AgentText,
    ApprovalRequired,
    HandoffOffered,
    HandoffStarted,
    HumanWaiting,
    Outcome,
    ToolCall,
    ToolResult,
)
from aiplatform.agent.graph import (  # noqa: F401  (re-exported for callers)
    MAX_TOOL_RESULT_CHARS,
    Decision,
    awaiting_approval_text,
    build_agent_graph,
    decision_text,
    handoff_text,
    initial_state,
    recursion_limit,
)
from aiplatform.agent.handoffs import Handoff, HandoffNotFound, HandoffStore, InMemoryHandoffStore
from aiplatform.agent.tools import Tool
from aiplatform.chat.history import check_history_size
from aiplatform.chat.inflight import InFlight
from aiplatform.chat.prompts import reply_language
from aiplatform.chat.repository import Conversation, ConversationRepository
from aiplatform.graph_stream import stream_graph
from aiplatform.llm.gateway import AIGateway
from aiplatform.tracing import Tracing, run_config
from aiplatform.usage import UsageStore

log = logging.getLogger(__name__)


class AgentRunner:
    def __init__(self, gateway: AIGateway, repo: ConversationRepository, usage: UsageStore,
                 inflight: InFlight, actions: ActionStore, tools: list[Tool], *,
                 handoffs: HandoffStore | None = None, max_iterations: int = 8,
                 tracing: Tracing | None = None):
        self._gateway = gateway
        self._repo = repo
        self._usage = usage
        self._inflight = inflight
        self._actions = actions
        self.handoffs = handoffs or InMemoryHandoffStore()
        self._tracing = tracing
        self._graph = build_agent_graph(
            gateway=gateway, repo=repo, usage=usage, actions=actions, handoffs=self.handoffs,
            tools={t.name: t for t in tools}, max_iterations=max_iterations)
        self._config = {"recursion_limit": recursion_limit(max_iterations)}

    async def run(self, user_id: str, conversation_id: str, text: str,
                  language: str | None = None) -> AsyncIterator[AgentEvent]:
        conv = await self._repo.get(conversation_id, user_id)
        with self._inflight.hold(conv.id):
            check_history_size(conv.messages)
            await self._repo.append(conv, {"role": "user", "content": text})
            async for event in self._stream(user_id, conv, language, trace_input=text):
                yield event

    async def decide(self, user_id: str, conversation_id: str, action_id: str,
                     approve: bool, language: str | None = None) -> AsyncIterator[AgentEvent]:
        """Approve (run the tool) or reject a pending action, then let the agent continue."""
        conv = await self._repo.get(conversation_id, user_id)
        with self._inflight.hold(conv.id):
            async for event in self._stream(user_id, conv, language,
                                            Decision(action_id, approve)):
                yield event

    async def _stream(self, user_id: str, conv: Conversation, language: str | None,
                      decision: Decision | None = None,
                      trace_input: str | None = None) -> AsyncIterator[AgentEvent]:
        handoff = await self.handoffs.latest(conv.id)
        state = initial_state(user_id, conv, reply_language(language), decision,
                              language=language, handoff=handoff)
        config = {**self._config, **run_config(
            "agent", user_id=user_id, conversation_id=conv.id, language=language,
            trace_input=trace_input or (decision._asdict() if decision else None),
            decision=decision._asdict() if decision else None)}
        async for event in stream_graph(self._graph, state, config, self._tracing):
            yield event

    # -- Handoffs ---------------------------------------------------------------------

    async def answer_offer(self, user_id: str, conversation_id: str, handoff_id: str,
                           accept: bool, language: str | None = None) -> Handoff:
        """The customer accepts (queue for an advisor) or declines an offered advisor."""
        conv = await self._repo.get(conversation_id, user_id)
        with self._inflight.hold(conv.id):
            handoff = await self.handoffs.transition(
                handoff_id, expected="offered", to="open" if accept else "declined",
                conversation_id=conv.id, user_id=user_id)
            if accept:
                await self._repo.append(conv, {"role": "assistant", "content": [
                    {"type": "text", "text": handoff_text(language)}]})
        return handoff

    async def operator_reply(self, handoff_id: str, operator: str, text: str) -> None:
        """An advisor writes to the customer of an open handoff."""
        handoff = await self.handoffs.get(handoff_id)
        if handoff.status != "open":
            raise HandoffNotFound(handoff_id)
        conv = await self._repo.get(handoff.conversation_id, handoff.user_id)
        with self._inflight.hold(conv.id):
            await self._repo.append(conv, {"role": "assistant", "author": "operator",
                                           "content": [{"type": "text", "text": text}]})
        log.info("operator reply", extra={"handoff_id": handoff_id, "operator": operator})

    async def close_handoff(self, handoff_id: str) -> Handoff:
        """The advisor is done: the bot answers the conversation again."""
        return await self.handoffs.transition(handoff_id, expected="open", to="closed")
