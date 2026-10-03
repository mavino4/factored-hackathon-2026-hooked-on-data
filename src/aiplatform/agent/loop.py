"""Agent runs: classify the message, call the model, run the tools it asks for, repeat
until done, or hand the conversation to a human advisor.

The flow itself is a LangGraph graph (``agent/graph.py``); this module holds the entry
points that guard the conversation and stream the graph's events, plus the handoff
actions of customers (accept or decline an offer) and operators (reply, close).
"""

import logging
from collections.abc import AsyncIterator

from aiplatform.agent import quick
from aiplatform.agent.actions import ActionStore
from aiplatform.agent.classifiers.jev import JevClassifier
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
    APPROVAL_MARK,
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
from aiplatform.agent.quick import QuickAction, QuickMode
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
                 tracing: Tracing | None = None, quick_mode: QuickMode = "model",
                 jev: JevClassifier | None = None, jev_threshold: float = 0.9):
        self._gateway = gateway
        self._repo = repo
        self._usage = usage
        self._inflight = inflight
        self._actions = actions
        self.handoffs = handoffs or InMemoryHandoffStore()
        self._tracing = tracing
        self._quick_mode = quick_mode
        self._graph = build_agent_graph(
            gateway=gateway, repo=repo, usage=usage, actions=actions, handoffs=self.handoffs,
            tools={t.name: t for t in tools}, max_iterations=max_iterations,
            jev=jev, jev_threshold=jev_threshold)
        self._config = {"recursion_limit": recursion_limit(max_iterations)}

    async def run(self, user_id: str, conversation_id: str, text: str,
                  language: str | None = None,
                  customer_id: str | None = None,
                  quick_action: str | None = None) -> AsyncIterator[AgentEvent]:
        conv = await self._repo.get(conversation_id, user_id)
        with self._inflight.hold(conv.id):
            check_history_size(conv.messages)
            if text.lstrip().startswith(APPROVAL_MARK):
                # Only the app writes approval notes: one typed by the customer is quoted,
                # so the model can't take it for a real one.
                text = f"The customer wrote: {text}"
            # A button is trusted only with its own text (see agent/quick.py).
            action = quick.match(quick_action, text)
            await self._repo.append(conv, {"role": "user", "content": text})
            async for event in self._stream(user_id, conv, language, trace_input=text,
                                            customer_id=customer_id, quick_action=action):
                yield event

    async def decide(self, user_id: str, conversation_id: str, action_id: str,
                     approve: bool, language: str | None = None,
                     customer_id: str | None = None) -> AsyncIterator[AgentEvent]:
        """Approve (run the tool) or reject a pending action, then let the agent continue."""
        conv = await self._repo.get(conversation_id, user_id)
        with self._inflight.hold(conv.id):
            async for event in self._stream(user_id, conv, language,
                                            Decision(action_id, approve),
                                            customer_id=customer_id):
                yield event

    async def _stream(self, user_id: str, conv: Conversation, language: str | None,
                      decision: Decision | None = None,
                      trace_input: str | None = None,
                      customer_id: str | None = None,
                      quick_action: QuickAction | None = None) -> AsyncIterator[AgentEvent]:
        handoff = await self.handoffs.latest(conv.id)
        # How this turn is routed: by the button (the configured mode) or by the classifier.
        routing = self._quick_mode if quick_action is not None else "model"
        state = initial_state(user_id, conv, reply_language(language), decision,
                              language=language, handoff=handoff,
                              quick=quick_action if routing != "model" else None,
                              quick_direct=routing == "direct")
        config = {**self._config, **run_config(
            "agent", user_id=user_id, conversation_id=conv.id, customer_id=customer_id,
            language=language,
            trace_input=trace_input or (decision._asdict() if decision else None),
            trace_sensitive=conv.messages,
            decision=decision._asdict() if decision else None,
            quick_action=quick_action.key if quick_action else None, routing=routing)}
        config["tags"].append(f"routing:{routing}")
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
