"""Agent runs: call the model, run the tools it asks for, repeat until done.

The loop itself is a LangGraph graph (``agent/graph.py``); this module holds the
entry points that guard the conversation and stream the graph's events.
"""

from collections.abc import AsyncIterator

from aiplatform.agent.actions import ActionStore
from aiplatform.agent.events import (  # noqa: F401  (re-exported for callers)
    AgentDone,
    AgentEvent,
    AgentText,
    ApprovalRequired,
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
    initial_state,
    recursion_limit,
)
from aiplatform.agent.tools import Tool
from aiplatform.chat.history import check_history_size
from aiplatform.chat.inflight import InFlight
from aiplatform.chat.prompts import reply_language
from aiplatform.chat.repository import Conversation, ConversationRepository
from aiplatform.graph_stream import stream_graph
from aiplatform.llm.gateway import AIGateway
from aiplatform.usage import UsageStore


class AgentRunner:
    def __init__(self, gateway: AIGateway, repo: ConversationRepository, usage: UsageStore,
                 inflight: InFlight, actions: ActionStore, tools: list[Tool], *,
                 max_iterations: int = 8):
        self._gateway = gateway
        self._repo = repo
        self._usage = usage
        self._inflight = inflight
        self._actions = actions
        self._graph = build_agent_graph(
            gateway=gateway, repo=repo, usage=usage, actions=actions,
            tools={t.name: t for t in tools}, max_iterations=max_iterations)
        self._config = {"recursion_limit": recursion_limit(max_iterations)}

    async def run(self, user_id: str, conversation_id: str, text: str,
                  language: str | None = None) -> AsyncIterator[AgentEvent]:
        conv = await self._repo.get(conversation_id, user_id)
        with self._inflight.hold(conv.id):
            check_history_size(conv.messages)
            await self._repo.append(conv, {"role": "user", "content": text})
            async for event in self._stream(user_id, conv, language):
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
                      decision: Decision | None = None) -> AsyncIterator[AgentEvent]:
        state = initial_state(user_id, conv, reply_language(language), decision)
        async for event in stream_graph(self._graph, state, self._config):
            yield event
