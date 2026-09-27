"""Agent tool loop: call the model, run the tools it asks for, repeat until done.

Runs are streamed as events. Irreversible tools never run inside the loop: they
become pending actions, and run only after the user approves them.
"""

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Literal

from aiplatform.agent.actions import ActionStore, PendingAction
from aiplatform.agent.tools import Tool
from aiplatform.chat.inflight import InFlight
from aiplatform.chat.prompts import AGENT_SYSTEM_PROMPT
from aiplatform.chat.repository import Conversation, ConversationRepository
from aiplatform.chat.service import assistant_turn, check_history_size
from aiplatform.llm.gateway import AIGateway, Completed, TextDelta
from aiplatform.llm.models import ROUTES
from aiplatform.usage import UsageEvent, UsageStore

log = logging.getLogger(__name__)

Outcome = Literal["done", "approval_required", "refused", "truncated", "max_iterations"]

# Keep one tool result from flooding the context window.
MAX_TOOL_RESULT_CHARS = 20_000


def awaiting_approval_text(action: PendingAction) -> str:
    return (f"Awaiting the user's approval (action {action.id}). This action has NOT been "
            "executed yet. Tell the user briefly what you intend to do; they will approve "
            "or reject it.")


def decision_text(action: PendingAction, result: str | None) -> str:
    # Sent as a user-role message: history stays append-only (tool results are never edited).
    header = f"[Approval] The user {action.status.upper()} action {action.id} ({action.tool_name})."
    if result is None:
        return header + " It was not executed."
    return f"{header} It was executed. Result:\n{result}"


# --- Events streamed to the client -------------------------------------------

@dataclass
class AgentText:
    text: str


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class ToolResult:
    id: str
    name: str
    is_error: bool
    content: str


@dataclass
class ApprovalRequired:
    action_id: str
    tool_name: str
    input: dict[str, Any]


@dataclass
class AgentDone:
    outcome: Outcome
    text: str
    tool_calls: list[str]


AgentEvent = AgentText | ToolCall | ToolResult | ApprovalRequired | AgentDone


class AgentRunner:
    def __init__(self, gateway: AIGateway, repo: ConversationRepository, usage: UsageStore,
                 inflight: InFlight, actions: ActionStore, tools: list[Tool], *,
                 max_iterations: int = 8):
        self._gateway = gateway
        self._repo = repo
        self._usage = usage
        self._inflight = inflight
        self._actions = actions
        self._tools = {t.name: t for t in tools}
        self._definitions = [t.definition() for t in tools]  # fixed per route for caching
        self._max_iterations = max_iterations

    async def run(self, user_id: str, conversation_id: str,
                  text: str) -> AsyncIterator[AgentEvent]:
        conv = await self._repo.get(conversation_id, user_id)
        with self._inflight.hold(conv.id):
            check_history_size(conv.messages)
            await self._repo.append(conv, {"role": "user", "content": text})
            async for event in self._loop(user_id, conv):
                yield event

    async def decide(self, user_id: str, conversation_id: str, action_id: str,
                     approve: bool) -> AsyncIterator[AgentEvent]:
        """Approve (run the tool) or reject a pending action, then let the agent continue."""
        conv = await self._repo.get(conversation_id, user_id)
        with self._inflight.hold(conv.id):
            action = await self._actions.decide(action_id, user_id=user_id,
                                                conversation_id=conv.id, approve=approve)
            result = None
            if approve:
                tool = self._tools.get(action.tool_name)
                if tool is None:
                    content, is_error = f"tool no longer available: {action.tool_name}", True
                else:
                    content, is_error = await self._invoke(tool, action.input)
                yield ToolResult(action.tool_use_id, action.tool_name, is_error, content)
                result = f"ERROR: {content}" if is_error else content
            await self._repo.append(conv, {"role": "user", "content": decision_text(action, result)})
            async for event in self._loop(user_id, conv):
                yield event

    async def _loop(self, user_id: str, conv: Conversation) -> AsyncIterator[AgentEvent]:
        route = ROUTES["agent"]
        tool_calls: list[str] = []
        pending = False
        for _ in range(self._max_iterations):
            check_history_size(conv.messages)
            completed: Completed | None = None
            async for event in self._gateway.stream(
                    route, system=AGENT_SYSTEM_PROMPT, messages=conv.messages,
                    tools=self._definitions, conversation_id=conv.id):
                if isinstance(event, TextDelta):
                    yield AgentText(event.text)
                else:
                    completed = event
            assert completed is not None
            message = completed.message
            await self._usage.record(UsageEvent.from_message(
                message, user_id=user_id, conversation_id=conv.id,
                route=route.name, provider=completed.provider))
            final_text = "".join(b.text for b in message.content if b.type == "text")

            if message.stop_reason == "refusal":
                yield AgentDone("refused", final_text, tool_calls)
                return
            tool_uses = [b for b in message.content if b.type == "tool_use"]
            if tool_uses and message.stop_reason == "max_tokens":
                # A truncated tool call must not run; don't store the half-finished turn.
                yield AgentDone("truncated", final_text, tool_calls)
                return

            await self._repo.append(conv, assistant_turn(message))
            if message.stop_reason == "pause_turn":
                continue
            if not tool_uses:
                yield AgentDone("approval_required" if pending else "done", final_text,
                                tool_calls)
                return

            for block in tool_uses:
                yield ToolCall(block.id, block.name, block.input)
            # Run all requested tools concurrently; return every result in ONE user message.
            outcomes = await asyncio.gather(*(self._execute(b, user_id, conv) for b in tool_uses))
            results = []
            for block, (result, event) in zip(tool_uses, outcomes, strict=True):
                results.append(result)
                yield event
                if isinstance(event, ApprovalRequired):
                    pending = True
            tool_calls.extend(b.name for b in tool_uses)
            await self._repo.append(conv, {"role": "user", "content": results})

        yield AgentDone("max_iterations", "", tool_calls)

    async def _execute(self, block, user_id: str,
                       conv: Conversation) -> tuple[dict, ToolResult | ApprovalRequired]:
        def result(content: str, is_error: bool = False) -> tuple[dict, ToolResult]:
            out = {"type": "tool_result", "tool_use_id": block.id, "content": content}
            if is_error:
                out["is_error"] = True
            return out, ToolResult(block.id, block.name, is_error, content)

        tool = self._tools.get(block.name)
        if tool is None:
            return result(f"unknown tool: {block.name}", True)
        if error := tool.validate(block.input):
            return result(f"invalid input: {error}", True)
        if tool.irreversible:
            action = await self._actions.create(
                conversation_id=conv.id, user_id=user_id, tool_use_id=block.id,
                tool_name=block.name, input=block.input)
            out = {"type": "tool_result", "tool_use_id": block.id,
                   "content": awaiting_approval_text(action)}
            return out, ApprovalRequired(action.id, block.name, block.input)
        return result(*await self._invoke(tool, block.input))

    async def _invoke(self, tool: Tool, args: dict[str, Any]) -> tuple[str, bool]:
        try:
            output = await asyncio.wait_for(tool.handler(args), tool.timeout_s)
        except TimeoutError:
            return "tool timed out", True
        except Exception as exc:
            log.exception("tool failed", extra={"tool": tool.name})
            return f"tool failed: {type(exc).__name__}", True
        if not isinstance(output, str):
            output = json.dumps(output, default=str)
        if len(output) > MAX_TOOL_RESULT_CHARS:
            output = output[:MAX_TOOL_RESULT_CHARS] + "\n[truncated: result too long]"
        return output, False
