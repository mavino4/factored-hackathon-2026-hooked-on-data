"""Agent tool loop: call the model, run the tools it asks for, repeat until done."""

import asyncio
import logging
from dataclasses import dataclass
from typing import Literal

from aiplatform.agent.tools import Tool
from aiplatform.chat.inflight import InFlight
from aiplatform.chat.prompts import AGENT_SYSTEM_PROMPT
from aiplatform.chat.repository import Conversation, ConversationRepository
from aiplatform.chat.service import assistant_turn, check_history_size, usage_tokens
from aiplatform.llm.gateway import AIGateway
from aiplatform.llm.models import ROUTES
from aiplatform.ratelimit import DailyTokenQuota

log = logging.getLogger(__name__)

Outcome = Literal["done", "refused", "truncated", "max_iterations"]

# Keep one tool result from flooding the context window.
MAX_TOOL_RESULT_CHARS = 20_000

APPROVAL_REQUIRED = ("This action requires the user's approval and was NOT executed. "
                     "Describe what you intended to do and ask the user to confirm.")


@dataclass
class AgentResult:
    outcome: Outcome
    text: str
    tool_calls: list[str]


class AgentRunner:
    def __init__(self, gateway: AIGateway, repo: ConversationRepository,
                 quota: DailyTokenQuota, inflight: InFlight, tools: list[Tool], *,
                 max_iterations: int = 8):
        self._gateway = gateway
        self._repo = repo
        self._quota = quota
        self._inflight = inflight
        self._tools = {t.name: t for t in tools}
        self._definitions = [t.definition() for t in tools]  # fixed per route for caching
        self._max_iterations = max_iterations

    async def run(self, user_id: str, conversation_id: str, text: str) -> AgentResult:
        conv = self._repo.get(conversation_id, user_id)
        with self._inflight.hold(conv.id):
            check_history_size(conv.messages)
            self._repo.append(conv.id, {"role": "user", "content": text})
            return await self._loop(user_id, conv)

    async def _loop(self, user_id: str, conv: Conversation) -> AgentResult:
        tool_calls: list[str] = []
        for _ in range(self._max_iterations):
            check_history_size(conv.messages)
            completed = await self._gateway.complete(
                ROUTES["agent"], system=AGENT_SYSTEM_PROMPT, messages=conv.messages,
                tools=self._definitions, conversation_id=conv.id)
            message = completed.message
            self._quota.charge(user_id, usage_tokens(message))
            final_text = "".join(b.text for b in message.content if b.type == "text")

            if message.stop_reason == "refusal":
                return AgentResult("refused", final_text, tool_calls)
            tool_uses = [b for b in message.content if b.type == "tool_use"]
            if tool_uses and message.stop_reason == "max_tokens":
                # A truncated tool call must not run; don't store the half-finished turn.
                return AgentResult("truncated", final_text, tool_calls)

            self._repo.append(conv.id, assistant_turn(message))
            if message.stop_reason == "pause_turn":
                continue
            if not tool_uses:
                return AgentResult("done", final_text, tool_calls)

            # Run all requested tools concurrently; return every result in ONE user message.
            results = await asyncio.gather(*(self._execute(b) for b in tool_uses))
            tool_calls.extend(b.name for b in tool_uses)
            self._repo.append(conv.id, {"role": "user", "content": list(results)})

        return AgentResult("max_iterations", "", tool_calls)

    async def _execute(self, block) -> dict:
        def result(content: str, is_error: bool = False) -> dict:
            out = {"type": "tool_result", "tool_use_id": block.id, "content": content}
            if is_error:
                out["is_error"] = True
            return out

        tool = self._tools.get(block.name)
        if tool is None:
            return result(f"unknown tool: {block.name}", True)
        if error := tool.validate(block.input):
            return result(f"invalid input: {error}", True)
        if tool.irreversible:
            return result(APPROVAL_REQUIRED, True)
        try:
            output = await asyncio.wait_for(tool.handler(block.input), tool.timeout_s)
            if len(output) > MAX_TOOL_RESULT_CHARS:
                output = output[:MAX_TOOL_RESULT_CHARS] + "\n[truncated: result too long]"
            return result(output)
        except TimeoutError:
            return result("tool timed out", True)
        except Exception as exc:
            log.exception("tool failed", extra={"tool": block.name})
            return result(f"tool failed: {type(exc).__name__}", True)
