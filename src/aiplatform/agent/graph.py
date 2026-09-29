"""The agent tool loop as a LangGraph graph.

    START ─┬──────────────────> call_model ──> run_tools ──┐
           └─> apply_decision ──┘  ▲  │                    │
                                   │  └─ pause_turn ──┐    │
                                   └──── while iterations < max_iterations
    call_model (no tool calls, refused, truncated) or the iteration cap ──> finish ──> END

The graph keeps nothing between requests (no checkpointer): the conversation and the
pending actions live in their stores, so an approval starts a new run at
``apply_decision``. Irreversible tools never run inside the graph: they become pending
actions. Nodes emit events with ``graph_stream.emitter()``, which waits for the client
to take each one (see ``graph_stream``).
"""

import asyncio
import json
import logging
from typing import Any, NamedTuple, TypedDict

from langgraph.graph import END, START, StateGraph

from aiplatform.agent.actions import ActionStore, PendingAction
from aiplatform.agent.events import (
    AgentDone,
    AgentText,
    ApprovalRequired,
    Outcome,
    ToolCall,
    ToolResult,
)
from aiplatform.agent.tools import Tool, ToolContext
from aiplatform.chat.history import assistant_turn, check_history_size
from aiplatform.chat.prompts import AGENT_SYSTEM_PROMPT
from aiplatform.chat.repository import Conversation, ConversationRepository
from aiplatform.graph_stream import emitter
from aiplatform.llm.gateway import AIGateway, Completed, TextDelta
from aiplatform.llm.models import ROUTES
from aiplatform.usage import UsageEvent, UsageStore

log = logging.getLogger(__name__)

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


class Decision(NamedTuple):
    action_id: str
    approve: bool


class AgentState(TypedDict):
    user_id: str
    conv: Conversation
    suffix: str | None  # per-request system text (reply language)
    decision: Decision | None  # set when the run resumes after an approval decision
    iterations: int  # model calls so far
    tool_calls: list[str]
    pending: bool  # an irreversible tool is waiting for approval
    tool_uses: list[Any]  # tool_use blocks of the last model message
    outcome: Outcome | None  # set when the run should finish
    final_text: str


def initial_state(user_id: str, conv: Conversation, suffix: str | None,
                  decision: Decision | None = None) -> AgentState:
    return AgentState(user_id=user_id, conv=conv, suffix=suffix, decision=decision,
                      iterations=0, tool_calls=[], pending=False, tool_uses=[],
                      outcome=None, final_text="")


def recursion_limit(max_iterations: int) -> int:
    # call_model + run_tools per iteration, plus apply_decision and finish; our own
    # iteration cap stops the run first.
    return 2 * max_iterations + 3


async def invoke_tool(tool: Tool, args: dict[str, Any], ctx: ToolContext) -> tuple[str, bool]:
    try:
        output = await asyncio.wait_for(tool.handler(args, ctx), tool.timeout_s)
    except TimeoutError:
        return "tool timed out", True
    except Exception as exc:
        log.exception("tool failed", extra={"tool": tool.name})
        return f"tool failed: {type(exc).__name__}", True
    if not isinstance(output, str):
        output = json.dumps(output, default=str, ensure_ascii=False)
    if len(output) > MAX_TOOL_RESULT_CHARS:
        output = output[:MAX_TOOL_RESULT_CHARS] + "\n[truncated: result too long]"
    return output, False


def build_agent_graph(*, gateway: AIGateway, repo: ConversationRepository, usage: UsageStore,
                      actions: ActionStore, tools: dict[str, Tool], max_iterations: int):
    definitions = [t.definition() for t in tools.values()]  # fixed per route for caching
    route = ROUTES["agent"]

    async def apply_decision(state: AgentState) -> dict:
        """Approve (run the tool) or reject a pending action."""
        emit = emitter()
        conv, user_id, decision = state["conv"], state["user_id"], state["decision"]
        action = await actions.decide(decision.action_id, user_id=user_id,
                                      conversation_id=conv.id, approve=decision.approve)
        result = None
        if decision.approve:
            tool = tools.get(action.tool_name)
            if tool is None:
                content, is_error = f"tool no longer available: {action.tool_name}", True
            else:
                content, is_error = await invoke_tool(tool, action.input,
                                                      ToolContext(user_id, conv.id))
            await emit(ToolResult(action.tool_use_id, action.tool_name, is_error, content))
            result = f"ERROR: {content}" if is_error else content
        await repo.append(conv, {"role": "user", "content": decision_text(action, result)})
        return {}

    async def call_model(state: AgentState) -> dict:
        emit = emitter()
        conv = state["conv"]
        check_history_size(conv.messages)
        completed: Completed | None = None
        async for event in gateway.stream(
                route, system=AGENT_SYSTEM_PROMPT, messages=conv.messages,
                tools=definitions, conversation_id=conv.id, system_suffix=state["suffix"]):
            if isinstance(event, TextDelta):
                await emit(AgentText(event.text))
            else:
                completed = event
        assert completed is not None
        message = completed.message
        await usage.record(UsageEvent.from_message(
            message, user_id=state["user_id"], conversation_id=conv.id,
            route=route.name, provider=completed.provider))
        update = {"iterations": state["iterations"] + 1, "tool_uses": [], "outcome": None,
                  "final_text": "".join(b.text for b in message.content if b.type == "text")}

        if message.stop_reason == "refusal":
            return {**update, "outcome": "refused"}
        tool_uses = [b for b in message.content if b.type == "tool_use"]
        if tool_uses and message.stop_reason == "max_tokens":
            # A truncated tool call must not run; don't store the half-finished turn.
            return {**update, "outcome": "truncated"}

        await repo.append(conv, assistant_turn(message))
        if message.stop_reason == "pause_turn":
            return update
        if not tool_uses:
            return {**update, "outcome": "approval_required" if state["pending"] else "done"}
        return {**update, "tool_uses": tool_uses}

    async def execute(block, user_id: str,
                      conv: Conversation) -> tuple[dict, ToolResult | ApprovalRequired]:
        def result(content: str, is_error: bool = False) -> tuple[dict, ToolResult]:
            out = {"type": "tool_result", "tool_use_id": block.id, "content": content}
            if is_error:
                out["is_error"] = True
            return out, ToolResult(block.id, block.name, is_error, content)

        tool = tools.get(block.name)
        if tool is None:
            return result(f"unknown tool: {block.name}", True)
        if error := tool.validate(block.input):
            return result(f"invalid input: {error}", True)
        if tool.irreversible:
            action = await actions.create(
                conversation_id=conv.id, user_id=user_id, tool_use_id=block.id,
                tool_name=block.name, input=block.input)
            out = {"type": "tool_result", "tool_use_id": block.id,
                   "content": awaiting_approval_text(action)}
            return out, ApprovalRequired(action.id, block.name, block.input)
        return result(*await invoke_tool(tool, block.input, ToolContext(user_id, conv.id)))

    async def run_tools(state: AgentState) -> dict:
        emit = emitter()
        conv, blocks, pending = state["conv"], state["tool_uses"], state["pending"]
        for block in blocks:
            await emit(ToolCall(block.id, block.name, block.input))
        # Run all requested tools concurrently; return every result in ONE user message.
        outcomes = await asyncio.gather(*(execute(b, state["user_id"], conv) for b in blocks))
        results = []
        for result, event in outcomes:
            results.append(result)
            await emit(event)
            if isinstance(event, ApprovalRequired):
                pending = True
        await repo.append(conv, {"role": "user", "content": results})
        return {"pending": pending, "tool_calls": [*state["tool_calls"], *(b.name for b in blocks)],
                "tool_uses": []}

    async def finish(state: AgentState) -> dict:
        outcome = state["outcome"]
        if outcome is None:  # stopped by the iteration cap
            await emitter()(AgentDone("max_iterations", "", state["tool_calls"]))
        else:
            await emitter()(AgentDone(outcome, state["final_text"], state["tool_calls"]))
        return {}

    def start(state: AgentState) -> str:
        return "apply_decision" if state["decision"] else "call_model"

    def next_call(state: AgentState) -> str:
        return "call_model" if state["iterations"] < max_iterations else "finish"

    def after_model(state: AgentState) -> str:
        if state["outcome"] is not None:
            return "finish"
        if state["tool_uses"]:
            return "run_tools"
        return next_call(state)  # pause_turn: continue if iterations remain

    graph = StateGraph(AgentState)
    graph.add_node("apply_decision", apply_decision)
    graph.add_node("call_model", call_model)
    graph.add_node("run_tools", run_tools)
    graph.add_node("finish", finish)
    graph.add_conditional_edges(START, start, ["apply_decision", "call_model"])
    graph.add_edge("apply_decision", "call_model")
    graph.add_conditional_edges("call_model", after_model, ["finish", "run_tools", "call_model"])
    graph.add_conditional_edges("run_tools", next_call, ["call_model", "finish"])
    graph.add_edge("finish", END)
    return graph.compile(name="agent")
