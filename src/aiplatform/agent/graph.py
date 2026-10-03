"""The agent as a LangGraph graph: classify the message, then answer (with or without
tools), or hand the conversation to a human.

    START ─┬─ open handoff ──> wait_for_human ──> END      (a human answers, not the bot)
           ├─ quick action ──> quick_intent ────┐   (AIP_QUICK_ACTIONS=intent: no classifier)
           ├─ decision ──────> apply_decision ──┤
           └─ new message ───> classify ─┬─ human ──> handoff ──> END
                                         ├─ attack ─> refuse_attack ──> END   (fixed reply)
                                         └─ account / general / out_of_scope
                                                        │
                           ┌────────────────────────────┴──> call_model ──> run_tools ──┐
                           │      (tools only for account)    ▲  │  └ pause_turn ─┐     │
                           │                                  └──┴─ while iterations < max
    call_model (no tool calls, refused, truncated) or the iteration cap ──> finish ──> END
    finish offers a human advisor when the customer insists without being resolved.

The graph keeps nothing between requests (no checkpointer): the conversation, pending
actions and handoffs live in their stores, so an approval starts a new run at
``apply_decision``. Irreversible tools never run inside the graph: they become pending
actions. Nodes emit events with ``graph_stream.emitter()``, which waits for the client
to take each one (see ``graph_stream``). Each node has its own trace span.
"""

import asyncio
import json
import logging
from dataclasses import replace
from typing import Any, NamedTuple, TypedDict

import anthropic
from langgraph.graph import END, START, StateGraph

from aiplatform import metrics
from aiplatform.agent import intent as intents
from aiplatform.agent.actions import ActionStore, PendingAction
from aiplatform.agent.events import (
    AgentDone,
    AgentText,
    ApprovalRequired,
    HandoffOffered,
    HandoffStarted,
    HumanWaiting,
    Outcome,
    ToolCall,
    ToolResult,
)
from aiplatform.agent.handoffs import Handoff, HandoffStore
from aiplatform.agent.quick import QuickAction
from aiplatform.agent.tools import Tool, ToolContext
from aiplatform.chat.history import assistant_turn, check_history_size
from aiplatform.chat.prompts import AGENT_SYSTEM_PROMPT, CLASSIFY_SYSTEM_PROMPT
from aiplatform.chat.repository import Conversation, ConversationRepository
from aiplatform.graph_stream import emitter
from aiplatform.llm.gateway import AIGateway, Completed, GatewayError, TextDelta
from aiplatform.llm.models import ROUTES
from aiplatform.tracing import (
    flag_prompt_injection,
    record_intent,
    record_outcome,
    register_sensitive,
    tool_span,
    traced_node,
)
from aiplatform.usage import UsageEvent, UsageStore

log = logging.getLogger(__name__)

# Keep one tool result from flooding the context window.
MAX_TOOL_RESULT_CHARS = 20_000

# After the customer declines an advisor, don't offer again for this many of their turns.
REOFFER_AFTER_TURNS = 3

# Fixed replies when the conversation goes to a human (no model call).
HANDOFF_TEXT = {
    "es": ("Entendido. Lo comunico con un asesor, que le responderá en esta misma "
           "conversación. Mientras tanto, puede dejar aquí sus mensajes."),
    "pt": ("Entendido. Vou transferir o senhor para um atendente, que responderá nesta "
           "mesma conversa. Enquanto isso, pode deixar suas mensagens aqui."),
    "en": ("Understood. I'm transferring you to an advisor, who will reply in this same "
           "conversation. Meanwhile, you can leave your messages here."),
}


# Fixed reply to a manipulation attempt (no model call): it says what the bot is for and
# gives the attacker nothing to iterate on.
ATTACK_TEXT = {
    "es": ("No puedo ayudarle con esa solicitud. Estoy aquí para consultar los saldos y el "
           "estado de sus productos y para resolver dudas sobre productos bancarios."),
    "pt": ("Não posso ajudar com essa solicitação. Estou aqui para consultar os saldos e a "
           "situação dos seus produtos e para tirar dúvidas sobre produtos bancários."),
    "en": ("I can't help with that request. I'm here to check the balances and status of "
           "your products and to answer questions about banking products."),
}

# Force a tool call on the first model call of an account question: figures must come
# from the bank, not from anything the customer typed.
REQUIRE_TOOL = {"type": "any"}

# How the app's own note about an approval decision starts (stored with author "system").
APPROVAL_MARK = "[Approval]"


def handoff_text(language: str | None) -> str:
    return HANDOFF_TEXT.get(language or "es", HANDOFF_TEXT["es"])


def attack_text(language: str | None) -> str:
    return ATTACK_TEXT.get(language or "es", ATTACK_TEXT["es"])


def customer_turns_since(messages: list[dict], index: int) -> int:
    return sum(1 for m in messages[index:] if intents.is_customer_message(m))


def awaiting_approval_text(action: PendingAction) -> str:
    return (f"Awaiting the user's approval (action {action.id}). This action has NOT been "
            "executed yet. Tell the user briefly what you intend to do; they will approve "
            "or reject it.")


def decision_text(action: PendingAction, result: str | None) -> str:
    # Sent as a user-role message: history stays append-only (tool results are never edited).
    header = (f"{APPROVAL_MARK} The user {action.status.upper()} action {action.id} "
              f"({action.tool_name}).")
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
    language: str | None  # the language the customer sees (for fixed texts)
    decision: Decision | None  # set when the run resumes after an approval decision
    handoff: Handoff | None  # the conversation's latest handoff when the run started
    intent: intents.Intent | None  # set by classify
    quick: QuickAction | None  # a trusted quick action that skips the classifier
    use_tools: bool  # send the tool definitions to the model
    iterations: int  # model calls so far
    tool_calls: list[str]
    pending: bool  # an irreversible tool is waiting for approval
    tool_uses: list[Any]  # tool_use blocks of the last model message
    outcome: Outcome | None  # set when the run should finish
    final_text: str


def initial_state(user_id: str, conv: Conversation, suffix: str | None,
                  decision: Decision | None = None, *, language: str | None = None,
                  handoff: Handoff | None = None,
                  quick: QuickAction | None = None) -> AgentState:
    return AgentState(user_id=user_id, conv=conv, suffix=suffix, language=language,
                      decision=decision, handoff=handoff, intent=None, quick=quick,
                      # An approval resumes a tool loop, which needs its tools.
                      use_tools=decision is not None,
                      iterations=0, tool_calls=[], pending=False, tool_uses=[],
                      outcome=None, final_text="")


def recursion_limit(max_iterations: int) -> int:
    # call_model + run_tools per iteration, plus classify (or apply_decision) and finish;
    # our own iteration cap stops the run first.
    return 2 * max_iterations + 4


async def invoke_tool(tool: Tool, args: dict[str, Any], ctx: ToolContext) -> tuple[str, bool]:
    # One traced tool call (a no-op unless tracing is on).
    async with tool_span(tool.name, args, {"irreversible": tool.irreversible}) as span:
        output, is_error = await _invoke_tool(tool, args, ctx)
        register_sensitive(output)  # mask these values wherever they appear later
        span.update(output=output, level="ERROR" if is_error else None,
                    status_message=output if is_error else None)
    return output, is_error


async def _invoke_tool(tool: Tool, args: dict[str, Any], ctx: ToolContext) -> tuple[str, bool]:
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


def _last_customer_text(conv: Conversation) -> str | None:
    for m in reversed(conv.messages):
        if intents.is_customer_message(m):
            return m["content"]
    return None


def build_agent_graph(*, gateway: AIGateway, repo: ConversationRepository, usage: UsageStore,
                      actions: ActionStore, handoffs: HandoffStore, tools: dict[str, Tool],
                      max_iterations: int):
    definitions = [t.definition() for t in tools.values()]  # fixed per route for caching
    route = ROUTES["agent"]
    classify_route = ROUTES["classify"]

    async def record(state: AgentState, completed: Completed, route_name: str) -> None:
        await usage.record(UsageEvent.from_message(
            completed.message, user_id=state["user_id"], conversation_id=state["conv"].id,
            route=route_name, provider=completed.provider))

    @traced_node("wait_for_human", lambda s: {"handoff": s["handoff"].id})
    async def wait_for_human(state: AgentState) -> dict:
        """A human handles this conversation: the message is stored, the bot stays quiet."""
        emit = emitter()
        record_outcome("handoff")
        await emit(HumanWaiting(state["handoff"].id))
        await emit(AgentDone("handoff", "", []))
        return {}

    @traced_node("classify", lambda s: {"message": _last_customer_text(s["conv"])})
    async def classify(state: AgentState) -> dict:
        """What does the customer want, can we resolve it, and does it need the tools?"""
        conv = state["conv"]
        try:
            completed = await gateway.complete(
                classify_route, system=CLASSIFY_SYSTEM_PROMPT,
                messages=intents.classify_messages(conv.messages),
                tools=[intents.CLASSIFY_TOOL], tool_choice=intents.FORCE_CLASSIFY,
                conversation_id=conv.id)
        except (GatewayError, anthropic.APIError):
            # Never lose the turn over the classifier: answer with the full agent.
            log.warning("intent classifier unavailable; using the full agent", exc_info=True)
            intent = intents.FALLBACK
        else:
            await record(state, completed, classify_route.name)
            intent = intents.parse(completed.message)
        metrics.AGENT_INTENTS.labels(intent.name).inc()
        record_intent(intent.name)
        if intent.name == "attack":
            # Repeating an attack is not a customer who needs a person: no advisor offer.
            intent = replace(intent, needs_tools=False, insistence=False)
            log.warning("manipulation attempt blocked", extra={
                "conversation_id": conv.id, "reason": intent.reason})
            flag_prompt_injection(intent.reason)
            return {"intent": intent, "use_tools": False}
        if not intent.insistence and intents.repeated(conv.messages):
            intent = replace(intent, insistence=True)
        use_tools = intent.name == "account" or intent.needs_tools
        return {"intent": intent, "use_tools": use_tools}

    @traced_node("quick_intent", lambda s: {"quick_action": s["quick"].key})
    async def quick_intent(state: AgentState) -> dict:
        """A quick action: an account question by construction, so no classifier call."""
        conv = state["conv"]
        intent = intents.Intent("account", needs_tools=True,
                                insistence=intents.repeated(conv.messages),
                                reason=f"quick action {state['quick'].key}")
        metrics.AGENT_INTENTS.labels(intent.name).inc()
        record_intent(intent.name)
        return {"intent": intent, "use_tools": True}

    @traced_node("refuse_attack", lambda s: {"reason": s["intent"].reason})
    async def refuse_attack(state: AgentState) -> dict:
        """A manipulation attempt: the fixed reply, without the agent or its tools."""
        emit, conv = emitter(), state["conv"]
        text = attack_text(state["language"])
        await repo.append(conv, {"role": "assistant", "content": [{"type": "text", "text": text}]})
        record_outcome("blocked", text)
        await emit(AgentText(text))
        await emit(AgentDone("blocked", text, []))
        return {}

    @traced_node("handoff", lambda s: {"reason": s["intent"].reason if s["intent"] else None})
    async def handoff(state: AgentState) -> dict:
        """The customer asked for a person: queue the conversation for an advisor."""
        emit, conv = emitter(), state["conv"]
        item = await handoffs.create(
            conversation_id=conv.id, user_id=state["user_id"], status="open",
            reason="customer_request", summary=state["intent"].reason,
            message_index=len(conv.messages))
        text = handoff_text(state["language"])
        await repo.append(conv, {"role": "assistant", "content": [{"type": "text", "text": text}]})
        record_outcome("handoff", text)
        await emit(AgentText(text))
        await emit(HandoffStarted(item.id))
        await emit(AgentDone("handoff", text, []))
        return {"handoff": item}

    @traced_node("apply_decision", lambda s: s["decision"]._asdict())
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
        # author "system": the app wrote this, not the customer (see is_customer_message).
        await repo.append(conv, {"role": "user", "author": "system",
                                 "content": decision_text(action, result)})
        return {}

    @traced_node("call_model", lambda s: {
        "iteration": s["iterations"] + 1, "tools": s["use_tools"],
        "intent": s["intent"].name if s["intent"] else None})
    async def call_model(state: AgentState) -> dict:
        emit = emitter()
        conv = state["conv"]
        check_history_size(conv.messages)
        intent = state["intent"]
        must_look_up = (state["use_tools"] and bool(definitions) and state["iterations"] == 0
                        and intent is not None and intent.name == "account"
                        and not intent.fallback)
        completed: Completed | None = None
        async for event in gateway.stream(
                route, system=AGENT_SYSTEM_PROMPT, messages=conv.messages,
                tools=definitions if state["use_tools"] else None, conversation_id=conv.id,
                system_suffix=state["suffix"],
                tool_choice=REQUIRE_TOOL if must_look_up else None):
            if isinstance(event, TextDelta):
                await emit(AgentText(event.text))
            else:
                completed = event
        assert completed is not None
        message = completed.message
        await record(state, completed, route.name)
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

    @traced_node("run_tools", lambda s: {"tools": [b.name for b in s["tool_uses"]]})
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

    async def may_offer(state: AgentState) -> bool:
        """Offer an advisor at most once per stretch of insistence."""
        intent, conv = state["intent"], state["conv"]
        if intent is None or not intent.insistence:
            return False
        latest = await handoffs.latest(conv.id)
        if latest is None or latest.status == "closed":
            return True
        if latest.status == "declined":
            return customer_turns_since(conv.messages, latest.message_index) > REOFFER_AFTER_TURNS
        return False  # already offered and unanswered (or open)

    @traced_node("finish", lambda s: {"outcome": s["outcome"] or "max_iterations"})
    async def finish(state: AgentState) -> dict:
        emit = emitter()
        outcome = state["outcome"]
        record_outcome(outcome or "max_iterations", state["final_text"])
        if outcome in ("done", "refused") and await may_offer(state):
            conv = state["conv"]
            offer = await handoffs.create(
                conversation_id=conv.id, user_id=state["user_id"], status="offered",
                reason="insistence", summary=state["intent"].reason,
                message_index=len(conv.messages))
            await emit(HandoffOffered(offer.id))
        if outcome is None:  # stopped by the iteration cap
            await emit(AgentDone("max_iterations", "", state["tool_calls"]))
        else:
            await emit(AgentDone(outcome, state["final_text"], state["tool_calls"]))
        return {}

    def start(state: AgentState) -> str:
        if state["handoff"] is not None and state["handoff"].status == "open":
            return "wait_for_human"
        if state["decision"]:
            return "apply_decision"
        return "quick_intent" if state["quick"] else "classify"

    def after_classify(state: AgentState) -> str:
        return {"human": "handoff", "attack": "refuse_attack"}.get(
            state["intent"].name, "call_model")

    def next_call(state: AgentState) -> str:
        return "call_model" if state["iterations"] < max_iterations else "finish"

    def after_model(state: AgentState) -> str:
        if state["outcome"] is not None:
            return "finish"
        if state["tool_uses"]:
            return "run_tools"
        return next_call(state)  # pause_turn: continue if iterations remain

    graph = StateGraph(AgentState)
    graph.add_node("wait_for_human", wait_for_human)
    graph.add_node("classify", classify)
    graph.add_node("quick_intent", quick_intent)
    graph.add_node("handoff", handoff)
    graph.add_node("refuse_attack", refuse_attack)
    graph.add_node("apply_decision", apply_decision)
    graph.add_node("call_model", call_model)
    graph.add_node("run_tools", run_tools)
    graph.add_node("finish", finish)
    graph.add_conditional_edges(START, start, ["wait_for_human", "apply_decision",
                                               "quick_intent", "classify"])
    graph.add_edge("wait_for_human", END)
    graph.add_edge("quick_intent", "call_model")
    graph.add_conditional_edges("classify", after_classify,
                                ["handoff", "refuse_attack", "call_model"])
    graph.add_edge("handoff", END)
    graph.add_edge("refuse_attack", END)
    graph.add_edge("apply_decision", "call_model")
    graph.add_conditional_edges("call_model", after_model, ["finish", "run_tools", "call_model"])
    graph.add_conditional_edges("run_tools", next_call, ["call_model", "finish"])
    graph.add_edge("finish", END)
    return graph.compile(name="agent")
