import pytest

from aiplatform.agent.actions import ActionNotPending, InMemoryActionStore
from aiplatform.agent.loop import (
    MAX_TOOL_RESULT_CHARS,
    AgentDone,
    AgentRunner,
    AgentText,
    ApprovalRequired,
    ToolCall,
    ToolResult,
)
from aiplatform.agent.tools import Tool, ToolContext
from aiplatform.chat.inflight import ConversationBusy, InFlight
from aiplatform.chat.repository import InMemoryConversationRepository
from aiplatform.config import Settings
from aiplatform.llm.gateway import AIGateway
from aiplatform.usage import InMemoryUsageStore
from tests.fakes import FakeClient, make_message

TICKET = {"title": "Printer broken", "details": "Paper jam on floor 2"}
TICKET_SCHEMA = {"type": "object",
                 "properties": {"title": {"type": "string"}, "details": {"type": "string"}},
                 "required": ["title", "details"], "additionalProperties": False}
SEEN_CONTEXTS: list[ToolContext] = []


async def _current_time(args, ctx):
    SEEN_CONTEXTS.append(ctx)
    return "12:00"


async def _ticket(args, ctx):
    return f"Ticket created: {args['title']}"


# Test tools: one read-only, one irreversible (goes through approval).
DEFAULT_TOOLS = [
    Tool("create_support_ticket", "d", TICKET_SCHEMA, _ticket, irreversible=True),
    Tool("get_current_time", "d", {"type": "object", "properties": {}}, _current_time),
]


def tool_call(name, args, id="tu_1"):
    return [], make_message({"type": "tool_use", "id": id, "name": name, "input": args},
                            stop_reason="tool_use")


def final(text):
    return [text], make_message({"type": "text", "text": text})


async def runner(client, tools=DEFAULT_TOOLS, inflight=None):
    gw = AIGateway({"anthropic": client}, Settings(providers=["anthropic"]))
    repo = InMemoryConversationRepository()
    actions = InMemoryActionStore()
    conv = await repo.create("u1", "agent")
    agent = AgentRunner(gw, repo, InMemoryUsageStore(), inflight or InFlight(), actions, tools)
    return agent, repo, actions, conv


async def collect(stream):
    return [e async for e in stream]


def last_tool_results(conv):
    return [m for m in conv.messages if m["role"] == "user"][-1]["content"]


async def test_runs_tool_and_streams_events():
    client = FakeClient(tool_call("get_current_time", {}), final("It is noon."))
    agent, _, _, conv = await runner(client)
    events = await collect(agent.run("u1", conv.id, "what time is it?"))
    kinds = [type(e) for e in events]
    assert kinds == [ToolCall, ToolResult, AgentText, AgentDone]
    assert events[-1] == AgentDone("done", "It is noon.", ["get_current_time"])
    # The tool got the session's identity from the server, not from the model.
    assert SEEN_CONTEXTS[-1] == ToolContext(user_id="u1", conversation_id=conv.id)
    assert "is_error" not in last_tool_results(conv)[0]
    # Tools were sent on every call, in a stable order.
    assert [t["name"] for t in client.calls[0]["tools"]] == [
        "create_support_ticket", "get_current_time"]


async def test_irreversible_tool_becomes_pending_and_runs_only_after_approval():
    ran = []

    async def create_ticket(args, ctx):
        ran.append(args)
        return "Ticket #42 created"

    ticket_tool = Tool("create_support_ticket", "d", TICKET_SCHEMA,
                       create_ticket, irreversible=True)
    client = FakeClient(
        tool_call("create_support_ticket", TICKET),
        final("I'll open a ticket once you approve."),
        final("Done: ticket #42."))
    agent, _, actions, conv = await runner(client, tools=[ticket_tool])

    events = await collect(agent.run("u1", conv.id, "open a ticket"))
    approval = next(e for e in events if isinstance(e, ApprovalRequired))
    assert events[-1].outcome == "approval_required" and ran == []
    assert "NOT been executed" in last_tool_results(conv)[0]["content"]
    assert [a.id for a in await actions.list_pending(conv.id, "u1")] == [approval.action_id]

    events = await collect(agent.decide("u1", conv.id, approval.action_id, approve=True))
    assert ran == [TICKET]
    assert isinstance(events[0], ToolResult) and events[0].content == "Ticket #42 created"
    assert events[-1] == AgentDone("done", "Done: ticket #42.", [])
    decision = conv.messages[-2]["content"]  # user [Approval] message, then assistant reply
    assert decision.startswith("[Approval] The user APPROVED") and "Ticket #42" in decision

    # Deciding twice never runs the tool again.
    with pytest.raises(ActionNotPending):
        await collect(agent.decide("u1", conv.id, approval.action_id, approve=True))
    assert len(ran) == 1


async def test_rejected_action_is_not_executed():
    ran = []

    async def create_ticket(args, ctx):
        ran.append(args)
        return "created"

    ticket_tool = Tool("create_support_ticket", "d", TICKET_SCHEMA,
                       create_ticket, irreversible=True)
    client = FakeClient(tool_call("create_support_ticket", TICKET), final("Waiting."),
                        final("OK, I won't."))
    agent, _, _, conv = await runner(client, tools=[ticket_tool])
    events = await collect(agent.run("u1", conv.id, "open a ticket"))
    action_id = next(e.action_id for e in events if isinstance(e, ApprovalRequired))
    events = await collect(agent.decide("u1", conv.id, action_id, approve=False))
    assert ran == [] and events[-1].text == "OK, I won't."
    assert "REJECTED" in conv.messages[-2]["content"]


async def test_invalid_tool_input_is_reported_to_the_model():
    client = FakeClient(tool_call("create_support_ticket", {"title": 3}), final("sorry"))
    agent, _, actions, conv = await runner(client)
    await collect(agent.run("u1", conv.id, "x"))
    assert "invalid input" in last_tool_results(conv)[0]["content"]
    assert await actions.list_pending(conv.id, "u1") == []


async def test_long_tool_results_are_truncated():
    async def huge(args, ctx):
        return "x" * (MAX_TOOL_RESULT_CHARS + 500)

    tool = Tool("dump", "d", {"type": "object", "properties": {}}, huge)
    agent, _, _, conv = await runner(FakeClient(tool_call("dump", {}), final("ok")), tools=[tool])
    await collect(agent.run("u1", conv.id, "x"))
    content = last_tool_results(conv)[0]["content"]
    assert len(content) < MAX_TOOL_RESULT_CHARS + 100 and content.endswith("too long]")


async def test_stops_at_max_iterations():
    client = FakeClient(*[tool_call("get_current_time", {}, id=f"t{i}") for i in range(8)])
    agent, _, _, conv = await runner(client)
    events = await collect(agent.run("u1", conv.id, "loop"))
    assert events[-1].outcome == "max_iterations"


async def test_busy_conversation_is_rejected():
    inflight = InFlight()
    agent, _, _, conv = await runner(FakeClient(), inflight=inflight)
    with inflight.hold(conv.id), pytest.raises(ConversationBusy):
        await collect(agent.run("u1", conv.id, "x"))
