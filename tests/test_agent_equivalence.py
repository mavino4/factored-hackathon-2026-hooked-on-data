"""Exact behavior of the agent's tool loop: event sequences, stored history, usage,
errors and cancellation. Written against the hand-rolled loop before the LangGraph
migration, so the graph must reproduce it event for event. (The intent classification
that runs first is answered by FakeClient and tested in test_intent.py.)"""

import asyncio

import pytest

from aiplatform.agent.actions import InMemoryActionStore
from aiplatform.agent.loop import (
    AgentDone,
    AgentRunner,
    AgentText,
    ApprovalRequired,
    ToolCall,
    ToolResult,
)
from aiplatform.agent.tools import Tool
from aiplatform.chat.history import MAX_HISTORY_CHARS, ConversationTooLong
from aiplatform.chat.inflight import InFlight
from aiplatform.chat.repository import InMemoryConversationRepository
from aiplatform.config import Settings
from aiplatform.llm.gateway import AIGateway, ModelUnavailable
from aiplatform.usage import InMemoryUsageStore
from tests.fakes import FakeClient, make_message, status_error
from tests.test_agent import TICKET, TICKET_SCHEMA, collect, final, tool_call

EMPTY = {"type": "object", "properties": {}}


async def _no_sleep(_):
    pass


def gateway(client):
    return AIGateway({"anthropic": client},
                     Settings(providers=["anthropic"], max_attempts_per_provider=1),
                     sleep=_no_sleep)


async def setup(client, tools, *, max_iterations=8):
    repo, usage, inflight = InMemoryConversationRepository(), InMemoryUsageStore(), InFlight()
    actions = InMemoryActionStore()
    conv = await repo.create("u1", "agent")
    agent = AgentRunner(gateway(client), repo, usage, inflight, actions, tools,
                        max_iterations=max_iterations)
    return agent, conv, usage, inflight, actions


def tool_uses(*calls):
    """One assistant message asking for several tools at once."""
    blocks = [{"type": "tool_use", "id": id, "name": name, "input": args}
              for id, name, args in calls]
    return [], make_message(*blocks, stop_reason="tool_use")


def roles(conv):
    return [m["role"] for m in conv.messages]


# --- Agent -------------------------------------------------------------------

async def test_tool_run_exact_events_history_and_usage():
    async def now(args, ctx):
        return {"time": "12:00"}  # non-string results are sent as JSON

    client = FakeClient(tool_call("now", {}), final("It is noon."))
    agent, conv, usage, _, _ = await setup(client, [Tool("now", "d", EMPTY, now)])
    events = await collect(agent.run("u1", conv.id, "time?", "es"))

    assert events == [ToolCall("tu_1", "now", {}),
                      ToolResult("tu_1", "now", False, '{"time": "12:00"}'),
                      AgentText("It is noon."),
                      AgentDone("done", "It is noon.", ["now"])]
    assert roles(conv) == ["user", "assistant", "user", "assistant"]
    assert conv.messages[0] == {"role": "user", "content": "time?"}
    assert conv.messages[1]["content"][0]["type"] == "tool_use"
    assert conv.messages[2]["content"] == [
        {"type": "tool_result", "tool_use_id": "tu_1", "content": '{"time": "12:00"}'}]
    # One usage event per model call (10 in + 5 out each).
    # 2 agent calls + the intent classification, 15 tokens each.
    assert await usage.tokens_used_today("u1") == 45
    # The reply language reaches every model call, after the cached system prompt.
    for call in client.calls:
        assert call["system"][1]["text"].startswith("Reply language: Spanish")
        assert [t["name"] for t in call["tools"]] == ["now"]


async def test_refusal_is_not_stored():
    refusal = (["I can't help"], make_message({"type": "text", "text": "I can't help"},
                                              stop_reason="refusal"))
    agent, conv, usage, _, _ = await setup(FakeClient(refusal), [])
    events = await collect(agent.run("u1", conv.id, "x"))
    assert events == [AgentText("I can't help"), AgentDone("refused", "I can't help", [])]
    assert roles(conv) == ["user"]
    assert await usage.tokens_used_today("u1") == 30  # the agent call + classification


async def test_truncated_tool_call_does_not_run_and_is_not_stored():
    ran = []

    async def record(args, ctx):
        ran.append(args)
        return "ok"

    truncated = ([], make_message({"type": "tool_use", "id": "tu_1", "name": "rec", "input": {}},
                                  stop_reason="max_tokens"))
    agent, conv, _, _, _ = await setup(FakeClient(truncated), [Tool("rec", "d", EMPTY, record)])
    events = await collect(agent.run("u1", conv.id, "x"))
    assert events == [AgentDone("truncated", "", [])]
    assert ran == [] and roles(conv) == ["user"]


async def test_pause_turn_continues_and_counts_as_an_iteration():
    paused = (["a"], make_message({"type": "text", "text": "a"}, stop_reason="pause_turn"))
    agent, conv, _, _, _ = await setup(FakeClient(paused, final("b")), [])
    events = await collect(agent.run("u1", conv.id, "x"))
    assert events == [AgentText("a"), AgentText("b"), AgentDone("done", "b", [])]
    assert roles(conv) == ["user", "assistant", "assistant"]

    agent, conv, _, _, _ = await setup(FakeClient(paused, paused), [], max_iterations=2)
    events = await collect(agent.run("u1", conv.id, "x"))
    assert events == [AgentText("a"), AgentText("a"), AgentDone("max_iterations", "", [])]


async def test_max_iterations_exact():
    async def now(args, ctx):
        return "12:00"

    client = FakeClient(*[tool_call("now", {}, id=f"t{i}") for i in range(3)])
    agent, conv, _, _, _ = await setup(client, [Tool("now", "d", EMPTY, now)], max_iterations=3)
    events = await collect(agent.run("u1", conv.id, "loop"))
    assert events[-1] == AgentDone("max_iterations", "", ["now", "now", "now"])
    assert len(client.calls) == 3 and client.outcomes == []


async def test_tools_in_one_message_run_concurrently_and_answer_in_one_message():
    b_started = asyncio.Event()

    async def a(args, ctx):
        await b_started.wait()  # only finishes if b runs at the same time
        return "A"

    async def b(args, ctx):
        b_started.set()
        return "B"

    tools = [Tool("a", "d", EMPTY, a, timeout_s=2), Tool("b", "d", EMPTY, b)]
    client = FakeClient(tool_uses(("t1", "a", {}), ("t2", "b", {})), final("done"))
    agent, conv, _, _, _ = await setup(client, tools)
    events = await collect(agent.run("u1", conv.id, "x"))
    assert events == [ToolCall("t1", "a", {}), ToolCall("t2", "b", {}),
                      ToolResult("t1", "a", False, "A"), ToolResult("t2", "b", False, "B"),
                      AgentText("done"), AgentDone("done", "done", ["a", "b"])]
    assert [r["tool_use_id"] for r in conv.messages[2]["content"]] == ["t1", "t2"]


async def test_tool_errors_are_reported_to_the_model():
    async def slow(args, ctx):
        await asyncio.sleep(5)

    async def boom(args, ctx):
        raise ValueError("secret detail")

    tools = [Tool("slow", "d", EMPTY, slow, timeout_s=0.01), Tool("boom", "d", EMPTY, boom)]
    client = FakeClient(tool_uses(("t1", "nope", {}), ("t2", "slow", {}), ("t3", "boom", {})),
                        final("sorry"))
    agent, conv, _, _, _ = await setup(client, tools)
    events = await collect(agent.run("u1", conv.id, "x"))
    assert [e for e in events if isinstance(e, ToolResult)] == [
        ToolResult("t1", "nope", True, "unknown tool: nope"),
        ToolResult("t2", "slow", True, "tool timed out"),
        ToolResult("t3", "boom", True, "tool failed: ValueError")]
    assert all(r["is_error"] for r in conv.messages[2]["content"])


async def test_approval_exact_events_and_removed_tool():
    async def ticket(args, ctx):
        return "Ticket #42"

    ticket_tool = Tool("create_support_ticket", "d", TICKET_SCHEMA, ticket, irreversible=True)
    client = FakeClient(tool_call("create_support_ticket", TICKET), final("Approve?"),
                        final("Done."), tool_call("create_support_ticket", TICKET),
                        final("Approve?"), final("Gone."))
    agent, conv, _, _, actions = await setup(client, [ticket_tool])

    events = await collect(agent.run("u1", conv.id, "open a ticket"))
    action_id = events[1].action_id
    assert events == [ToolCall("tu_1", "create_support_ticket", TICKET),
                      ApprovalRequired(action_id, "create_support_ticket", TICKET),
                      AgentText("Approve?"),
                      AgentDone("approval_required", "Approve?", ["create_support_ticket"])]
    events = await collect(agent.decide("u1", conv.id, action_id, approve=True))
    assert events == [ToolResult("tu_1", "create_support_ticket", False, "Ticket #42"),
                      AgentText("Done."), AgentDone("done", "Done.", [])]

    # The tool was removed (e.g. a new deploy) between the request and the approval.
    events = await collect(agent.run("u1", conv.id, "another"))
    action_id = events[1].action_id
    other = AgentRunner(agent._gateway, agent._repo, agent._usage, agent._inflight, actions, [])
    events = await collect(other.decide("u1", conv.id, action_id, approve=True))
    assert events == [ToolResult("tu_1", "create_support_ticket", True,
                                 "tool no longer available: create_support_ticket"),
                      AgentText("Gone."), AgentDone("done", "Gone.", [])]
    assert "ERROR: tool no longer available" in conv.messages[-2]["content"]


async def test_provider_outage_propagates_and_frees_the_conversation():
    agent, conv, _, inflight, _ = await setup(FakeClient(status_error(500)), [])
    with pytest.raises(ModelUnavailable):
        await collect(agent.run("u1", conv.id, "x"))
    assert roles(conv) == ["user"] and not inflight.is_busy(conv.id)


async def test_too_long_conversation_is_rejected_before_calling_the_model():
    client = FakeClient()
    agent, conv, _, _, _ = await setup(client, [])
    await agent._repo.append(conv, {"role": "user", "content": "x" * (MAX_HISTORY_CHARS + 1)})
    with pytest.raises(ConversationTooLong):
        await collect(agent.run("u1", conv.id, "x"))
    assert client.calls == [] and len(conv.messages) == 1


async def test_closing_the_stream_frees_the_conversation():
    agent, conv, _, inflight, _ = await setup(FakeClient((["a", "b"], make_message())), [])
    stream = agent.run("u1", conv.id, "x")
    assert await anext(stream) == AgentText("a")
    assert inflight.is_busy(conv.id)
    await stream.aclose()  # the client disconnected
    await asyncio.sleep(0.05)  # nothing keeps running in the background
    assert not inflight.is_busy(conv.id) and roles(conv) == ["user"]
