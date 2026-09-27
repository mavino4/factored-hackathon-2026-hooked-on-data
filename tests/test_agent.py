import pytest

from aiplatform.agent.loop import APPROVAL_REQUIRED, MAX_TOOL_RESULT_CHARS, AgentRunner
from aiplatform.agent.tools import DEFAULT_TOOLS, Tool
from aiplatform.chat.inflight import ConversationBusy, InFlight
from aiplatform.chat.repository import InMemoryConversationRepository
from aiplatform.config import Settings
from aiplatform.llm.gateway import AIGateway
from aiplatform.ratelimit import DailyTokenQuota
from tests.fakes import FakeClient, make_message


def tool_call(name, args, id="tu_1"):
    return [], make_message({"type": "tool_use", "id": id, "name": name, "input": args},
                            stop_reason="tool_use")


def final(text):
    return [text], make_message({"type": "text", "text": text})


def runner(client, tools=DEFAULT_TOOLS, inflight=None):
    gw = AIGateway({"anthropic": client}, Settings(providers=["anthropic"]))
    repo = InMemoryConversationRepository()
    conv = repo.create("u1", "agent")
    agent = AgentRunner(gw, repo, DailyTokenQuota(10**9), inflight or InFlight(), tools)
    return agent, repo, conv


def last_tool_results(repo, conv):
    return [m for m in conv.messages if m["role"] == "user"][-1]["content"]


async def test_runs_tool_and_returns_final_answer():
    client = FakeClient(tool_call("get_current_time", {}), final("It is noon."))
    agent, repo, conv = runner(client)
    result = await agent.run("u1", conv.id, "what time is it?")
    assert result.outcome == "done" and result.text == "It is noon."
    assert result.tool_calls == ["get_current_time"]
    tool_result = last_tool_results(repo, conv)[0]
    assert tool_result["tool_use_id"] == "tu_1" and "is_error" not in tool_result
    # Tools were sent on every call, in a stable order.
    assert [t["name"] for t in client.calls[0]["tools"]] == [
        "create_support_ticket", "get_current_time"]


async def test_irreversible_tool_needs_approval():
    client = FakeClient(
        tool_call("create_support_ticket", {"title": "t", "details": "d"}), final("Confirm?"))
    agent, repo, conv = runner(client)
    await agent.run("u1", conv.id, "open a ticket")
    tool_result = last_tool_results(repo, conv)[0]
    assert tool_result["is_error"] and tool_result["content"] == APPROVAL_REQUIRED


async def test_invalid_tool_input_is_reported_to_the_model():
    client = FakeClient(tool_call("create_support_ticket", {"title": 3}), final("sorry"))
    agent, repo, conv = runner(client)
    await agent.run("u1", conv.id, "x")
    assert "invalid input" in last_tool_results(repo, conv)[0]["content"]


async def test_stops_at_max_iterations():
    client = FakeClient(*[tool_call("get_current_time", {}, id=f"t{i}") for i in range(8)])
    agent, _, conv = runner(client)
    assert (await agent.run("u1", conv.id, "loop")).outcome == "max_iterations"


async def test_long_tool_results_are_truncated():
    async def huge(_):
        return "x" * (MAX_TOOL_RESULT_CHARS + 500)

    tool = Tool("dump", "d", {"type": "object", "properties": {}}, huge)
    agent, repo, conv = runner(FakeClient(tool_call("dump", {}), final("ok")), tools=[tool])
    await agent.run("u1", conv.id, "x")
    content = last_tool_results(repo, conv)[0]["content"]
    assert len(content) < MAX_TOOL_RESULT_CHARS + 100 and content.endswith("too long]")


async def test_busy_conversation_is_rejected():
    inflight = InFlight()
    agent, _, conv = runner(FakeClient(), inflight=inflight)
    with inflight.hold(conv.id), pytest.raises(ConversationBusy):
        await agent.run("u1", conv.id, "x")
