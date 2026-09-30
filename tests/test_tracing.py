import json

import pytest
from langfuse import Langfuse
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import ValidationError

from aiplatform import privacy
from aiplatform.agent.actions import InMemoryActionStore
from aiplatform.agent.loop import AgentRunner
from aiplatform.agent.tools import Tool
from aiplatform.chat.inflight import InFlight
from aiplatform.chat.repository import InMemoryConversationRepository
from aiplatform.chat.service import ChatService
from aiplatform.config import Settings
from aiplatform.llm.gateway import AIGateway
from aiplatform.tracing import Tracing
from aiplatform.usage import InMemoryUsageStore
from tests.fakes import FakeClient, text_reply
from tests.test_agent import collect, final, tool_call

USER = "auth0|customer-42"
PROFILE = {"status": "ok", "first_name": "María", "city": "Córdoba",
           "products": [{"type": "Cuenta Ahorro", "number_last4": "4821", "currency": "ARS",
                         "balance": 1234.56}]}
EMPTY = {"type": "object", "properties": {}}


async def _profile(args, ctx):
    return PROFILE


async def _broken(args, ctx):
    raise RuntimeError("bank down")


TOOLS = [Tool("get_customer_profile", "d", EMPTY, _profile),
         Tool("get_products", "d", EMPTY, _broken)]


@pytest.fixture
def traced(request):
    """A Tracing that exports to memory instead of a Langfuse server."""
    exporter = InMemorySpanExporter()
    masker = privacy.Masker(b"test-key")
    key = f"pk-{request.node.name}"  # Langfuse keeps one client per public key
    client = Langfuse(public_key=key, secret_key="sk", base_url="http://langfuse.invalid",
                      span_exporter=exporter, mask=masker.mask)

    def spans() -> list:
        client.flush()
        return list(exporter.get_finished_spans())

    return Tracing(client, masker, key), masker, spans


def settings() -> Settings:
    return Settings(providers=["anthropic"], auth_mode="dev")


def by_name(spans) -> dict:
    return {s.name: s for s in spans}


def everything(spans) -> str:
    return json.dumps([dict(s.attributes) for s in spans], ensure_ascii=False, default=str)


async def agent_run(tracing, *replies, text="hola, mi email es maria.p@gmail.com"):
    gw = AIGateway({"anthropic": FakeClient(*replies)}, settings())
    repo = InMemoryConversationRepository()
    conv = await repo.create(USER, "agent")
    agent = AgentRunner(gw, repo, InMemoryUsageStore(), InFlight(), InMemoryActionStore(),
                        TOOLS, tracing=tracing)
    await collect(agent.run(USER, conv.id, text))
    return conv


async def test_agent_run_is_one_masked_trace(traced):
    tracing, masker, spans = traced
    conv = await agent_run(tracing, tool_call("get_customer_profile", {}),
                           final("Hola María, su saldo en la cuenta 4821 es 1.234,56 ARS."))
    sent = spans()
    assert len({s.context.trace_id for s in sent}) == 1
    root = by_name(sent)["agent"]
    assert root.parent is None
    assert root.attributes["session.id"] == conv.id
    assert root.attributes["user.id"] == masker.pseudonym(USER)
    generations = [s for s in sent if s.name == "agent.anthropic"]
    assert len(generations) == 2
    assert all(s.attributes["langfuse.observation.type"] == "generation" for s in generations)
    assert "cache_read_input_tokens" in generations[0].attributes[
        "langfuse.observation.usage_details"]
    assert json.loads(generations[0].attributes["langfuse.observation.cost_details"])["total"] > 0
    tool = by_name(sent)["get_customer_profile"]
    assert tool.attributes["langfuse.observation.type"] == "tool"
    # The name, city, last 4, balance, email and user ID never leave the app, including
    # the model's reply, which only repeats them.
    dump = everything(sent)
    for secret in ("María", "Córdoba", "4821", "1234.56", "1.234,56", "maria.p@gmail.com",
                   USER):
        assert secret not in dump
    reply = json.loads(generations[1].attributes["langfuse.observation.output"])
    assert reply["content"][0]["text"] == "Hola <NAME>, su saldo en la cuenta <LAST4> es <AMOUNT> ARS."


async def test_failed_tool_is_an_error(traced):
    tracing, _, spans = traced
    await agent_run(tracing, tool_call("get_products", {}), final("Lo siento."))
    tool = by_name(spans())["get_products"]
    assert tool.attributes["langfuse.observation.level"] == "ERROR"
    assert tool.attributes["langfuse.observation.status_message"] == "tool failed: RuntimeError"


async def test_chat_run_is_traced(traced):
    tracing, _, spans = traced
    gw = AIGateway({"anthropic": FakeClient(text_reply("Hola"))}, settings())
    repo = InMemoryConversationRepository()
    conv = await repo.create(USER, "chat")
    chat = ChatService(gw, repo, InMemoryUsageStore(), InFlight(), tracing)
    await collect(chat.send(USER, conv.id, "hi"))
    sent = by_name(spans())
    assert sent["chat"].attributes["session.id"] == conv.id
    assert sent["chat.anthropic"].context.trace_id == sent["chat"].context.trace_id


async def test_runs_do_not_share_sensitive_values(traced):
    tracing, _, spans = traced
    await agent_run(tracing, tool_call("get_customer_profile", {}), final("Hola María."))
    spans()
    # Outside a run, "María" is only a word: nothing collected earlier leaks into masking.
    assert privacy.Masker(b"k").mask(data="Hola María") == "Hola María"


ON = {"LANGFUSE_TRACING_ENABLED": True}  # settings with an env alias are set by that name


def test_off_by_default():
    assert not Tracing.from_settings(settings()).enabled
    no_keys = Settings(providers=["anthropic"], auth_mode="dev", **ON)
    assert no_keys.langfuse_enabled and not Tracing.from_settings(no_keys).enabled


def test_production_tracing_needs_a_hash_key():
    with pytest.raises(ValidationError, match="AIP_TRACE_HASH_KEY"):
        Settings(env="production", auth_mode="oidc", oidc_issuer="https://i/",
                 oidc_audience="a", **ON)
