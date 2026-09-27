import pytest

from aiplatform.config import Settings
from aiplatform.llm.gateway import (
    AIGateway,
    Completed,
    ModelUnavailable,
    StreamInterrupted,
    TextDelta,
    build_params,
)
from aiplatform.llm.models import OPUS_5, ROUTES, Route
from aiplatform.llm.resilience import CircuitBreaker
from tests.fakes import FakeClient, MidStreamFailure, status_error, text_reply

CHAT = ROUTES["chat"]


async def no_sleep(_):
    pass


def gateway(clients, providers=None):
    settings = Settings(providers=providers or list(clients), max_attempts_per_provider=3)
    return AIGateway(clients, settings, sleep=no_sleep)


async def collect(gw, **kw):
    return [e async for e in gw.stream(CHAT, system="sys", messages=[
        {"role": "user", "content": "hi"}], **kw)]


async def test_streams_deltas_then_completed():
    client = FakeClient(text_reply("hello"))
    events = await collect(gateway({"anthropic": client}))
    assert events[0] == TextDelta("hello")
    assert isinstance(events[-1], Completed) and events[-1].provider == "anthropic"


def test_params_set_cache_breakpoints_without_mutating_history():
    history = [{"role": "user", "content": "hi"}]
    params = build_params(CHAT, "anthropic", system="sys", messages=history, tools=None)
    assert params["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert params["messages"][-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert history == [{"role": "user", "content": "hi"}]


def test_light_model_sends_no_thinking_or_effort():
    params = build_params(CHAT, "anthropic", system="s", messages=[], tools=None)
    assert params["model"] == "claude-haiku-4-5"
    assert "thinking" not in params and "output_config" not in params


def test_larger_model_gets_adaptive_thinking_and_effort():
    route = Route("x", OPUS_5, max_tokens=1000, effort="high")
    params = build_params(route, "anthropic", system="s", messages=[], tools=None)
    assert params["thinking"] == {"type": "adaptive"}
    assert params["output_config"] == {"effort": "high"}


def test_tools_sorted_and_provider_model_ids():
    tools = [{"name": "b"}, {"name": "a"}]
    params = build_params(CHAT, "vertex", system="s", messages=[], tools=tools)
    assert [t["name"] for t in params["tools"]] == ["a", "b"]
    assert params["model"] == "claude-haiku-4-5@20251001"


async def test_retries_transient_errors():
    client = FakeClient(status_error(429), status_error(500), text_reply("ok"))
    events = await collect(gateway({"anthropic": client}))
    assert isinstance(events[-1], Completed)
    assert len(client.calls) == 3


async def test_fails_over_and_sticks_to_the_serving_provider():
    primary = FakeClient(status_error(500), status_error(500), status_error(500), text_reply("x"))
    secondary = FakeClient(text_reply("from bedrock"), text_reply("again"))
    gw = gateway({"anthropic": primary, "bedrock": secondary})
    events = await collect(gw, conversation_id="c1")
    assert events[-1].provider == "bedrock"
    events = await collect(gw, conversation_id="c1")
    assert events[-1].provider == "bedrock"  # sticky: keeps its prompt cache warm
    assert len(primary.calls) == 3


async def test_auth_error_fails_over_immediately():
    primary = FakeClient(status_error(401))
    secondary = FakeClient(text_reply("ok"))
    events = await collect(gateway({"anthropic": primary, "vertex": secondary}))
    assert events[-1].provider == "vertex" and len(primary.calls) == 1


async def test_bad_request_is_not_retried():
    client = FakeClient(status_error(400))
    with pytest.raises(Exception) as info:
        await collect(gateway({"anthropic": client, "bedrock": FakeClient()}))
    assert getattr(info.value, "status_code", None) == 400
    assert len(client.calls) == 1


async def test_failure_after_text_is_not_silently_retried():
    client = FakeClient(MidStreamFailure(["partial"], status_error(500)))
    with pytest.raises(StreamInterrupted):
        await collect(gateway({"anthropic": client}))


async def test_all_providers_down():
    client = FakeClient(*[status_error(500)] * 3)
    with pytest.raises(ModelUnavailable):
        await collect(gateway({"anthropic": client}))


def test_breaker_allows_a_single_half_open_probe():
    now = [0.0]
    breaker = CircuitBreaker(threshold=1, cooldown_s=10, clock=lambda: now[0])
    breaker.record_failure()
    assert not breaker.acquire()
    now[0] = 11
    assert breaker.acquire()          # the probe
    assert not breaker.acquire()      # everyone else waits for its result
    breaker.release()
    breaker.record_success()
    assert breaker.acquire() and breaker.acquire()


async def test_sticky_map_is_bounded():
    client = FakeClient(*[text_reply("x") for _ in range(5)])
    gw = AIGateway({"anthropic": client}, Settings(providers=["anthropic"]),
                   sleep=no_sleep, sticky_max=2)
    for i in range(5):
        await collect(gw, conversation_id=f"c{i}")
    assert list(gw._sticky) == ["c3", "c4"]
