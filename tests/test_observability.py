import asyncio
import json
import logging
from dataclasses import replace

from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from aiplatform.api.app import create_app
from aiplatform.config import Settings
from aiplatform.llm.gateway import AIGateway, Completed
from aiplatform.llm.models import ROUTES
from aiplatform.logging import JsonFormatter, request_id
from aiplatform.storage.sql import create_engine
from aiplatform.storage.tables import metadata
from tests.fakes import FakeClient, Stall, text_reply

U1 = {"X-User-Id": "u1"}


def sample(name, **labels):
    return REGISTRY.get_sample_value(name, labels) or 0.0


async def test_stalled_provider_times_out_and_is_retried():
    client = FakeClient(Stall(5, text_reply("late")), text_reply("fast"))
    gw = AIGateway({"anthropic": client}, Settings(providers=["anthropic"]),
                   sleep=lambda _: asyncio.sleep(0))
    route = replace(ROUTES["chat"], first_event_timeout_s=0.05)
    before = sample("aip_llm_errors_total", provider="anthropic", kind="timeout")
    events = [e async for e in gw.stream(route, system="s",
                                         messages=[{"role": "user", "content": "hi"}])]
    assert isinstance(events[-1], Completed) and len(client.calls) == 2
    assert sample("aip_llm_errors_total", provider="anthropic", kind="timeout") == before + 1


async def test_usage_metrics_and_cost():
    labels = {"route": "chat", "provider": "anthropic", "model": "claude-haiku-4-5"}
    before_out = sample("aip_llm_tokens_total", kind="output", **labels)
    before_cost = sample("aip_llm_cost_usd_total", **labels)
    before_ttft = sample("aip_llm_time_to_first_token_seconds_count",
                         route="chat", provider="anthropic")
    gw = AIGateway({"anthropic": FakeClient(text_reply("hi"))}, Settings(providers=["anthropic"]))
    await gw.complete(ROUTES["chat"], system="s", messages=[{"role": "user", "content": "x"}])
    assert sample("aip_llm_tokens_total", kind="output", **labels) == before_out + 5
    # Haiku 4.5: 10 input x $1 + 5 output x $5 per million tokens.
    assert abs(sample("aip_llm_cost_usd_total", **labels) - before_cost - 35e-6) < 1e-12
    assert sample("aip_llm_time_to_first_token_seconds_count",
                  route="chat", provider="anthropic") == before_ttft + 1


def client_for(fake=None, **settings):
    s = Settings(providers=["anthropic"], max_attempts_per_provider=1, **settings)
    return TestClient(create_app(s, gateway=AIGateway({"anthropic": fake or FakeClient()}, s)))


def test_request_id_is_echoed_or_generated_and_http_metrics_use_route_templates():
    with client_for() as http:
        assert http.get("/healthz", headers={"X-Request-ID": "abc-123"}).headers[
            "x-request-id"] == "abc-123"
        generated = http.get("/healthz", headers={"X-Request-ID": "bad id\nwith newline"})
        assert len(generated.headers["x-request-id"]) == 32
        http.get("/v1/conversations/some-id", headers=U1)
    assert sample("aip_http_requests_total", method="GET",
                  route="/v1/conversations/{conversation_id}", status="404") >= 1


def test_log_lines_carry_the_request_id():
    token = request_id.set("req-42")
    try:
        record = logging.makeLogRecord({"msg": "hello", "levelname": "INFO", "name": "x"})
        assert json.loads(JsonFormatter().format(record))["request_id"] == "req-42"
    finally:
        request_id.reset(token)


def test_readyz(tmp_path):
    with client_for() as http:  # in-memory storage
        assert http.get("/readyz").json() == {"status": "ok"}
    with client_for(database_url=f"sqlite+aiosqlite:///{tmp_path}/r.db") as http:
        assert http.get("/readyz").status_code == 200
    with client_for(database_url="postgresql+asyncpg://x:y@127.0.0.1:1/none") as http:
        resp = http.get("/readyz")
        assert resp.status_code == 503 and resp.json()["database"] == "down"
        assert http.get("/healthz").status_code == 200  # liveness doesn't depend on the DB


def test_admin_usage_report(tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path}/u.db"

    async def setup():
        engine = create_engine(url)
        async with engine.begin() as conn:
            await conn.run_sync(metadata.create_all)
        await engine.dispose()

    asyncio.run(setup())
    fake = FakeClient(text_reply("a"), text_reply("b"))
    with client_for(fake, database_url=url, admin_users=["boss"]) as http:
        cid = http.post("/v1/conversations", json={"kind": "chat"}, headers=U1).json()["id"]
        for text in ("one", "two"):
            http.post(f"/v1/conversations/{cid}/messages", json={"text": text}, headers=U1)
        assert http.get("/v1/admin/usage", headers=U1).status_code == 403
        report = http.get("/v1/admin/usage?days=1", headers={"X-User-Id": "boss"}).json()
    [row] = report["rows"]
    assert row["requests"] == 2 and row["model"] == "claude-haiku-4-5"
    assert row["input_tokens"] == 20 and row["output_tokens"] == 10
    assert report["total_cost_usd"] == 70e-6


def test_admin_usage_needs_a_database():
    with client_for(admin_users=["boss"]) as http:
        assert http.get("/v1/admin/usage", headers={"X-User-Id": "boss"}).status_code == 501
