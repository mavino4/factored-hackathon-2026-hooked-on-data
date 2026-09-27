from fastapi.testclient import TestClient

from aiplatform.api.app import create_app
from aiplatform.config import Settings
from aiplatform.llm.gateway import AIGateway
from tests.fakes import FakeClient, MidStreamFailure, status_error, text_reply


def client_for(fake, **settings):
    s = Settings(providers=["anthropic"], max_attempts_per_provider=1, **settings)
    return TestClient(create_app(s, gateway=AIGateway({"anthropic": fake}, s)))


def new_conversation(http, user="u1", kind="chat"):
    return http.post("/v1/conversations", json={"kind": kind}, headers={"X-User-Id": user}).json()["id"]


def test_chat_streams_sse_and_persists_history():
    with client_for(FakeClient(text_reply("Hello!"))) as http:
        cid = new_conversation(http)
        resp = http.post(f"/v1/conversations/{cid}/messages", json={"text": "hi"},
                         headers={"X-User-Id": "u1"})
        assert resp.headers["content-type"].startswith("text/event-stream")
        assert 'event: delta\ndata: {"text": "Hello!"}' in resp.text
        assert "event: done" in resp.text
        messages = http.get(f"/v1/conversations/{cid}", headers={"X-User-Id": "u1"}).json()["messages"]
        assert [m["role"] for m in messages] == ["user", "assistant"]
        assert messages[1]["content"][0]["text"] == "Hello!"


def test_provider_outage_becomes_sse_error_event():
    with client_for(FakeClient(status_error(500))) as http:
        cid = new_conversation(http)
        resp = http.post(f"/v1/conversations/{cid}/messages", json={"text": "hi"},
                         headers={"X-User-Id": "u1"})
        assert "event: error" in resp.text and "unavailable" in resp.text


def test_users_cannot_read_each_others_conversations():
    with client_for(FakeClient()) as http:
        cid = new_conversation(http, user="alice")
        assert http.get(f"/v1/conversations/{cid}", headers={"X-User-Id": "bob"}).status_code == 404


def test_rate_limit():
    with client_for(FakeClient(*[text_reply("x")] * 3), user_requests_per_minute=2) as http:
        cid = new_conversation(http)
        codes = [http.post(f"/v1/conversations/{cid}/messages", json={"text": "hi"},
                           headers={"X-User-Id": "u1"}).status_code for _ in range(3)]
        assert codes == [200, 200, 429]


def post(http, path, **kw):
    return http.post(path, headers={"X-User-Id": "u1"}, **kw)


def test_bad_request_from_model_api_becomes_error_event():
    with client_for(FakeClient(status_error(400))) as http:
        cid = new_conversation(http)
        resp = post(http, f"/v1/conversations/{cid}/messages", json={"text": "hi"})
        assert "event: error" in resp.text and "invalid_request" in resp.text


def test_unexpected_exception_becomes_error_event():
    with client_for(FakeClient(RuntimeError("boom"))) as http:
        cid = new_conversation(http)
        resp = post(http, f"/v1/conversations/{cid}/messages", json={"text": "hi"})
        assert "event: error" in resp.text and '"internal"' in resp.text


def test_second_request_while_replying_gets_409():
    with client_for(FakeClient()) as http:
        cid = new_conversation(http)
        http.app.state.inflight._busy.add(cid)  # a reply is streaming
        assert post(http, f"/v1/conversations/{cid}/messages", json={"text": "hi"}).status_code == 409


def test_regenerate_after_interrupted_stream_does_not_duplicate_user_message():
    fake = FakeClient(MidStreamFailure(["par"], status_error(500)), text_reply("Full answer"))
    with client_for(fake) as http:
        cid = new_conversation(http)
        first = post(http, f"/v1/conversations/{cid}/messages", json={"text": "hi"})
        assert "event: error" in first.text
        again = post(http, f"/v1/conversations/{cid}/regenerate")
        assert "Full answer" in again.text
        messages = http.get(f"/v1/conversations/{cid}", headers={"X-User-Id": "u1"}).json()["messages"]
        assert [m["role"] for m in messages] == ["user", "assistant"]
        # Nothing left to regenerate now.
        assert post(http, f"/v1/conversations/{cid}/regenerate").status_code == 409


def test_agent_bad_request_maps_to_422():
    with client_for(FakeClient(status_error(400))) as http:
        cid = new_conversation(http, kind="agent")
        resp = post(http, f"/v1/conversations/{cid}/agent-runs", json={"text": "x"})
        assert resp.status_code == 422


def test_two_replicas_share_conversations_through_the_database(tmp_path):
    import asyncio

    from aiplatform.storage.sql import create_engine
    from aiplatform.storage.tables import metadata

    url = f"sqlite+aiosqlite:///{tmp_path}/shared.db"

    async def setup():
        engine = create_engine(url)
        async with engine.begin() as conn:
            await conn.run_sync(metadata.create_all)
        await engine.dispose()

    asyncio.run(setup())
    fake = FakeClient(text_reply("Stored answer"))
    with client_for(fake, database_url=url) as replica_a, \
            client_for(FakeClient(), database_url=url) as replica_b:
        cid = new_conversation(replica_a)
        post(replica_a, f"/v1/conversations/{cid}/messages", json={"text": "remember me"})
        seen_by_b = replica_b.get(f"/v1/conversations/{cid}", headers={"X-User-Id": "u1"}).json()
        assert [m["role"] for m in seen_by_b["messages"]] == ["user", "assistant"]
        listed = replica_b.get("/v1/conversations", headers={"X-User-Id": "u1"}).json()
        assert listed["conversations"][0]["title"] == "remember me"
