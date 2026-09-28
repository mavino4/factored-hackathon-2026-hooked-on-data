from fastapi.testclient import TestClient

from aiplatform.api.app import create_app
from aiplatform.config import Settings
from aiplatform.llm.gateway import AIGateway
from tests.fakes import FakeClient, MidStreamFailure, status_error, text_reply


def client_for(fake, tools=None, **settings):
    s = Settings(providers=["anthropic"], max_attempts_per_provider=1, auth_mode="dev",
                 **settings)
    return TestClient(create_app(s, gateway=AIGateway({"anthropic": fake}, s), tools=tools))


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


def test_agent_bad_request_becomes_error_event():
    with client_for(FakeClient(status_error(400))) as http:
        cid = new_conversation(http, kind="agent")
        resp = post(http, f"/v1/conversations/{cid}/agent-runs", json={"text": "x"})
        assert "event: error" in resp.text and "invalid_request" in resp.text


def test_agent_approval_flow_over_http():
    from tests.fakes import make_message
    ticket = {"title": "t", "details": "d"}
    fake = FakeClient(
        ([], make_message({"type": "tool_use", "id": "tu_1", "name": "create_support_ticket",
                           "input": ticket}, stop_reason="tool_use")),
        text_reply("Please approve."),
        text_reply("Ticket opened."))
    from tests.test_agent import DEFAULT_TOOLS as TEST_TOOLS
    with client_for(fake, tools=TEST_TOOLS) as http:
        cid = new_conversation(http, kind="agent")
        run = post(http, f"/v1/conversations/{cid}/agent-runs", json={"text": "open a ticket"})
        assert "event: approval_required" in run.text
        assert '"outcome": "approval_required"' in run.text
        conv = http.get(f"/v1/conversations/{cid}", headers={"X-User-Id": "u1"}).json()
        [action] = conv["pending_actions"]
        assert action["tool_name"] == "create_support_ticket" and action["input"] == ticket

        # Other users can't decide it; unknown ids are 404.
        other = http.post(f"/v1/conversations/{cid}/actions/{action['id']}",
                          json={"decision": "approve"}, headers={"X-User-Id": "mallory"})
        assert other.status_code == 404
        assert post(http, f"/v1/conversations/{cid}/actions/nope",
                    json={"decision": "approve"}).status_code == 404

        done = post(http, f"/v1/conversations/{cid}/actions/{action['id']}",
                    json={"decision": "approve"})
        assert "event: tool_result" in done.text and '"is_error": false' in done.text
        assert "Ticket created: t" not in done.text  # raw tool output stays server-side
        assert "Ticket opened." in done.text
        conv = http.get(f"/v1/conversations/{cid}", headers={"X-User-Id": "u1"}).json()
        assert conv["pending_actions"] == []
        assert post(http, f"/v1/conversations/{cid}/actions/{action['id']}",
                    json={"decision": "approve"}).status_code == 404


def test_me_returns_the_customer_name_from_the_bank():
    from decimal import Decimal

    from aiplatform.banking.repository import CustomerProfile, InMemoryBankRepository

    bank = InMemoryBankRepository(
        customers={"C1": CustomerProfile("Norma", "Colombia", None, None, "Active",
                                         Decimal(1))},
        products={}, logins={"ana": "C1"})
    s = Settings(providers=["anthropic"], auth_mode="dev")
    app = create_app(s, gateway=AIGateway({"anthropic": FakeClient()}, s), bank_repo=bank)
    with TestClient(app) as http:
        assert http.get("/v1/me", headers={"X-User-Id": "ana"}).json() == {
            "user_id": "ana", "first_name": "Norma"}
        assert http.get("/v1/me", headers={"X-User-Id": "bob"}).json() == {
            "user_id": "bob", "first_name": None}  # not linked: greeted without a name
        assert http.get("/v1/me").status_code == 401


def test_agent_stream_hides_tool_arguments_and_results():
    from tests.fakes import make_message
    from tests.test_agent import DEFAULT_TOOLS as TEST_TOOLS
    fake = FakeClient(
        ([], make_message({"type": "tool_use", "id": "tu_1", "name": "get_current_time",
                           "input": {}}, stop_reason="tool_use")),
        text_reply("It is noon."))
    with client_for(fake, tools=TEST_TOOLS) as http:
        cid = new_conversation(http, kind="agent")
        run = post(http, f"/v1/conversations/{cid}/agent-runs", json={"text": "time?"})
    assert 'event: tool_call\ndata: {"id": "tu_1", "name": "get_current_time"}' in run.text
    assert "12:00" not in run.text  # the tool's raw result isn't streamed
    assert "It is noon." in run.text


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


def test_reply_language_is_the_one_the_customer_sees():
    """The UI sends its language; it reaches the model as a system block after the
    cached prompt (so the cached prefix stays byte-identical)."""
    from tests.fakes import make_message
    from tests.test_agent import DEFAULT_TOOLS as TEST_TOOLS
    fake = FakeClient(text_reply("Olá!"), text_reply("Oi!"),
                      ([], make_message({"type": "text", "text": "Hola"})))
    with client_for(fake, tools=TEST_TOOLS) as http:
        cid = new_conversation(http)
        post(http, f"/v1/conversations/{cid}/messages", json={"text": "hola", "language": "pt"})
        post(http, f"/v1/conversations/{cid}/messages", json={"text": "hola"})
        assert post(http, f"/v1/conversations/{cid}/messages",
                    json={"text": "x", "language": "fr"}).status_code == 422  # not sent
        agent = new_conversation(http, kind="agent")
        post(http, f"/v1/conversations/{agent}/agent-runs", json={"text": "saldo", "language": "es"})
    with_pt, without, agent_es = [call["system"] for call in fake.calls]
    assert with_pt[0] == without[0] and "cache_control" in with_pt[0]
    assert "Brazilian Portuguese" in with_pt[1]["text"] and "cache_control" not in with_pt[1]
    assert len(without) == 1
    assert "Reply language: Spanish" in agent_es[1]["text"]
