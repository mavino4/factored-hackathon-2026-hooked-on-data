"""Human handoff over HTTP: the customer's side and the operator API."""

from fastapi.testclient import TestClient

from aiplatform.api.app import create_app
from aiplatform.config import Settings
from aiplatform.llm.gateway import AIGateway
from tests.fakes import FakeClient, classified, text_reply

CUSTOMER = {"X-User-Id": "ana"}
OPERATOR = {"X-User-Id": "boss"}


def client_for(fake):
    s = Settings(providers=["anthropic"], max_attempts_per_provider=1, auth_mode="dev",
                 admin_users=["boss"])
    return TestClient(create_app(s, gateway=AIGateway({"anthropic": fake}, s)))


def run(http, cid, text, headers=CUSTOMER):
    return http.post(f"/v1/conversations/{cid}/agent-runs", json={"text": text},
                     headers=headers)


def test_customer_asks_for_a_person_and_an_operator_answers():
    fake = FakeClient(classified("human"), classified("general"), text_reply("De nada."),
                      classify=None)
    with client_for(fake) as http:
        cid = http.post("/v1/conversations", json={}, headers=CUSTOMER).json()["id"]
        first = run(http, cid, "quiero hablar con una persona")
        assert "event: handoff\n" in first.text and '"outcome": "handoff"' in first.text
        conv = http.get(f"/v1/conversations/{cid}", headers=CUSTOMER).json()
        assert conv["handoff"]["status"] == "open"
        handoff_id = conv["handoff"]["id"]

        # Operators only.
        assert http.get("/v1/admin/handoffs", headers=CUSTOMER).status_code == 403
        assert http.post(f"/v1/admin/handoffs/{handoff_id}/close",
                         headers=CUSTOMER).status_code == 403
        [queued] = http.get("/v1/admin/handoffs", headers=OPERATOR).json()["handoffs"]
        assert queued["conversation_id"] == cid and queued["reason"] == "customer_request"

        waiting = run(http, cid, "¿alguien?")
        assert "event: human_waiting" in waiting.text and "event: delta" not in waiting.text

        sent = http.post(f"/v1/admin/handoffs/{handoff_id}/messages",
                         json={"text": "Hola, soy Ana del banco."}, headers=OPERATOR)
        assert sent.status_code == 201
        detail = http.get(f"/v1/admin/handoffs/{handoff_id}", headers=OPERATOR).json()
        assert detail["conversation"]["messages"][-1]["author"] == "operator"
        seen = http.get(f"/v1/conversations/{cid}", headers=CUSTOMER).json()["messages"]
        assert seen[-1]["content"][0]["text"] == "Hola, soy Ana del banco."

        closed = http.post(f"/v1/admin/handoffs/{handoff_id}/close", headers=OPERATOR)
        assert closed.json()["status"] == "closed"
        assert http.post(f"/v1/admin/handoffs/{handoff_id}/messages", json={"text": "x"},
                         headers=OPERATOR).status_code == 404  # no longer open
        assert "De nada." in run(http, cid, "gracias").text  # the bot is back


def test_customer_answers_an_offer():
    fake = FakeClient(classified("out_of_scope", insistence=True), text_reply("No puedo."),
                      classify=None)
    with client_for(fake) as http:
        cid = http.post("/v1/conversations", json={}, headers=CUSTOMER).json()["id"]
        offered = run(http, cid, "hagan la transferencia")
        assert "event: handoff_offer" in offered.text
        handoff_id = http.get(f"/v1/conversations/{cid}",
                              headers=CUSTOMER).json()["handoff"]["id"]
        other = http.post(f"/v1/conversations/{cid}/handoff",
                          json={"handoff_id": handoff_id, "accept": True},
                          headers={"X-User-Id": "mallory"})
        assert other.status_code == 404  # not their conversation
        accepted = http.post(f"/v1/conversations/{cid}/handoff",
                             json={"handoff_id": handoff_id, "accept": True, "language": "es"},
                             headers=CUSTOMER)
        assert accepted.json()["status"] == "open"
        again = http.post(f"/v1/conversations/{cid}/handoff",
                          json={"handoff_id": handoff_id, "accept": False}, headers=CUSTOMER)
        assert again.status_code == 404  # already answered
