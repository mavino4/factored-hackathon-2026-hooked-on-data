"""A successful sign-in emails the notify list and never includes the password."""

import smtplib
from email.message import EmailMessage

from fastapi.testclient import TestClient

from aiplatform.api.app import create_app
from aiplatform.config import Settings
from aiplatform.llm.gateway import AIGateway
from aiplatform.mail import send_login_notice
from tests.fakes import FakeClient
from tests.test_password_auth import PASSWORD, add_user, sign_in

SENT: list[EmailMessage] = []


class FakeSMTP:
    def __init__(self, host, port, timeout=None):
        self.host, self.port = host, port

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def ehlo(self):
        return None

    def starttls(self):
        return None

    def login(self, username, password):
        return None

    def send_message(self, message):
        SENT.append(message)


class DownSMTP:
    def __init__(self, *args, **kwargs):
        raise OSError("smtp down")


def test_notice_names_the_user_and_both_addresses():
    SENT.clear()
    settings = Settings(
        auth_mode="password", smtp_host="smtp.example", smtp_from="bot@example.com",
        smtp_username="mailer", smtp_password="secret")
    send_login_notice(settings, username="ana", ip="203.0.113.8", smtp_factory=FakeSMTP)
    assert len(SENT) == 1
    body = SENT[0].get_content()
    assert "ana" in body and "203.0.113.8" in body and PASSWORD not in body
    assert SENT[0]["To"] == "trinogutz@gmail.com, marco.antonio.vino@gmail.com"


def test_a_wrong_password_does_not_send(monkeypatch):
    SENT.clear()
    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)
    settings = Settings(
        providers=["anthropic"], auth_mode="password", login_requests_per_minute=100,
        smtp_host="smtp.example", smtp_from="bot@example.com")
    with TestClient(create_app(
            settings, gateway=AIGateway({"anthropic": FakeClient()}, settings))) as http:
        add_user(http, "ana")
        assert sign_in(http, "ana", "nope").status_code == 401
    assert SENT == []


def test_sign_in_still_succeeds_when_smtp_is_down(monkeypatch):
    monkeypatch.setattr(smtplib, "SMTP", DownSMTP)
    settings = Settings(
        providers=["anthropic"], auth_mode="password", login_requests_per_minute=100,
        smtp_host="smtp.example", smtp_from="bot@example.com")
    with TestClient(create_app(
            settings, gateway=AIGateway({"anthropic": FakeClient()}, settings))) as http:
        add_user(http, "ana")
        res = sign_in(http, "ana")
        assert res.status_code == 200
        assert res.json()["user_id"] == "ana"
