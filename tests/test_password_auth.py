"""The API in password mode: sign-in, the session cookie, sign-out, password change."""

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from aiplatform.api.app import SESSION_COOKIE, create_app
from aiplatform.config import Settings
from aiplatform.llm.gateway import AIGateway
from tests.fakes import FakeClient

PASSWORD = "correct horse battery"


def client(**overrides) -> TestClient:
    s = Settings(providers=["anthropic"], auth_mode="password",
                 **{"login_requests_per_minute": 100, **overrides})
    return TestClient(create_app(s, gateway=AIGateway({"anthropic": FakeClient()}, s)))


def add_user(http, username, customer_id=None, password=PASSWORD):
    http.portal.call(http.app.state.accounts.create_user, username, password, customer_id)


def sign_in(http, username, password=PASSWORD):
    return http.post("/v1/auth/login", json={"username": username, "password": password})


def test_api_needs_a_session_and_ignores_the_dev_header():
    with client() as http:
        add_user(http, "ana")
        assert http.get("/healthz").status_code == 200
        assert http.get("/config.json").json()["auth_mode"] == "password"
        assert http.get("/v1/conversations").status_code == 401
        assert http.get("/v1/conversations", headers={"X-User-Id": "ana"}).status_code == 401
        http.cookies.set(SESSION_COOKIE, "made-up")
        assert http.get("/v1/conversations").status_code == 401


def test_sign_in_sets_a_cookie_scripts_cannot_read():
    with client() as http:
        add_user(http, "ana")
        res = sign_in(http, "ANA ")
        assert res.status_code == 200 and res.json() == {"user_id": "ana"}
        cookie = res.headers["set-cookie"]
        assert "HttpOnly" in cookie and "SameSite=strict" in cookie and "Path=/" in cookie
        assert "Secure" not in cookie  # plain-HTTP local setups; always on in production
        assert PASSWORD not in res.text
        assert http.get("/v1/me").json()["user_id"] == "ana"


def test_every_sign_in_failure_looks_the_same():
    with client(login_max_failures=2) as http:
        add_user(http, "ana")
        wrong = sign_in(http, "ana", "nope")
        unknown = sign_in(http, "nobody", "nope")
        sign_in(http, "ana", "nope")  # second failure: locked
        locked = sign_in(http, "ana")
        assert {r.status_code for r in (wrong, unknown, locked)} == {401}
        assert wrong.json() == unknown.json() == locked.json()
        assert "set-cookie" not in locked.headers


def test_sign_in_attempts_are_rate_limited_per_address():
    with client(login_requests_per_minute=3) as http:
        codes = [sign_in(http, "nobody", "nope").status_code for _ in range(5)]
        assert codes == [401, 401, 401, 429, 429]


def test_sign_out_ends_the_session():
    with client() as http:
        add_user(http, "ana")
        sign_in(http, "ana")
        token = http.cookies[SESSION_COOKIE]
        assert http.post("/v1/auth/logout").status_code == 204
        http.cookies.set(SESSION_COOKIE, token)  # a copied cookie is useless afterwards
        assert http.get("/v1/me").status_code == 401


def test_users_only_see_their_own_conversations():
    with client() as ana, TestClient(ana.app) as bruno:
        add_user(ana, "ana")
        add_user(ana, "bruno")
        sign_in(ana, "ana")
        sign_in(bruno, "bruno")
        cid = ana.post("/v1/conversations", json={}).json()["id"]
        assert ana.get(f"/v1/conversations/{cid}").status_code == 200
        assert bruno.get(f"/v1/conversations/{cid}").status_code == 404


def test_password_change():
    with client() as http:
        add_user(http, "ana")
        sign_in(http, "ana")
        change = {"current_password": PASSWORD, "new_password": "a-brand-new-password"}
        assert http.post("/v1/auth/password",
                         json={**change, "current_password": "nope"}).status_code == 403
        assert http.post("/v1/auth/password",
                         json={**change, "new_password": "short"}).status_code == 422
        assert http.post("/v1/auth/password", json=change).status_code == 204
        assert http.get("/v1/me").status_code == 200  # this session stays signed in
        assert sign_in(http, "ana").status_code == 401
        assert sign_in(http, "ana", "a-brand-new-password").status_code == 200


def test_auth_endpoints_only_exist_in_password_mode():
    s = Settings(providers=["anthropic"], auth_mode="dev")
    with TestClient(create_app(s, gateway=AIGateway({"anthropic": FakeClient()}, s))) as http:
        assert sign_in(http, "ana").status_code == 404


def test_production_needs_a_database_and_a_secure_cookie():
    with pytest.raises(ValidationError, match="AIP_DATABASE_URL"):
        Settings(auth_mode="password", env="production")
    s = Settings(auth_mode="password", env="production", database_url="postgresql+asyncpg://x")
    assert s.session_cookie_secure
