import base64
import hashlib
import importlib.util
import re
import urllib.parse
from pathlib import Path

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient

from aiplatform.api.app import create_app
from aiplatform.config import Settings
from aiplatform.llm.gateway import AIGateway
from tests.fakes import FakeClient

ROOT = Path(__file__).resolve().parent.parent


def app_client(**settings):
    s = Settings(providers=["anthropic"], **settings)
    return TestClient(create_app(s, gateway=AIGateway({"anthropic": FakeClient()}, s)))


def test_ui_is_served_with_security_headers():
    with app_client() as http:
        page = http.get("/")
        assert page.status_code == 200 and 'src="/i18n.js"' in page.text
        csp = page.headers["content-security-policy"]
        assert "script-src 'self'" in csp and "frame-ancestors 'none'" in csp
        assert page.headers["x-content-type-options"] == "nosniff"
        assert http.get("/app.js").headers["content-type"].startswith(
            ("text/javascript", "application/javascript"))
        assert http.get("/config.json").json()["auth_mode"] == "dev"
        # API routes still win over the static mount.
        assert http.get("/healthz").json() == {"status": "ok"}


def test_ui_config_and_csp_allow_the_oidc_issuer():
    with app_client(auth_mode="oidc", oidc_issuer="https://tenant.example.com/",
                    oidc_audience="https://api.example", oidc_client_id="spa-123") as http:
        config = http.get("/config.json").json()
        assert config == {"auth_mode": "oidc", "oidc_issuer": "https://tenant.example.com/",
                          "oidc_client_id": "spa-123", "oidc_audience": "https://api.example"}
        assert "connect-src 'self' https://tenant.example.com" in http.get("/").headers[
            "content-security-policy"]


def test_ui_never_uses_innerhtml():
    for name in ("app.js", "i18n.js"):
        source = (ROOT / "src/aiplatform/web" / name).read_text()
        assert not re.search(r"innerHTML|outerHTML|insertAdjacentHTML|document\.write", source)


def test_translations_are_complete():
    source = (ROOT / "src/aiplatform/web/i18n.js").read_text()
    blocks = dict(re.findall(r"\n  (es|pt|en): \{(.*?)\n  \},", source, re.DOTALL))
    keys = {lang: set(re.findall(r"^\s+(\w+):", body, re.MULTILINE))
            for lang, body in blocks.items()}
    assert set(keys) == {"es", "pt", "en"}
    assert keys["es"] == keys["pt"] == keys["en"]
    used = set(re.findall(r'\bt\("(\w+)"', (ROOT / "src/aiplatform/web/app.js").read_text()))
    used |= set(re.findall(r'data-i18n(?:-[a-z-]+)?="(\w+)"',
                           (ROOT / "src/aiplatform/web/index.html").read_text()))
    assert used <= keys["en"], used - keys["en"]


# --- dev OIDC issuer: authorization code + PKCE -----------------------------

def load_dev_oidc():
    spec = importlib.util.spec_from_file_location("dev_oidc", ROOT / "scripts/dev_oidc.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ISSUER = "http://127.0.0.1:9000/"
REDIRECT = "http://localhost:8000/"


@pytest.fixture
def issuer():
    dev = load_dev_oidc()
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return TestClient(dev.create_issuer_app(key, ISSUER)), key


def pkce():
    verifier = "v" * 50
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    return verifier, challenge


def login(http, challenge, username="ana", redirect=REDIRECT):
    form = {"username": username, "client_id": "web", "redirect_uri": redirect,
            "state": "s1", "code_challenge": challenge, "code_challenge_method": "S256",
            "audience": "aiplatform-dev", "scope": "openid"}
    resp = http.post("/authorize", content=urllib.parse.urlencode(form),
                     headers={"content-type": "application/x-www-form-urlencoded"},
                     follow_redirects=False)
    return resp


def exchange(http, code, verifier):
    return http.post("/token", content=urllib.parse.urlencode({
        "grant_type": "authorization_code", "code": code, "redirect_uri": REDIRECT,
        "client_id": "web", "code_verifier": verifier}),
        headers={"content-type": "application/x-www-form-urlencoded"})


def test_pkce_login_issues_a_valid_access_token(issuer):
    http, key = issuer
    verifier, challenge = pkce()
    meta = http.get("/.well-known/openid-configuration").json()
    assert meta["authorization_endpoint"] == f"{ISSUER}authorize"
    form = http.get("/authorize", params={"redirect_uri": REDIRECT, "code_challenge": challenge,
                                          "code_challenge_method": "S256", "state": "<x>"})
    assert form.status_code == 200 and "&lt;x&gt;" in form.text  # reflected params are escaped

    redirect = login(http, challenge)
    assert redirect.status_code == 303
    query = urllib.parse.parse_qs(urllib.parse.urlparse(redirect.headers["location"]).query)
    assert query["state"] == ["s1"]
    token = exchange(http, query["code"][0], verifier).json()["access_token"]
    claims = jwt.decode(token, key.public_key(), algorithms=["RS256"], audience="aiplatform-dev")
    assert claims["sub"] == "ana" and claims["iss"] == ISSUER

    # Codes are single-use.
    assert exchange(http, query["code"][0], verifier).status_code == 400


def test_pkce_rejects_wrong_verifier_and_foreign_redirects(issuer):
    http, _ = issuer
    _, challenge = pkce()
    code = urllib.parse.parse_qs(urllib.parse.urlparse(
        login(http, challenge).headers["location"]).query)["code"][0]
    assert exchange(http, code, "not-the-verifier").status_code == 400
    assert login(http, challenge, redirect="https://evil.example/").status_code == 400
    assert http.get("/authorize", params={"redirect_uri": "https://evil.example/",
                                          "code_challenge": challenge,
                                          "code_challenge_method": "S256"}).status_code == 400


def test_user_data_is_never_cached():
    with app_client() as http:
        headers = {"X-User-Id": "ana"}
        assert http.get("/v1/conversations", headers=headers).headers["cache-control"] == "no-store"
        cid = http.post("/v1/conversations", json={"kind": "chat"}, headers=headers).json()["id"]
        assert http.get(f"/v1/conversations/{cid}", headers=headers).headers[
            "cache-control"] == "no-store"
        assert http.get("/config.json").headers["cache-control"] == "no-store"
        assert http.get("/").headers["cache-control"] == "no-store"


def test_dev_issuer_logout_redirects_only_to_localhost(issuer):
    http, _ = issuer
    meta = http.get("/.well-known/openid-configuration").json()
    assert meta["end_session_endpoint"] == f"{ISSUER}logout"
    ok = http.get("/logout", params={"post_logout_redirect_uri": REDIRECT}, follow_redirects=False)
    assert ok.status_code == 303 and ok.headers["location"] == REDIRECT
    assert http.get("/logout", params={"post_logout_redirect_uri": "https://evil.example/"},
                    follow_redirects=False).status_code == 400
