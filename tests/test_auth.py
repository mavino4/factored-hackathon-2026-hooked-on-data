import json
import time

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient
from pydantic import ValidationError

from aiplatform.api.app import create_app
from aiplatform.auth import REFETCH_MIN_INTERVAL_S, AuthError, OIDCVerifier
from aiplatform.config import Settings
from aiplatform.llm.gateway import AIGateway
from tests.fakes import FakeClient

ISSUER = "https://idp.example.com/"
AUDIENCE = "https://api.example.com"


class FakeIdP:
    """Discovery + JWKS served from memory; counts fetches."""

    def __init__(self):
        self.keys: dict[str, rsa.RSAPrivateKey] = {}
        self.fetches: list[str] = []
        self.add_key("k1")

    def add_key(self, kid: str) -> None:
        self.keys[kid] = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    def fetch(self, url: str) -> dict:
        self.fetches.append(url)
        if url.endswith("/.well-known/openid-configuration"):
            return {"issuer": ISSUER, "jwks_uri": "https://idp.example.com/jwks.json"}
        keys = []
        for kid, key in self.keys.items():
            jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(key.public_key()))
            keys.append({**jwk, "kid": kid, "use": "sig", "alg": "RS256"})
        return {"keys": keys}

    def token(self, kid="k1", key=None, **overrides) -> str:
        now = int(time.time())
        claims = {"iss": ISSUER, "aud": AUDIENCE, "sub": "user-123",
                  "iat": now, "exp": now + 600, **overrides}
        return jwt.encode(claims, key or self.keys[kid], algorithm="RS256", headers={"kid": kid})


@pytest.fixture
def idp():
    return FakeIdP()


def verifier_for(idp, clock=time.monotonic):
    return OIDCVerifier(ISSUER, AUDIENCE, fetch=idp.fetch, clock=clock)


async def test_valid_token(idp):
    principal = await verifier_for(idp).verify(idp.token())
    assert principal.user_id == "user-123" and principal.claims["aud"] == AUDIENCE
    assert idp.fetches[0].endswith("/.well-known/openid-configuration")


@pytest.mark.parametrize("overrides, message", [
    ({"exp": int(time.time()) - 3600}, "token expired"),
    ({"aud": "https://someone-else.example"}, "invalid token"),
    ({"iss": "https://evil.example/"}, "invalid token"),
    ({"sub": None}, "invalid token"),
])
async def test_rejected_claims(idp, overrides, message):
    claims = {k: v for k, v in overrides.items() if v is not None}
    token = idp.token(**claims)
    if "sub" in overrides:  # remove the claim entirely
        payload = jwt.decode(token, options={"verify_signature": False})
        payload.pop("sub")
        token = jwt.encode(payload, idp.keys["k1"], algorithm="RS256", headers={"kid": "k1"})
    with pytest.raises(AuthError, match=message):
        await verifier_for(idp).verify(token)


async def test_token_signed_by_another_key_is_rejected(idp):
    forged = idp.token(key=rsa.generate_private_key(public_exponent=65537, key_size=2048))
    with pytest.raises(AuthError, match="invalid token"):
        await verifier_for(idp).verify(forged)


async def test_malformed_and_unsigned_tokens(idp):
    unsigned = jwt.encode({"iss": ISSUER, "aud": AUDIENCE, "sub": "x", "exp": 9999999999},
                          key=None, algorithm="none")
    for token in ["not-a-jwt", unsigned]:
        with pytest.raises(AuthError):
            await verifier_for(idp).verify(token)


async def test_key_rotation_refetches_once_then_rate_limits(idp):
    now = [0.0]
    verifier = verifier_for(idp, clock=lambda: now[0])
    await verifier.verify(idp.token())                 # discovery + jwks
    idp.add_key("k2")
    now[0] = REFETCH_MIN_INTERVAL_S + 1
    assert (await verifier.verify(idp.token(kid="k2"))).user_id == "user-123"  # refetched
    fetches = len(idp.fetches)
    for _ in range(3):  # unknown kids within the interval must not hammer the IdP
        with pytest.raises(AuthError, match="unknown signing key"):
            await verifier.verify(idp.token(kid="k3", key=idp.keys["k1"]))
    assert len(idp.fetches) == fetches
    now[0] += REFETCH_MIN_INTERVAL_S + 1  # after the interval, one more refetch is allowed
    with pytest.raises(AuthError, match="unknown signing key"):
        await verifier.verify(idp.token(kid="k3", key=idp.keys["k1"]))
    assert len(idp.fetches) == fetches + 1


async def test_idp_down_is_an_auth_error():
    def down(url):
        raise OSError("connection refused")
    with pytest.raises(AuthError, match="identity provider unavailable"):
        await OIDCVerifier(ISSUER, AUDIENCE, fetch=down).verify(FakeIdP().token())


def test_settings_guard_rails():
    with pytest.raises(ValidationError, match="not allowed"):
        Settings(auth_mode="dev", env="production")
    with pytest.raises(ValidationError, match="AIP_OIDC_ISSUER"):
        Settings(auth_mode="oidc", oidc_issuer=None, oidc_audience=None)


def oidc_client(idp):
    s = Settings(providers=["anthropic"], auth_mode="oidc", oidc_issuer=ISSUER,
                 oidc_audience=AUDIENCE)
    app = create_app(s, gateway=AIGateway({"anthropic": FakeClient()}, s),
                     verifier=verifier_for(idp))
    return TestClient(app)


def test_api_requires_a_valid_bearer_token(idp):
    with oidc_client(idp) as http:
        assert http.get("/healthz").status_code == 200  # liveness stays public
        missing = http.get("/v1/conversations")
        assert missing.status_code == 401 and missing.headers["www-authenticate"] == "Bearer"
        # The dev header is ignored in OIDC mode.
        assert http.get("/v1/conversations", headers={"X-User-Id": "alice"}).status_code == 401
        bad = http.get("/v1/conversations", headers={"Authorization": "Bearer nope"})
        assert bad.status_code == 401 and "invalid_token" in bad.headers["www-authenticate"]

        ok = {"Authorization": f"Bearer {idp.token()}"}
        cid = http.post("/v1/conversations", json={"kind": "chat"}, headers=ok).json()["id"]
        assert http.get(f"/v1/conversations/{cid}", headers=ok).status_code == 200


def test_users_are_isolated_by_token_subject(idp):
    with oidc_client(idp) as http:
        alice = {"Authorization": f"Bearer {idp.token(sub='alice')}"}
        bob = {"Authorization": f"Bearer {idp.token(sub='bob')}"}
        cid = http.post("/v1/conversations", json={"kind": "chat"}, headers=alice).json()["id"]
        assert http.get(f"/v1/conversations/{cid}", headers=bob).status_code == 404
        assert http.get("/v1/conversations", headers=bob).json()["conversations"] == []
