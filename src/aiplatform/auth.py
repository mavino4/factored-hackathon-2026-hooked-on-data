"""Authentication: verify OIDC access tokens (JWT) from any standard provider.

Works with Auth0, Keycloak, Cognito, Entra ID, Zitadel... The provider is chosen
purely by configuration: issuer URL + audience. Signing keys come from the
issuer's JWKS (found via OIDC discovery) and are cached; an unknown key ID
triggers one refetch, which handles key rotation.
"""

import asyncio
import json
import logging
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import jwt

log = logging.getLogger(__name__)

ALGORITHMS = ["RS256", "ES256"]
REFETCH_MIN_INTERVAL_S = 60.0


class AuthError(Exception):
    """The request is not authenticated. The message is safe to return to clients."""


@dataclass(frozen=True)
class Principal:
    user_id: str
    claims: dict[str, Any] = field(default_factory=dict)
    # The bank customer behind the user, when sign-in knows it (password mode).
    customer_id: str | None = None


def fetch_json(url: str, timeout: float = 5.0) -> dict:
    # URLs come from settings or the issuer's discovery document, never from requests.
    if not url.startswith(("https://", "http://")):
        raise ValueError(f"unsupported URL scheme: {url}")
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read())


class OIDCVerifier:
    def __init__(self, issuer: str, audience: str, *, jwks_url: str | None = None,
                 fetch: Callable[[str], dict] = fetch_json, clock=time.monotonic,
                 leeway_s: int = 30):
        self.issuer = issuer
        self.audience = audience
        self._jwks_url = jwks_url
        self._fetch = fetch
        self._clock = clock
        self._leeway = leeway_s
        self._keys: jwt.PyJWKSet | None = None
        self._fetched_at = float("-inf")
        self._lock = asyncio.Lock()

    async def verify(self, token: str) -> Principal:
        try:
            kid = jwt.get_unverified_header(token).get("kid")
        except jwt.PyJWTError:
            raise AuthError("malformed token") from None
        key = await self._signing_key(kid)
        try:
            claims = jwt.decode(
                token, key.key, algorithms=ALGORITHMS, audience=self.audience,
                issuer=self.issuer, leeway=self._leeway,
                options={"require": ["exp", "iss", "aud", "sub"]})
        except jwt.ExpiredSignatureError:
            raise AuthError("token expired") from None
        except jwt.PyJWTError as exc:
            log.info("token rejected", extra={"reason": str(exc)})
            raise AuthError("invalid token") from None
        return Principal(user_id=claims["sub"], claims=claims)

    async def _signing_key(self, kid: str | None) -> jwt.PyJWK:
        key = self._find(kid)
        if key is None:
            async with self._lock:  # one refetch at a time
                key = self._find(kid)
                if key is None and self._clock() - self._fetched_at >= REFETCH_MIN_INTERVAL_S:
                    await self._refresh()
                    key = self._find(kid)
        if key is None:
            raise AuthError("unknown signing key")
        return key

    def _find(self, kid: str | None) -> jwt.PyJWK | None:
        if self._keys is None:
            return None
        for key in self._keys.keys:
            if kid is None or key.key_id == kid:
                return key
        return None

    async def _refresh(self) -> None:
        self._fetched_at = self._clock()
        try:
            if self._jwks_url is None:
                discovery_url = self.issuer.rstrip("/") + "/.well-known/openid-configuration"
                discovery = await asyncio.to_thread(self._fetch, discovery_url)
                self._jwks_url = discovery["jwks_uri"]
            jwks = await asyncio.to_thread(self._fetch, self._jwks_url)
            self._keys = jwt.PyJWKSet.from_dict(jwks)
        except Exception as exc:
            log.exception("could not load signing keys from the identity provider")
            raise AuthError("identity provider unavailable") from exc
