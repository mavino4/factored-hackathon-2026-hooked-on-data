"""A minimal local OIDC issuer for development and testing. NOT for production.

It serves discovery + JWKS, a browser login (authorization code + PKCE, any
username) for the web UI, and mints tokens from the command line, so the API
and UI can run in real OIDC mode locally.

    uv run python scripts/dev_oidc.py serve            # http://127.0.0.1:9000/
    uv run python scripts/dev_oidc.py token --sub ana  # prints a Bearer token

Point the API at it:
    AIP_AUTH_MODE=oidc AIP_OIDC_ISSUER=http://127.0.0.1:9000/ \\
    AIP_OIDC_AUDIENCE=aiplatform-dev make run
"""

import argparse
import base64
import hashlib
import html
import json
import re
import secrets
import time
import urllib.parse
from pathlib import Path

import jwt
import uvicorn
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse

KEY_FILE = Path(".dev-oidc/key.pem")  # gitignored; keeps tokens valid across restarts
DEFAULT_AUDIENCE = "aiplatform-dev"
USERNAME = re.compile(r"[A-Za-z0-9._-]{1,40}")


def load_key() -> rsa.RSAPrivateKey:
    if KEY_FILE.exists():
        return serialization.load_pem_private_key(KEY_FILE.read_bytes(), password=None)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    KEY_FILE.parent.mkdir(exist_ok=True)
    KEY_FILE.write_bytes(key.private_bytes(serialization.Encoding.PEM,
                                           serialization.PrivateFormat.PKCS8,
                                           serialization.NoEncryption()))
    KEY_FILE.chmod(0o600)
    return key


def public_jwk(key: rsa.RSAPrivateKey) -> dict:
    jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(key.public_key()))
    kid = hashlib.sha256(f"{jwk['n']}{jwk['e']}".encode()).hexdigest()[:16]
    return {**jwk, "kid": kid, "use": "sig", "alg": "RS256"}


def issuer_url(host: str, port: int) -> str:
    return f"http://{host}:{port}/"


def mint(key, issuer: str, sub: str, audience: str, ttl: int) -> str:
    now = int(time.time())
    claims = {"iss": issuer, "aud": audience, "sub": sub, "iat": now, "exp": now + ttl}
    return jwt.encode(claims, key, algorithm="RS256", headers={"kid": public_jwk(key)["kid"]})


LOGIN_PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Dev login</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>body{{font-family:system-ui,sans-serif;display:grid;place-items:center;min-height:100vh;
margin:0;background:#f4f4f5}}form{{background:#fff;padding:2rem;border-radius:12px;
box-shadow:0 2px 12px #0002;display:grid;gap:.75rem;min-width:280px}}
input,button{{font:inherit;padding:.6rem;border-radius:8px;border:1px solid #ccc}}
button{{background:#18181b;color:#fff;border:0;cursor:pointer}}</style></head>
<body><form method="post" action="{action}">
<h2 style="margin:0">Dev OIDC login</h2>
<p style="margin:0;color:#666">Local development issuer: any username works.</p>
<input name="username" placeholder="username" required autofocus pattern="[A-Za-z0-9._\\-]{{1,40}}">
{hidden}
<button type="submit">Sign in</button></form></body></html>"""

AUTHORIZE_PARAMS = ("client_id", "redirect_uri", "state", "code_challenge",
                    "code_challenge_method", "audience", "scope")


def _local_redirect(uri: str) -> bool:
    parsed = urllib.parse.urlparse(uri)
    return parsed.scheme in ("http", "https") and parsed.hostname in ("localhost", "127.0.0.1")


def create_issuer_app(key: rsa.RSAPrivateKey, issuer: str) -> FastAPI:
    """Discovery, JWKS, and the authorization-code + PKCE flow (browser login)."""
    app = FastAPI(title="dev OIDC issuer")
    # The web UI calls discovery and the token endpoint from another origin.
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"],
                       allow_headers=["*"])
    codes: dict[str, dict] = {}  # one-time authorization codes

    @app.get("/.well-known/openid-configuration")
    def discovery() -> dict:
        return {"issuer": issuer, "jwks_uri": f"{issuer}jwks.json",
                "authorization_endpoint": f"{issuer}authorize",
                "token_endpoint": f"{issuer}token",
                "end_session_endpoint": f"{issuer}logout",
                "response_types_supported": ["code"],
                "code_challenge_methods_supported": ["S256"],
                "id_token_signing_alg_values_supported": ["RS256"]}

    @app.get("/jwks.json")
    def jwks() -> dict:
        return {"keys": [public_jwk(key)]}

    @app.get("/authorize", response_class=HTMLResponse)
    def authorize_form(request: Request) -> str:
        params = {k: request.query_params.get(k, "") for k in AUTHORIZE_PARAMS}
        if not _local_redirect(params["redirect_uri"]):
            raise HTTPException(400, "redirect_uri must be on localhost")
        if params["code_challenge_method"] != "S256" or not params["code_challenge"]:
            raise HTTPException(400, "PKCE with S256 is required")
        hidden = "".join(f'<input type="hidden" name="{k}" value="{html.escape(v)}">'
                         for k, v in params.items())
        return LOGIN_PAGE.format(action=f"{issuer}authorize", hidden=hidden)

    @app.post("/authorize")
    async def authorize_submit(request: Request) -> RedirectResponse:
        form = {k: v[0] for k, v in urllib.parse.parse_qs((await request.body()).decode()).items()}
        username = form.get("username", "")
        if not USERNAME.fullmatch(username) or not _local_redirect(form.get("redirect_uri", "")):
            raise HTTPException(400, "invalid login request")
        code = secrets.token_urlsafe(24)
        codes[code] = {**form, "expires": time.time() + 60}
        query = urllib.parse.urlencode({"code": code, "state": form.get("state", "")})
        return RedirectResponse(f"{form['redirect_uri']}?{query}", status_code=303)

    @app.get("/logout")
    def logout(post_logout_redirect_uri: str = "") -> RedirectResponse:
        # The dev issuer keeps no login session; just send the browser back.
        if not _local_redirect(post_logout_redirect_uri):
            raise HTTPException(400, "post_logout_redirect_uri must be on localhost")
        return RedirectResponse(post_logout_redirect_uri, status_code=303)

    @app.post("/token")
    async def token(request: Request) -> dict:
        form = {k: v[0] for k, v in urllib.parse.parse_qs((await request.body()).decode()).items()}
        grant = codes.pop(form.get("code", ""), None)  # one-time use
        if (grant is None or grant["expires"] < time.time()
                or form.get("grant_type") != "authorization_code"
                or form.get("redirect_uri") != grant["redirect_uri"]
                or form.get("client_id") != grant["client_id"]):
            raise HTTPException(400, "invalid_grant")
        challenge = base64.urlsafe_b64encode(
            hashlib.sha256(form.get("code_verifier", "").encode()).digest()).rstrip(b"=").decode()
        if not secrets.compare_digest(challenge, grant["code_challenge"]):
            raise HTTPException(400, "invalid_grant: PKCE verification failed")
        ttl = 3600
        access_token = mint(key, issuer, grant["username"],
                            grant.get("audience") or DEFAULT_AUDIENCE, ttl)
        return {"access_token": access_token, "token_type": "Bearer", "expires_in": ttl}

    return app


def serve(host: str, port: int) -> None:
    issuer = issuer_url(host, port)
    print(f"dev OIDC issuer at {issuer} (audience for tokens: {DEFAULT_AUDIENCE})")
    uvicorn.run(create_issuer_app(load_key(), issuer), host=host, port=port, log_level="warning")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    t = sub.add_parser("token")
    for p in (s, t):
        p.add_argument("--host", default="127.0.0.1")
        p.add_argument("--port", type=int, default=9000)
    t.add_argument("--sub", default="dev-user")
    t.add_argument("--aud", default=DEFAULT_AUDIENCE)
    t.add_argument("--ttl", type=int, default=3600)
    args = parser.parse_args()
    if args.cmd == "serve":
        serve(args.host, args.port)
    else:
        print(mint(load_key(), issuer_url(args.host, args.port), args.sub, args.aud, args.ttl))


if __name__ == "__main__":
    main()
