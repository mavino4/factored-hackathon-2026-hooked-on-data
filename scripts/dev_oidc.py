"""A minimal local OIDC issuer for development and testing. NOT for production.

It serves a discovery document and JWKS over HTTP, and mints signed access
tokens from the command line, so the API can run in real OIDC mode locally.

    uv run python scripts/dev_oidc.py serve            # http://127.0.0.1:9000/
    uv run python scripts/dev_oidc.py token --sub ana  # prints a Bearer token

Point the API at it:
    AIP_AUTH_MODE=oidc AIP_OIDC_ISSUER=http://127.0.0.1:9000/ \\
    AIP_OIDC_AUDIENCE=aiplatform-dev make run
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import jwt
import uvicorn
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI

KEY_FILE = Path(".dev-oidc/key.pem")  # gitignored; keeps tokens valid across restarts
DEFAULT_AUDIENCE = "aiplatform-dev"


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


def serve(host: str, port: int) -> None:
    key = load_key()
    issuer = issuer_url(host, port)
    app = FastAPI(title="dev OIDC issuer")

    @app.get("/.well-known/openid-configuration")
    def discovery() -> dict:
        return {"issuer": issuer, "jwks_uri": f"{issuer}jwks.json",
                "id_token_signing_alg_values_supported": ["RS256"]}

    @app.get("/jwks.json")
    def jwks() -> dict:
        return {"keys": [public_jwk(key)]}

    print(f"dev OIDC issuer at {issuer} (audience for tokens: {DEFAULT_AUDIENCE})")
    uvicorn.run(app, host=host, port=port, log_level="warning")


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
