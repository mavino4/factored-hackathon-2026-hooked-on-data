# AI Platform (v1)

A streaming chat assistant and tool-using agent, built with Python, FastAPI and the Anthropic SDK.
The first versions run on the light model, **Claude Haiku 4.5**.

- Architecture and growth path: [`docs/architecture/README.md`](docs/architecture/README.md)
- Decisions: [`docs/adr/`](docs/adr/)

## Layout

```
src/aiplatform/
  config.py          settings from env (AIP_*)
  llm/models.py      model profiles + per-route config  <- change models here
  llm/gateway.py     AI Gateway: retries, circuit breaker, failover, cache breakpoints
  llm/providers.py   Claude API / Bedrock / Vertex clients
  chat/              conversations (append-only) and streamed chat turns
  storage/           Postgres/SQL schema + repository and usage store
  usage.py           token usage events and the daily token quota
  auth.py            OIDC access-token (JWT) verification
  agent/             tool registry and the agent tool loop
  api/app.py         HTTP API (SSE streaming)
migrations/          Alembic migrations (must match storage/tables.py)
tests/               run with a fake provider, no network
```

## Run locally

Quickest (in-memory storage, data lost on restart):

```bash
uv sync
cp .env.example .env          # set ANTHROPIC_API_KEY
make run
```

Without a Claude API key, use a local Ollama model (Ollama serves the same Messages API):

```bash
ollama pull llama3.2:3b
AIP_PROVIDERS='["ollama"]' make run
```

End-to-end smoke test against a real provider (chat streaming, history, agent tool call):

```bash
AIP_PROVIDERS='["ollama"]' uv run python scripts/smoke.py
```

### Authentication

- `AIP_AUTH_MODE=dev` (local only): the API trusts an `X-User-Id` header. Refused when `AIP_ENV=production`.
- `AIP_AUTH_MODE=oidc` (default): every `/v1/*` call needs `Authorization: Bearer <JWT>` from the
  configured issuer. Any standard OIDC provider works (Auth0, Keycloak, Cognito, Entra ID…):
  set `AIP_OIDC_ISSUER` and `AIP_OIDC_AUDIENCE`. `user_id` is the token's `sub`.

Try real OIDC locally with the bundled dev issuer (not for production):

```bash
uv run python scripts/dev_oidc.py serve &                 # http://127.0.0.1:9000/
export AIP_AUTH_MODE=oidc AIP_OIDC_ISSUER=http://127.0.0.1:9000/ AIP_OIDC_AUDIENCE=aiplatform-dev
AIP_SMOKE_TOKEN=$(uv run python scripts/dev_oidc.py token --sub ana) \
AIP_SMOKE_TOKEN_OTHER=$(uv run python scripts/dev_oidc.py token --sub bruno) \
AIP_PROVIDERS='["ollama"]' make smoke
```

Full stack (Postgres + 2 app replicas behind nginx, needs Docker):

```bash
docker compose up --build -d  # API on http://localhost:8000 (dev auth, Ollama provider)
docker compose down           # stop (add -v to also delete the database volume)
```

The stack also creates an `aiplatform_test` database for the Postgres storage tests:

```bash
AIP_TEST_DATABASE_URL=postgresql+asyncpg://aiplatform:aiplatform@localhost:5432/aiplatform_test \
  make test-postgres
```

The app containers reach the host's Ollama at `host.docker.internal:11434`, so Ollama must listen
on an address containers can reach, not only `127.0.0.1`.

Against your own Postgres: set `AIP_DATABASE_URL=postgresql+asyncpg://...`, then `make migrate && make run`.

```bash
# create a chat conversation and stream a reply
CID=$(curl -s localhost:8000/v1/conversations -H 'X-User-Id: demo' \
      -H 'content-type: application/json' -d '{"kind":"chat"}' | jq -r .id)
curl -N localhost:8000/v1/conversations/$CID/messages -H 'X-User-Id: demo' \
     -H 'content-type: application/json' -d '{"text":"Hello!"}'

# agent run
AID=$(curl -s localhost:8000/v1/conversations -H 'X-User-Id: demo' \
      -H 'content-type: application/json' -d '{"kind":"agent"}' | jq -r .id)
curl -s localhost:8000/v1/conversations/$AID/agent-runs -H 'X-User-Id: demo' \
     -H 'content-type: application/json' -d '{"text":"What time is it in UTC?"}'
```

## Test

```bash
make check                    # lint + tests (in-memory and SQLite)
make test-postgres            # same storage tests against real Postgres (see above)
```

## Before production

- Create the production OIDC application/API in your provider (Auth0 recommended) and set `AIP_OIDC_*`.
- Always set `AIP_DATABASE_URL` in deployed environments. In-memory storage is for local dev only.
- Replace the example tools in `agent/tools.py` with real integrations, and build the approval flow for irreversible tools.
- Check the Bedrock/Vertex model IDs in `llm/models.py` against your cloud accounts before enabling failover.
