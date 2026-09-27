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

Full stack (Postgres + 2 app replicas behind nginx, needs Docker):

```bash
docker compose up --build     # API on http://localhost:8000
```

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
AIP_TEST_DATABASE_URL=postgresql+asyncpg://aiplatform:aiplatform@localhost:5432/aiplatform_test \
  make test-postgres          # same storage tests against real Postgres
```

## Before production

- Replace the `X-User-Id` header placeholder with OIDC/JWT verification (`api/app.py`).
- Always set `AIP_DATABASE_URL` in deployed environments. In-memory storage is for local dev only.
- Replace the example tools in `agent/tools.py` with real integrations, and build the approval flow for irreversible tools.
- Check the Bedrock/Vertex model IDs in `llm/models.py` against your cloud accounts before enabling failover.
