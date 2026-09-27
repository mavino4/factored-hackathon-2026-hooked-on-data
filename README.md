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
  agent/             tool registry and the agent tool loop
  api/app.py         HTTP API (SSE streaming)
tests/               run with a fake provider, no network
```

## Run locally

```bash
uv sync
cp .env.example .env          # set ANTHROPIC_API_KEY
uv run uvicorn aiplatform.api.app:app --reload
```

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
uv run pytest
uv run ruff check src tests
```

## Before production

- Replace the `X-User-Id` header placeholder with OIDC/JWT verification (`api/app.py`).
- Replace `InMemoryConversationRepository` with Postgres. It is required before running more than one replica.
- Replace the example tools in `agent/tools.py` with real integrations, and build the approval flow for irreversible tools.
- Check the Bedrock/Vertex model IDs in `llm/models.py` against your cloud accounts before enabling failover.
