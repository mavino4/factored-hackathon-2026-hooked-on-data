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
  api/app.py         HTTP API (SSE streaming) + serves the web UI
  web/               web chat UI (plain HTML/CSS/JS, no build step)
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

### Web UI

Open http://localhost:8000 (from `make run` or `docker compose up`).
- Chat with streamed replies, a conversation list, and history that survives reloads.
- Agent tasks show each tool call. Actions that change something show an **approval card**
  (Approve or Reject) and run only after approval.
- Sign-in follows `AIP_AUTH_MODE`: a username field in `dev` mode, or the provider's login page
  in `oidc` mode (authorization code + PKCE).

Browser end-to-end test (headless Chromium; covers both login modes, chat, reload, agent approval):

```bash
uv run --with playwright python -m playwright install chromium   # once
AIP_PROVIDERS='["ollama"]' uv run --with playwright python scripts/e2e_ui.py
```

### Authentication

- `AIP_AUTH_MODE=dev` (local only): the API trusts an `X-User-Id` header. Refused when `AIP_ENV=production`.
- `AIP_AUTH_MODE=oidc` (default): every `/v1/*` call needs `Authorization: Bearer <JWT>` from the
  configured issuer. Any standard OIDC provider works (Auth0, Keycloak, Cognito, Entra ID…):
  set `AIP_OIDC_ISSUER` and `AIP_OIDC_AUDIENCE`. `user_id` is the token's `sub`.
  For the web UI, also create a Single Page Application in the provider with callback URL
  `https://<your host>/` and set `AIP_OIDC_CLIENT_ID`.

Try real OIDC locally with the bundled dev issuer (not for production). It also serves a login
page for the web UI: start the API with `AIP_OIDC_CLIENT_ID=web-ui` and open http://localhost:8000.

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

The app containers reach the host's Ollama at `host.docker.internal:11434`. If your Ollama
listens only on `127.0.0.1`, install the small user-level proxy in
[`deploy/ollama-docker-proxy/`](deploy/ollama-docker-proxy/README.md). It needs no sudo and
doesn't expose Ollama on your LAN.

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

## Observability

- Logs are JSON on stdout; every line in a request carries `request_id` (also returned as the `X-Request-ID` header).
- Prometheus metrics on the internal port `AIP_METRICS_PORT` (default 9090), including time-to-first-token, tokens, estimated cost, provider errors and breaker state. See [`docs/deploy.md`](docs/deploy.md#6-what-to-monitor).
- `GET /readyz` checks the database; `GET /healthz` is liveness only.
- `GET /v1/admin/usage?days=7` gives a daily usage and cost report, for users listed in `AIP_ADMIN_USERS`.

## Banking assistant (Spanish / Portuguese)

The agent answers **balance questions about the signed-in customer's own products**
(accounts, cards, loans), in the language the customer sees on screen.

- **UI:** every session opens a new query ("¡Hola, {nombre}! Soy BankBot…") with
  quick-action buttons. A selector at the top right switches the language (es/pt/en,
  remembered per browser). While BankBot works the customer sees a "Consultando…"
  animation; tool names, arguments and results are never sent to the browser.
- **Reply language = UI language:** every message carries `"language": "es"|"pt"|"en"`
  (the UI's current language). The server adds it as a *second* system block after the
  cached prompt ("Reply language: …"), so the prompt cache is unaffected. Without it,
  the model falls back to the customer's language.

- **Data:** an external core-banking database loaded from the Datathon `customers` and
  `products` tables with `make bank-db` (150k customers / 400k products). Only
  non-sensitive columns are loaded: no documents, last names, birth dates, contact data,
  income or credit scores, and product numbers keep only their last 4 digits.
- **Isolation, enforced by the database:** the app connects as the read-only role
  `bank_reader`, and every query sets `app.subject` to the user's verified token `sub`.
  Postgres **Row-Level Security** only shows the customer linked to that subject in
  `bank.customer_logins`. Without it, even `SELECT * FROM bank.products` returns 0 rows.
  The model can never pass a customer or account ID to a tool. See
  [`deploy/bankdb/schema.sql`](deploy/bankdb/schema.sql) and `tests/test_banking_postgres.py`.
- **Tools:** `get_customer_profile` and `get_products` (read-only). Balances come with
  their meaning (funds vs. amount owed), so the model doesn't confuse card debt with money.
- **Demo users** (dev mode username, or dev issuer `sub`): `ana`, `bruno`, `carol`, `dave`,
  `eval-es`, `eval-pt`; see [`deploy/bankdb/demo_logins.json`](deploy/bankdb/demo_logins.json).
  Any other user isn't linked, and the assistant says it can't see account data.
- **Out of scope for now:** card blocking, complaints, transactions, transfers. The
  assistant says so and points to other channels.

```bash
docker compose up -d db && make bank-db     # once: load the bank DB
AIP_BANK_DATABASE_URL=postgresql+asyncpg://bank_reader:bank_reader@localhost:5432/bank \
AIP_PROVIDERS='["ollama"]' AIP_OLLAMA_MODEL=qwen2.5:7b make run
```

## Evals

`evals/` holds a small quality baseline. The cases are placeholders; replace them with 20–50 real
questions. `evals/baseline.json` was recorded on Claude Haiku 4.5: 15/15, about $0.011 per full run.

```bash
make eval                                          # compare with evals/baseline.json (Claude)
uv run python evals/run.py --save-baseline         # record a new baseline
AIP_PROVIDERS='["ollama"]' uv run python evals/run.py   # free local run (don't compare with Claude)
```

A route's model in `ROUTES` may only change if its eval score improves, or stays equal at lower
cost. See [`evals/README.md`](evals/README.md).

**Banking evals** (`evals/banking.jsonl`, 25 cases in es/pt) are built from the Datathon
transcripts plus the figures in the bank DB (`make banking-cases`). They cover balances,
available credit, limits, loans, days past due, the transcripts' follow-ups, out-of-scope
requests and security (other people's data, PIN, prompt injection, unlinked users).
Amounts are matched in any number format (`1.234,56` / `1,234.56`), and the reply language
is checked. Each case sends its expected language as the UI language, as the web UI does.

```bash
make compare-models   # llama3.2:3b vs qwen2.5:7b on Ollama, by topic and language
make eval-banking     # the provider configured in .env / AIP_PROVIDERS
```

## Before production

- Create the production OIDC application/API in your provider (Auth0 recommended) and set `AIP_OIDC_*`.
- Always set `AIP_DATABASE_URL` in deployed environments. In-memory storage is for local dev only.
- Deployment guide (Kubernetes manifests, CI, Cloud Run / ECS notes): [`docs/deploy.md`](docs/deploy.md).
- Replace the example tools in `agent/tools.py` with real integrations, and build the approval flow for irreversible tools.
- Check the Bedrock/Vertex model IDs in `llm/models.py` against your cloud accounts before enabling failover.
