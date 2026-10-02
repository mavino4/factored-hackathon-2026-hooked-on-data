# AI Platform (v1)

A streaming banking assistant (one tool-using agent that first classifies each message, with a
human handoff), built with Python, FastAPI and the Anthropic SDK.
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
  chat/              conversations (append-only), history rules and prompts
  storage/           Postgres/SQL schema + repository and usage store
  usage.py           token usage events and the daily token quota
  auth.py            OIDC access-token (JWT) verification
  accounts.py        username + password accounts, sessions and their audit trail
  agent/             the agent graph (LangGraph): intent classification, tool loop,
                     approvals and human handoffs
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
- One button, **+ Nueva consulta**: streamed replies, a conversation list, and history that
  survives reloads. General questions and questions about the customer's own products go
  through the same agent.
- Progress shows while the agent consults the bank. Actions that change something show an
  **approval card** (Approve or Reject) and run only after approval.
- If the customer insists without being resolved, BankBot offers an advisor (Yes / No card).
  While an advisor attends the conversation, their messages show with an "Asesor" label.
- Sign-in follows `AIP_AUTH_MODE`: a username field in `dev` mode, username + password in
  `password` mode, or the provider's login page in `oidc` mode (authorization code + PKCE).

Browser end-to-end test (headless Chromium; covers the three login modes, chat, reload, agent approval):

```bash
uv run --with playwright python -m playwright install chromium   # once
AIP_PROVIDERS='["ollama"]' uv run --with playwright python scripts/e2e_ui.py
```

### Authentication

- `AIP_AUTH_MODE=dev` (local only): the API trusts an `X-User-Id` header. Refused when `AIP_ENV=production`.
- `AIP_AUTH_MODE=password`: username + password accounts kept by the app (`src/aiplatform/accounts.py`).
  - **Accounts** are created by the operator, never by sign-up: `make users` gives every Active
    customer a username (the first part of their email, numbered when several customers share
    it: `maria.gomez`, `maria.gomez2`…) and a random 16-character password. Customers without
    an email get no account. Running it again only adds the missing ones.
  - **Passwords** are stored only as Argon2id hashes. The plain text exists once, in
    `credentials/users-<timestamp>.csv` (mode 600, git-ignored): deliver it safely and delete
    it. A lost password can't be recovered, only replaced:
    `uv run python scripts/users.py rotate <username>`. Users can change theirs in the UI
    (12+ characters), which signs out their other sessions. The email is never stored.
  - **Sign-in** (`POST /v1/auth/login`) answers the same `401` for an unknown user, a wrong
    password, a locked account and a disabled one. 5 failures in a row lock the account for
    15 minutes (`unlock` lifts it), and attempts are rate-limited per client address.
  - **Sessions** are an opaque random token in an `HttpOnly`, `SameSite=Strict` cookie
    (`Secure` in production); the database keeps only its SHA-256. They end after 30 minutes
    without activity, 8 hours at most, on sign-out, and when the password changes.
  - **Audit**: every account event (created, sign-in, failure, lock, sign-out, password
    change, rotation…) is appended to `auth_events` with the time and client address.
  - Other operator commands: `scripts/users.py create|disable|enable|relink` (see its header).
    An operator account for `/v1/admin/*`: `uv run python scripts/users.py create operador`.
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

To demo it to other devices on the local network (phones, other laptops), over HTTPS:

```bash
make bank-db && make migrate && make users   # once: the bank data and the customers' accounts
make lan   # UI on https://<this machine's IP>; password login, Claude, per-IP rate limit,
           # traces to Langfuse; operator API (/v1/admin/) only from this machine (https://localhost)
```

Customers sign in with the username and password from `credentials/users-*.csv` (see
[Authentication](#authentication)). Postgres only listens on `127.0.0.1`. A plain
`docker compose up` goes back to the settings in `.env`.

**HTTPS.** There is no public domain on a LAN, so no public authority can issue the
certificate: `make lan` creates a private one for this machine (`deploy/tls/gen-cert.sh`)
and a server certificate for its hostname and addresses. Port 8000 (HTTP) only redirects
to HTTPS, and the session cookie is `Secure`.
- A device shows a certificate warning until it trusts the authority: open
  `http://<this machine's IP>:8000/ca.crt` on it and install it as a trusted root
  (Android: Settings > Security > Install a certificate > CA; iOS: install the profile,
  then enable it in Settings > General > About > Certificate Trust Settings;
  desktop: import it in the browser's or system's authorities).
- The machine's IP changed? `make tls` issues a new server certificate (valid 397 days);
  devices that already trust the authority need nothing new.
- `deploy/tls/ca.key` can sign a certificate for any site for whoever trusts `ca.crt`:
  it never leaves this machine (mode 600, git-ignored). Remove the authority from a device
  when the demo is over.

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
# create a conversation (a "consulta") and stream a reply
CID=$(curl -s localhost:8000/v1/conversations -H 'X-User-Id: demo' \
      -H 'content-type: application/json' -d '{}' | jq -r .id)
curl -N localhost:8000/v1/conversations/$CID/agent-runs -H 'X-User-Id: demo' \
     -H 'content-type: application/json' -d '{"text":"¿Qué es el cupo disponible?"}'
```

### How a message is handled

Every message goes through one LangGraph graph (`agent/graph.py`):

1. **classify**: a short model call with a forced tool (`classify_intent`, route `classify`)
   decides the intent: `account` (the customer's own data: uses the tools), `general`
   (banking knowledge: no tools), `out_of_scope` (transfers, payments, complaints...: a
   polite answer, no tools) or `human` (asks for a person). It also flags **insistence**
   (repeating an unresolved request, frustration); three nearly identical messages in a row
   count as insistence too. If the classifier fails, the full agent with tools answers.
   **Manipulation attempts** (prompt injection: "ignore your rules", "show your prompt", a
   fake system message, tool result or approval, a claimed administrator asking for another
   customer's data…) get the `attack` intent: a fixed reply with no agent call and no tools
   (outcome `blocked`), never an advisor offer, a `aip_agent_intents_total{intent="attack"}`
   metric, and in Langfuse a `WARNING` on the classify step plus a `prompt_injection` score
   on the trace. A real question wrapped in odd text, or a customer who shares a PIN by
   mistake, is not an attack. Classifying and answering stay two separate model calls on
   purpose: the classifier is what keeps the tools away from anything that isn't an account
   question (the prompts are too short for Haiku to cache, so it costs about $0.002 a turn).
   For an `account` question the agent's first model call **must** call a tool, so a figure
   can never come from the customer's own text.
2. **call_model / run_tools**: the tool loop (only `account` gets the tool definitions).
3. **handoff**: `human` queues the conversation for an advisor; insistence makes BankBot
   *offer* an advisor instead (the customer accepts or declines; no new offer for 3 turns
   after declining).

While an advisor attends a conversation, the bot doesn't answer: messages are stored and
the advisor replies through the operator API. Closing the case gives it back to the bot.

### Operator API (human handoff)

For users in `AIP_ADMIN_USERS` (there is no operator screen yet):

```bash
OP=(-H 'X-User-Id: operador' -H 'content-type: application/json')   # dev mode
curl -s "localhost:8000/v1/admin/handoffs?status=open" "${OP[@]}"      # the queue
curl -s localhost:8000/v1/admin/handoffs/$HID "${OP[@]}"               # the conversation
curl -s localhost:8000/v1/admin/handoffs/$HID/messages "${OP[@]}" -d '{"text":"Hola, soy Laura."}'
curl -s -X POST localhost:8000/v1/admin/handoffs/$HID/close "${OP[@]}" # back to the bot
```

The customer answers an offer with `POST /v1/conversations/{id}/handoff`
(`{"handoff_id": "...", "accept": true}`); the web UI does it from the offer card.

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
- **Tracing (self-hosted Langfuse, masked)**, off by default. Each turn is one trace: one span per graph step (classify, call_model, run_tools, handoff...), and under each step its model calls (prompt, reply, tokens, provider, retry attempt) and every tool call (input, output, errors). Traces are grouped by conversation in Langfuse's **Sessions** view. Customer data is masked before it leaves the app: names, cities, last 4 digits, amounts, document and account numbers, emails, phones, and PINs, CVVs, passwords and codes the customer types. The trace's user is the bank customer ID in password mode, a keyed hash of the user ID otherwise; never the username. See [`docs/deploy.md`](docs/deploy.md#7-tracing-langfuse). To run it locally:

  ```bash
  make langfuse-env   # once: deploy/langfuse/.env with random secrets; prints the app settings
  make langfuse-up    # Langfuse UI on http://localhost:3000
  ```

  Then add these to `.env`: `LANGFUSE_TRACING_ENABLED=true`, `LANGFUSE_BASE_URL=http://localhost:3000`, and the `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` it printed. Run the app with `make run` to trace locally. Langfuse listens on 127.0.0.1 only, so the docker-compose app containers can't reach it. Set `AIP_TRACE_MASK_AMOUNTS=false` to see real figures while debugging in dev.

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
  In password mode they need an account first:
  `uv run python scripts/users.py create ana --customer-id <id from that file>`.
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
