# Deploying the AI Platform

Hosting is not decided yet. This guide lists what any environment must provide, the release order, a ready-to-use Kubernetes setup (`deploy/k8s/`), and notes for the two serverless options that fit the current load (6,000–10,000 messages/day).

## 1. Requirements checklist

| Requirement | Why |
|---|---|
| Container registry | Images built from the repo `Dockerfile` (runs as non-root UID 1000, port 8000) |
| **Managed Postgres** (16+) with automated daily backups and point-in-time recovery | Conversations, messages and usage live here |
| Secret storage: `AIP_DATABASE_URL`, `ANTHROPIC_API_KEY` | Never in images, ConfigMaps or git |
| OIDC provider (Auth0 recommended): API audience + SPA client | `AIP_OIDC_ISSUER`, `AIP_OIDC_AUDIENCE`, `AIP_OIDC_CLIENT_ID` |
| TLS on the public endpoint | Tokens and conversations are sensitive |
| **Load balancer / proxy idle and read timeout ≥ 300 s, response buffering off** | Chat and agent replies stream over SSE for up to minutes |
| **2+ replicas** | No downtime during deploys or single-instance failures |
| Egress to HTTPS (model providers, OIDC issuer) and Postgres | Nothing else is needed |

Production settings (non-secret):

```
AIP_ENV=production            # refuses AIP_AUTH_MODE=dev
AIP_AUTH_MODE=oidc
AIP_OIDC_ISSUER=https://<tenant>.eu.auth0.com/
AIP_OIDC_AUDIENCE=https://api.<your-domain>
AIP_OIDC_CLIENT_ID=<spa client id>
AIP_PROVIDERS=["anthropic"]
AIP_ADMIN_USERS=["<admin sub>"]
AIP_LOG_LEVEL=INFO
```

## 2. Release order

1. **Build and push** the image, tagged with the git SHA (CI `docker` job; enable the push step).
2. **Migrate**: run `alembic upgrade head` once against the production database and wait for it to succeed. In Kubernetes that's the `aiplatform-migrate` Job.
3. **Roll out** the new image. The readiness probe (`/readyz`, checks the DB) gates traffic, and `maxUnavailable: 0` keeps full capacity during the rollout.

Migrations must stay backward compatible with the previous release (add columns/tables first, remove them in a later release), because old and new pods run side by side during step 3.

## 3. Kubernetes (`deploy/k8s/`)

```
deploy/k8s/
  base/                     Deployment, Service, ConfigMap, PDB, HPA, NetworkPolicy, Ingress, migrate Job
    secret.example.yaml     example only, NOT applied
  overlays/production/      namespace, image name/tag, replicas, hostname
```

What's in the base:
- Deployment: 2 replicas, rolling update with `maxUnavailable: 0`, liveness `/healthz`, readiness `/readyz`, 60 s grace period plus a 5 s preStop pause so SSE streams can drain, non-root, read-only root filesystem (`/tmp` is an emptyDir), all capabilities dropped, seccomp `RuntimeDefault`, Prometheus scrape annotations for the internal metrics port 9090 (`AIP_METRICS_PORT`). It is not exposed through the Service or Ingress, because metrics include usage and cost; the NetworkPolicy only lets the `monitoring` namespace reach it.
- HPA 2–6 pods at 70% CPU with a slow scale-down (removing a pod cuts its open streams). PDB `minAvailable: 1`.
- NetworkPolicy: ingress only from the `ingress-nginx` and `monitoring` namespaces on port 8000. Egress limited to DNS, 443 and 5432.
- Ingress (ingress-nginx): buffering off, 300 s read/send timeouts, TLS via cert-manager (`cert-manager.io/cluster-issuer: letsencrypt-prod`, change to your issuer).

Before the first deploy, edit `overlays/production/kustomization.yaml` (registry, tag, hostname) and `base/configmap.yaml` (OIDC values), then:

```bash
kubectl apply -f deploy/k8s/overlays/production/namespace.yaml
kubectl -n aiplatform create secret generic aiplatform-secrets \
  --from-literal=AIP_DATABASE_URL='postgresql+asyncpg://USER:PASS@HOST:5432/aiplatform' \
  --from-literal=ANTHROPIC_API_KEY='sk-ant-...'
# (or sync the Secret from your cloud secret manager with External Secrets Operator)

# Each release (after pushing the image and updating newTag):
kubectl -n aiplatform delete job aiplatform-migrate --ignore-not-found
kubectl apply -k deploy/k8s/overlays/production
kubectl -n aiplatform wait --for=condition=complete job/aiplatform-migrate --timeout=5m
kubectl -n aiplatform rollout status deployment/aiplatform
```

`kubectl apply -k` creates the Job and updates the Deployment together. Pods with the new image only become Ready once `/readyz` passes, but that doesn't wait for the migration. For strict ordering, apply the Job first (for example with a CI step, or an Argo CD sync-wave/Helm hook), then the Deployment.

Assumptions to adjust for your cluster: the ingress class is `nginx` in namespace `ingress-nginx`, Prometheus runs in `monitoring`, and the cluster supports the `sleep` preStop action (Kubernetes 1.30+).

Validate the manifests locally without a cluster (CI does the same):

```bash
docker run --rm -v "$PWD/deploy/k8s:/k8s:ro" registry.k8s.io/kubectl:v1.33.0 \
  kustomize /k8s/overlays/production > /tmp/rendered.yaml
docker run --rm -i ghcr.io/yannh/kubeconform:latest \
  -strict -summary -ignore-missing-schemas -kubernetes-version 1.33.0 - < /tmp/rendered.yaml
```

## 4. Serverless alternatives (simplest at this load)

**Google Cloud Run + Cloud SQL (Postgres)**
- Deploy the same image. `--min-instances=2` (avoids cold starts and keeps 2 replicas), `--timeout=900` (request timeout covers SSE; the default is 300 s), `--concurrency=80`.
- Connect to Cloud SQL through the Cloud SQL connector/Auth Proxy or private IP, and put secrets in Secret Manager.
- Cloud Run supports streaming responses over HTTP/1.1 and HTTP/2. Streams end when the request timeout is reached.
- Run migrations as a Cloud Run **Job** with the same image and `alembic upgrade head` before shifting traffic.
- Vertex AI is available as a Claude failover provider in the same project.

**AWS ECS Fargate + RDS (Postgres)**
- 2+ tasks behind an Application Load Balancer. **Raise the ALB idle timeout to ≥ 300 s** (the default is 60 s and cuts long SSE streams). Health check path `/readyz`.
- Secrets in Secrets Manager, injected as task environment variables. RDS in private subnets.
- Run migrations as a one-off ECS task (`alembic upgrade head`) before updating the service. Deployment circuit breaker with rollback enabled.
- Amazon Bedrock is available as a Claude failover provider.

Both options handle scaling, TLS and patching for you. Choose Kubernetes only if you already run a cluster or need its flexibility.

## 5. Rollback

- **App:** redeploy the previous image tag (`kubectl -n aiplatform rollout undo deployment/aiplatform`, or pin the previous `newTag` and apply). On Cloud Run, shift traffic to the previous revision; on ECS, update the service to the previous task definition.
- **Database:** migrations are forward-only in production. Because they are backward compatible (section 2), the previous app version keeps working on the new schema. Use `alembic downgrade` only in an emergency and after a backup. For data loss, restore from point-in-time recovery.

## 6. What to monitor

From the metrics endpoint (port 9090) and the JSON logs (every line has `request_id`):

| Metric | What it tells you |
|---|---|
| `aip_http_requests_total{route,status}`, `aip_http_response_start_seconds` | 5xx rate, latency per route (time until the SSE stream starts) |
| `aip_llm_time_to_first_token_seconds{route,provider}` | how long users wait for the first word |
| `aip_llm_errors_total{provider,kind}` | retries, timeouts (`kind="timeout"`, provider sent nothing within the route's first-event timeout), failovers |
| `aip_llm_breaker_open{provider}` | provider currently considered down |
| `aip_llm_tokens_total{kind}`, `aip_llm_cost_usd_total` | spend by route/model (list prices); daily report at `GET /v1/admin/usage` (users in `AIP_ADMIN_USERS`) |
| `aip_llm_prompt_below_cache_minimum_total` | prompts too short for prompt caching to apply |
| `aip_requests_rejected_total{reason}` | rate limit, token quota, concurrent reply (409) |

Also watch pod CPU/memory and the HPA replica count.
- Database: connections, slow queries, storage growth, backup success.

Suggested alerts: 5xx > 2% for 5 min, p95 `aip_llm_time_to_first_token_seconds` > 10 s for 10 min, provider error rate > 10%, `aip_llm_breaker_open == 1`, daily cost above budget, readiness failures, and failed backups.

## 7. Tracing (Langfuse)

Each chat or agent turn is sent as one trace to a **self-hosted Langfuse**: the graph steps, every model call (prompt, reply, tokens, provider, retry attempt) and every tool call. Traces are grouped by conversation (Langfuse **Sessions**). The data is masked before it leaves the app (`src/aiplatform/privacy.py`):

| What | Becomes |
|---|---|
| First name, city, card/account last 4 digits (tool fields, and the same values anywhere later in the conversation, e.g. in the model's reply or in a later turn) | `<NAME>`, `<CITY>`, `<LAST4>` |
| Balances, limits, available credit, money amounts in text | `<AMOUNT>` (turn off with `AIP_TRACE_MASK_AMOUNTS=false`, dev only) |
| Cards (Luhn-checked), CPF, CNPJ, DNI, RUT, IBAN, CBU/CLABE and other 8+ digit runs, emails, phones | `<CARD>`, `<CPF>`, …, `<ID>`, `<EMAIL>`, `<PHONE>` |
| A PIN, CVV, one-time code or password the customer types, recognized by the word before it ("mi PIN es…", "senha: …") | `<SECRET>` |
| User ID | The bank customer ID (`CLI-…`) when sign-in knows it (password mode): it names no one and can be matched to the bank's data. Otherwise (OIDC `sub`, operators) `u_<keyed hash>`: stable per user, not reversible without `AIP_TRACE_HASH_KEY`. The username is never sent |

Product types, currencies, status, segment, interest rates and days past due stay visible for debugging. Masking is pattern-based: a name the customer types in free text, in a message where no tool returned it, is not caught. That's why the Langfuse UI is internal only.

**Kubernetes** (`deploy/langfuse/values.yaml`, official chart `langfuse/langfuse` v2, Langfuse v4):

1. Once per cluster: cert-manager and the ClickHouse operator (chart requirement, Kubernetes 1.28+). See the [chart README](https://github.com/langfuse/langfuse-k8s#prerequisites).
2. Create the `langfuse-secrets` Secret with every key referenced in `values.yaml` (`salt`, `encryption-key`, `nextauth-secret`, `oidc-client-secret`, `postgres-password`, `clickhouse-password`, `redis-password` and `default` (the same Redis password, for the Valkey ACL user), `s3-access-key-id`, `s3-secret-access-key`). Keep `salt` and `encryption-key` stable: rotating them breaks the stored API keys and encrypted data.
3. Fill in the `CHANGEME` values, then run `helm install langfuse langfuse/langfuse --version 2.1.3 -n langfuse --create-namespace -f deploy/langfuse/values.yaml`.
4. In the Langfuse UI: create the `aiplatform` project and its API keys. Put them in `aiplatform-secrets` (`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`), along with `AIP_TRACE_HASH_KEY` (`openssl rand -hex 32`). With tracing on in production, the app refuses to start without it.
5. The app's NetworkPolicy already allows egress to the `langfuse` namespace on port 3000, and the ConfigMap points `LANGFUSE_BASE_URL` at `langfuse-web.langfuse.svc`.

**Operations:**
- **Backups:** Postgres (users, projects, API keys) through the managed database. ClickHouse holds the traces: back it up (the ClickHouse `BACKUP` command to object storage) or accept losing trace history.
- **Size:** at 6,000–10,000 messages/day, expect on the order of 0.5–1 GB/day of raw trace data (each agent step carries the masked history), a few times less after ClickHouse compression. Measure after the first week and set a retention period that matches your data policy.
- If the Langfuse server is down, the app keeps serving: spans are batched and sent in the background, and failed exports are dropped, never retried into the request path.
- **Local and single-host:** `make langfuse-env && make langfuse-up` (see `deploy/langfuse/docker-compose.yml`).
