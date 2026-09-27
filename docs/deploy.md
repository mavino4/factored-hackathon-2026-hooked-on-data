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
- Deployment: 2 replicas, rolling update with `maxUnavailable: 0`, liveness `/healthz`, readiness `/readyz`, 60 s grace period plus a 5 s preStop pause so SSE streams can drain, non-root, read-only root filesystem (`/tmp` is an emptyDir), all capabilities dropped, seccomp `RuntimeDefault`, Prometheus scrape annotations for `/metrics`.
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

From `/metrics` and the logs:
- Availability and latency: HTTP 5xx rate, p95 latency per route, **time-to-first-token** for chat.
- Model providers: error rate by provider/class, circuit-breaker state, retries and failovers.
- Cost: tokens by route/model (input, output, cache read/write), daily cost from `/v1/admin/usage`.
- Abuse and capacity: 429 (rate limit / quota) and 409 (concurrent reply) counts, pod CPU/memory, HPA replica count.
- Database: connections, slow queries, storage growth, backup success.

Suggested alerts: 5xx > 2% for 5 min, p95 time-to-first-token > 10 s for 10 min, provider error rate > 10%, circuit breaker open, daily cost above budget, readiness failures, and failed backups.
