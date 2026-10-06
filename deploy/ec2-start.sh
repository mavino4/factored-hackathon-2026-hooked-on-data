#!/bin/bash
# Start Postgres, Langfuse, and the app image on the EC2 host.
# The app image is the one built from the repository Dockerfile.
set -euo pipefail

: "${IMAGE:?}" "${AWS_REGION:?}" "${ECR_REGISTRY:?}"

aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$ECR_REGISTRY"
docker pull "$IMAGE"

docker network create factored || true
docker volume create pgdata || true
if ! docker inspect factored-db >/dev/null 2>&1; then
  docker run -d --name factored-db --restart unless-stopped \
    --network factored \
    -e POSTGRES_USER=aiplatform \
    -e POSTGRES_PASSWORD=aiplatform \
    -e POSTGRES_DB=aiplatform \
    -v pgdata:/var/lib/postgresql/data \
    postgres:16
fi
docker start factored-db
ready=0
for _ in $(seq 1 30); do
  if docker exec factored-db pg_isready -U aiplatform; then
    ready=1
    break
  fi
  sleep 2
done
test "$ready" = 1

docker run --rm --network factored \
  -e AIP_DATABASE_URL=postgresql+asyncpg://aiplatform:aiplatform@factored-db:5432/aiplatform \
  "$IMAGE" /app/.venv/bin/alembic upgrade head

if ! docker compose version >/dev/null 2>&1; then
  mkdir -p /usr/libexec/docker/cli-plugins /usr/local/lib/docker/cli-plugins
  curl -fsSL -o /usr/local/lib/docker/cli-plugins/docker-compose \
    https://github.com/docker/compose/releases/download/v2.39.4/docker-compose-linux-x86_64
  chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
  cp /usr/local/lib/docker/cli-plugins/docker-compose /usr/libexec/docker/cli-plugins/docker-compose
fi

sed -i 's|127.0.0.1:3000:3000|3000:3000|' /opt/langfuse/docker-compose.yml

if [ ! -f /opt/langfuse/.env ]; then
  hex() { openssl rand -hex "$1"; }
  cat > /opt/langfuse/.env <<EOF
NEXTAUTH_SECRET=$(hex 32)
SALT=$(hex 32)
ENCRYPTION_KEY=$(hex 32)
POSTGRES_PASSWORD=$(hex 24)
CLICKHOUSE_PASSWORD=$(hex 24)
MINIO_ROOT_PASSWORD=$(hex 24)
REDIS_AUTH=$(hex 24)
LANGFUSE_INIT_USER_EMAIL=admin@example.com
LANGFUSE_INIT_USER_PASSWORD=$(hex 12)
LANGFUSE_INIT_PROJECT_PUBLIC_KEY=pk-lf-$(hex 16)
LANGFUSE_INIT_PROJECT_SECRET_KEY=sk-lf-$(hex 16)
EOF
  chmod 600 /opt/langfuse/.env
fi

TOKEN=$(curl -fsS -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
PUBLIC_IP=$(curl -fsS -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/public-ipv4)
if grep -q '^NEXTAUTH_URL=' /opt/langfuse/.env; then
  sed -i "s|^NEXTAUTH_URL=.*|NEXTAUTH_URL=http://${PUBLIC_IP}:3000|" /opt/langfuse/.env
else
  echo "NEXTAUTH_URL=http://${PUBLIC_IP}:3000" >> /opt/langfuse/.env
fi

docker compose -f /opt/langfuse/docker-compose.yml --env-file /opt/langfuse/.env -p langfuse up -d

lf_ok=0
for _ in $(seq 1 90); do
  if curl -fsS -m 5 http://127.0.0.1:3000/api/public/health >/dev/null; then
    lf_ok=1
    break
  fi
  sleep 5
done
if [ "$lf_ok" != 1 ]; then
  docker compose -f /opt/langfuse/docker-compose.yml --env-file /opt/langfuse/.env -p langfuse ps
  docker compose -f /opt/langfuse/docker-compose.yml --env-file /opt/langfuse/.env -p langfuse logs --tail 40
  exit 1
fi

set -a
# shellcheck disable=SC1091
. /opt/langfuse/.env
set +a

docker rm -f factored-backend >/dev/null 2>&1 || true
env_file=()
if [ -f /opt/factored.env ]; then
  env_file=(--env-file /opt/factored.env)
fi
docker create --name factored-backend --restart unless-stopped \
  --network factored -p 8000:8000 \
  "${env_file[@]}" \
  -e AIP_ENV=dev \
  -e AIP_AUTH_MODE=dev \
  -e AIP_DATABASE_URL=postgresql+asyncpg://aiplatform:aiplatform@factored-db:5432/aiplatform \
  -e AIP_BANK_DATABASE_URL=postgresql+asyncpg://bank_reader:bank_reader@factored-db:5432/bank \
  -e LANGFUSE_TRACING_ENABLED=true \
  -e LANGFUSE_BASE_URL=http://langfuse-web:3000 \
  -e LANGFUSE_PUBLIC_KEY="$LANGFUSE_INIT_PROJECT_PUBLIC_KEY" \
  -e LANGFUSE_SECRET_KEY="$LANGFUSE_INIT_PROJECT_SECRET_KEY" \
  -e AIP_TRACE_HASH_KEY="$SALT" \
  "$IMAGE" >/dev/null
docker network connect langfuse_default factored-backend
docker start factored-backend >/dev/null

ok=0
for _ in $(seq 1 20); do
  if curl -fsS -m 5 http://127.0.0.1:8000/ -o /dev/null; then
    ok=1
    break
  fi
  sleep 2
done
test "$ok" = 1

# HTTPS with a certificate browsers trust (Let's Encrypt, for this public IP).
# IP certificates last about six days, so a daily timer renews them. A new
# public IP on stop/start gets a new certificate. Port 80 only redirects.
mkdir -p /opt/factored-tls/acme /opt/factored-tls/letsencrypt /opt/factored-tls/letsencrypt-lib
if [ ! -f /opt/factored-tls/default.conf ]; then
  echo "missing /opt/factored-tls/default.conf (deploy/nginx.ec2.conf)" >&2
  exit 1
fi
if [ ! -s /opt/factored-tls/cert.pem ] || [ ! -s /opt/factored-tls/key.pem ]; then
  openssl req -x509 -newkey rsa:2048 -sha256 -days 2 -nodes \
    -keyout /opt/factored-tls/key.pem -out /opt/factored-tls/cert.pem \
    -subj "/CN=${PUBLIC_IP}" \
    -addext "subjectAltName=IP:${PUBLIC_IP}" 2>/dev/null
  chmod 600 /opt/factored-tls/key.pem
fi
cat > /usr/local/bin/factored-renew-tls <<'EOF'
#!/bin/bash
set -euo pipefail
TOKEN=$(curl -fsS -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 300")
PUBLIC_IP=$(curl -fsS -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/public-ipv4)
mkdir -p /opt/factored-tls/acme /opt/factored-tls/letsencrypt /opt/factored-tls/letsencrypt-lib
if ! docker image inspect certbot/certbot >/dev/null 2>&1; then
  docker pull certbot/certbot
fi
LIVE="/opt/factored-tls/letsencrypt/live/${PUBLIC_IP}"
need_issue=1
if [ -f "${LIVE}/fullchain.pem" ] \
  && openssl x509 -in "${LIVE}/fullchain.pem" -noout -checkend 172800 >/dev/null 2>&1 \
  && openssl x509 -in "${LIVE}/fullchain.pem" -noout -ext subjectAltName 2>/dev/null \
    | grep -q "IP Address:${PUBLIC_IP}"; then
  need_issue=0
fi
if [ "$need_issue" = 1 ]; then
  for _ in $(seq 1 30); do
    if curl -sS -m 2 -o /dev/null http://127.0.0.1/.well-known/acme-challenge/; then
      break
    fi
    sleep 1
  done
  docker run --rm --name factored-certbot \
    -v /opt/factored-tls/letsencrypt:/etc/letsencrypt \
    -v /opt/factored-tls/letsencrypt-lib:/var/lib/letsencrypt \
    -v /opt/factored-tls/acme:/var/www/acme \
    certbot/certbot certonly --webroot -w /var/www/acme \
    --preferred-profile shortlived \
    --ip-address "$PUBLIC_IP" \
    --cert-name "$PUBLIC_IP" \
    --non-interactive --agree-tos --register-unsafely-without-email \
    --force-renewal
else
  docker run --rm --name factored-certbot \
    -v /opt/factored-tls/letsencrypt:/etc/letsencrypt \
    -v /opt/factored-tls/letsencrypt-lib:/var/lib/letsencrypt \
    -v /opt/factored-tls/acme:/var/www/acme \
    certbot/certbot renew --non-interactive
fi
CONF="/opt/factored-tls/letsencrypt/renewal/${PUBLIC_IP}.conf"
if [ -f "$CONF" ] && grep -q '^renew_before_expiry' "$CONF"; then
  sed -i 's/^renew_before_expiry.*/renew_before_expiry = 2 days/' "$CONF"
fi
cat "${LIVE}/fullchain.pem" > /opt/factored-tls/cert.pem
cat "${LIVE}/privkey.pem" > /opt/factored-tls/key.pem
chmod 644 /opt/factored-tls/cert.pem
chmod 600 /opt/factored-tls/key.pem
if docker inspect factored-https >/dev/null 2>&1; then
  docker exec factored-https nginx -s reload
fi
EOF
chmod 755 /usr/local/bin/factored-renew-tls
cat > /etc/systemd/system/factored-renew-tls.service <<'EOF'
[Unit]
Description=Renew the BankBot TLS certificate
[Service]
Type=oneshot
ExecStart=/usr/local/bin/factored-renew-tls
EOF
cat > /etc/systemd/system/factored-renew-tls.timer <<'EOF'
[Unit]
Description=Daily BankBot TLS renewal
[Timer]
OnCalendar=daily
Persistent=true
[Install]
WantedBy=timers.target
EOF
cat > /etc/systemd/system/factored-renew-tls-boot.service <<'EOF'
[Unit]
Description=Refresh the BankBot TLS certificate after boot
After=docker.service
Wants=docker.service
[Service]
Type=oneshot
ExecStart=/usr/local/bin/factored-renew-tls
[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable --now factored-renew-tls.timer
systemctl enable factored-renew-tls-boot.service
if ! docker image inspect nginx:1.27-alpine >/dev/null 2>&1; then
  docker pull nginx:1.27-alpine
fi
docker rm -f factored-https >/dev/null 2>&1 || true
docker run -d --name factored-https --restart unless-stopped --network host \
  -v /opt/factored-tls/default.conf:/etc/nginx/conf.d/default.conf:ro \
  -v /opt/factored-tls/cert.pem:/etc/nginx/tls/server.crt:ro \
  -v /opt/factored-tls/key.pem:/etc/nginx/tls/server.key:ro \
  -v /opt/factored-tls/acme:/var/www/acme:ro \
  nginx:1.27-alpine >/dev/null
factored-renew-tls
https_ok=0
for _ in $(seq 1 15); do
  if curl -fsS -m 5 --resolve "${PUBLIC_IP}:443:127.0.0.1" "https://${PUBLIC_IP}/" -o /dev/null; then
    https_ok=1
    break
  fi
  sleep 2
done
test "$https_ok" = 1
echo "App:      https://${PUBLIC_IP}/"
echo "Langfuse: http://${PUBLIC_IP}:3000/"
