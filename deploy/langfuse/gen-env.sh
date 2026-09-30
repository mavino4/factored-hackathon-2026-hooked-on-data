#!/usr/bin/env sh
# Write deploy/langfuse/.env from .env.example with random secrets (hex: safe in URLs).
set -eu
dir=$(dirname "$0")
[ -f "$dir/.env" ] && { echo "$dir/.env already exists; not overwriting" >&2; exit 1; }
hex() { openssl rand -hex "$1"; }
sed -e "s|^NEXTAUTH_SECRET=.*|NEXTAUTH_SECRET=$(hex 32)|" \
    -e "s|^SALT=.*|SALT=$(hex 32)|" \
    -e "s|^ENCRYPTION_KEY=.*|ENCRYPTION_KEY=$(hex 32)|" \
    -e "s|^POSTGRES_PASSWORD=.*|POSTGRES_PASSWORD=$(hex 24)|" \
    -e "s|^CLICKHOUSE_PASSWORD=.*|CLICKHOUSE_PASSWORD=$(hex 24)|" \
    -e "s|^MINIO_ROOT_PASSWORD=.*|MINIO_ROOT_PASSWORD=$(hex 24)|" \
    -e "s|^REDIS_AUTH=.*|REDIS_AUTH=$(hex 24)|" \
    -e "s|^LANGFUSE_INIT_USER_PASSWORD=.*|LANGFUSE_INIT_USER_PASSWORD=$(hex 12)|" \
    -e "s|^LANGFUSE_INIT_PROJECT_PUBLIC_KEY=.*|LANGFUSE_INIT_PROJECT_PUBLIC_KEY=pk-lf-$(hex 16)|" \
    -e "s|^LANGFUSE_INIT_PROJECT_SECRET_KEY=.*|LANGFUSE_INIT_PROJECT_SECRET_KEY=sk-lf-$(hex 16)|" \
    "$dir/.env.example" > "$dir/.env"
chmod 600 "$dir/.env"
echo "Wrote $dir/.env. Add these to the app's .env to send traces:"
echo "  LANGFUSE_TRACING_ENABLED=true"
echo "  LANGFUSE_BASE_URL=http://localhost:3000"
grep -E '^LANGFUSE_INIT_PROJECT_(PUBLIC|SECRET)_KEY=' "$dir/.env" | sed -e 's/^LANGFUSE_INIT_PROJECT_/  LANGFUSE_/'
echo "UI login: $(grep '^LANGFUSE_INIT_USER_EMAIL=' "$dir/.env" | cut -d= -f2) / see LANGFUSE_INIT_USER_PASSWORD in $dir/.env"
