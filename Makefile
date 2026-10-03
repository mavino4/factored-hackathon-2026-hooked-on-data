.PHONY: intent-data classify-bench trace-report trace-score langfuse-mcp-env check lint test test-postgres smoke eval eval-banking eval-security banking-cases compare-models bank-db users users-relink run migrate docker-build lan prod tls langfuse-env langfuse-up langfuse-down

check: lint test

lint:
	uv run ruff check src tests scripts evals

test:
	uv run pytest -q

test-postgres:  # needs `docker compose up -d db`
	AIP_TEST_DATABASE_URL=$${AIP_TEST_DATABASE_URL:-postgresql+asyncpg://aiplatform:aiplatform@localhost:5432/aiplatform_test} \
	AIP_TEST_BANK_ADMIN_URL=$${AIP_TEST_BANK_ADMIN_URL:-postgresql://aiplatform:aiplatform@localhost:5432/postgres} \
	uv run pytest -q -m postgres

eval-banking:  # banking evals on the provider in .env / AIP_PROVIDERS (needs `make bank-db`)
	uv run python -m evals.run --dataset evals/banking.jsonl

eval-security:  # manipulation attempts, shared secrets and controls through the whole assistant (needs `make bank-db`)
	uv run python -m evals.run --dataset evals/security.jsonl

trace-report:  # speed, cost and behaviour of the turns traced in Langfuse; ARGS="--since 24h --check"
	uv run python -m evals.traces $(ARGS)

intent-data:  # test set from the suites + handwritten; train from Qwen (local) and Claude; see evals/README.md
	uv run --group classifiers python -m evals.intent_dataset test
	uv run --group classifiers python -m evals.intent_dataset generate --source qwen
	uv run --group classifiers python -m evals.intent_dataset generate --source claude
	uv run --group classifiers python -m evals.intent_dataset train

classify-bench:  # non-LLM intent classifiers vs the LLM on the test set; ARGS="--only rules ml --errors"
	uv run --group classifiers python -m evals.classify_bench $(ARGS)

trace-score:  # grade traced turns and write the grades to Langfuse; ARGS="--dry-run" or "--judge --sample 50"
	uv run python -m evals.score_traces $(ARGS)

langfuse-mcp-env:  # prints the export line .mcp.json needs: eval "$$(make -s langfuse-mcp-env)"
	@echo "export LANGFUSE_MCP_AUTH=$$(printf '%s:%s' "$$(grep '^LANGFUSE_PUBLIC_KEY=' .env | cut -d= -f2-)" "$$(grep '^LANGFUSE_SECRET_KEY=' .env | cut -d= -f2-)" | base64 -w0)"

banking-cases:  # regenerate evals/banking.jsonl from the transcripts + the bank DB
	uv run python evals/import_transcripts.py ../data/call_transcripts
	uv run python evals/build_banking_cases.py

compare-models:  # banking evals on the local Ollama models (llama3.2:3b vs qwen2.5:7b)
	uv run python -m evals.compare

bank-db:  # load the simulated core-banking DB from ../data_clean (needs `docker compose up -d db`)
	uv run --with pandas --with pyarrow python scripts/load_bank_db.py

users:  # password accounts for the Active customers (needs `make bank-db` and `make migrate`); passwords go to credentials/
	uv run --with pandas --with pyarrow python scripts/users.py provision

users-relink:  # after `make bank-db`: link the accounts to their customers again
	uv run python scripts/users.py relink

migrate:
	uv run alembic upgrade head

smoke:  # real provider from .env / AIP_PROVIDERS
	uv run python scripts/smoke.py

eval:  # quality baseline against the provider in .env / AIP_PROVIDERS
	uv run python evals/run.py --compare evals/baseline.json

run:
	uv run uvicorn --factory aiplatform.api.app:create_app --reload

docker-build:
	docker build -t aiplatform:dev .

lan: langfuse-up  # web UI over HTTPS for other devices on the LAN (password login, Claude, traced); see docker-compose.lan.yml
	@[ -f deploy/tls/server.crt ] || ./deploy/tls/gen-cert.sh
	AIP_RELEASE=$$(git rev-parse --short HEAD) docker compose -f docker-compose.yml -f docker-compose.lan.yml up -d --build
	@echo "Open https://$$(hostname -I | cut -d' ' -f1) from another device on this network"
	@echo "First time on a device: install http://$$(hostname -I | cut -d' ' -f1):8000/ca.crt as a trusted authority"

prod:  # `make lan` with AIP_ENV=production (needs AIP_TRACE_HASH_KEY in .env)
	AIP_ENV=production $(MAKE) lan

tls:  # (re)issue the LAN demo's HTTPS certificate, e.g. after this machine's IP changed
	./deploy/tls/gen-cert.sh
	-docker compose -f docker-compose.yml -f docker-compose.lan.yml exec lb nginx -s reload

langfuse-env:  # once: deploy/langfuse/.env with random secrets and the project API keys
	./deploy/langfuse/gen-env.sh

langfuse-up:  # self-hosted Langfuse (tracing UI) on http://localhost:3000
	docker compose -f deploy/langfuse/docker-compose.yml --env-file deploy/langfuse/.env up -d

langfuse-down:
	docker compose -f deploy/langfuse/docker-compose.yml --env-file deploy/langfuse/.env down
