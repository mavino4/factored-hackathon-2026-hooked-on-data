.PHONY: check lint test test-postgres smoke eval eval-banking banking-cases compare-models bank-db run migrate docker-build

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

banking-cases:  # regenerate evals/banking.jsonl from the transcripts + the bank DB
	uv run python evals/import_transcripts.py ../data/call_transcripts
	uv run python evals/build_banking_cases.py

compare-models:  # banking evals on the local Ollama models (llama3.2:3b vs qwen2.5:7b)
	uv run python -m evals.compare

bank-db:  # load the simulated core-banking DB from ../data_clean (needs `docker compose up -d db`)
	uv run --with pandas --with pyarrow python scripts/load_bank_db.py

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
