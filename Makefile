.PHONY: check lint test test-postgres smoke eval run migrate docker-build

check: lint test

lint:
	uv run ruff check src tests scripts evals

test:
	uv run pytest -q

test-postgres:  # needs AIP_TEST_DATABASE_URL, e.g. from `docker compose up db`
	uv run pytest -q -m postgres

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
