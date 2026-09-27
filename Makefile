.PHONY: check lint test test-postgres smoke run migrate docker-build

check: lint test

lint:
	uv run ruff check src tests scripts

test:
	uv run pytest -q

test-postgres:  # needs AIP_TEST_DATABASE_URL, e.g. from `docker compose up db`
	uv run pytest -q -m postgres

migrate:
	uv run alembic upgrade head

smoke:  # real provider from .env / AIP_PROVIDERS
	uv run python scripts/smoke.py

run:
	uv run uvicorn aiplatform.api.app:app --reload

docker-build:
	docker build -t aiplatform:dev .
