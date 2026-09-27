.PHONY: check lint test test-postgres run migrate docker-build

check: lint test

lint:
	uv run ruff check src tests

test:
	uv run pytest -q

test-postgres:  # needs AIP_TEST_DATABASE_URL, e.g. from `docker compose up db`
	uv run pytest -q -m postgres

migrate:
	uv run alembic upgrade head

run:
	uv run uvicorn aiplatform.api.app:app --reload

docker-build:
	docker build -t aiplatform:dev .
