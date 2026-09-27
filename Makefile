.PHONY: check lint test run docker-build

check: lint test

lint:
	uv run ruff check src tests

test:
	uv run pytest -q

run:
	uv run uvicorn aiplatform.api.app:app --reload

docker-build:
	docker build -t aiplatform:dev .
