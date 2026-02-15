.PHONY: install dev test lint format docker-up docker-down clean

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	pytest --cov=rehab_os tests/

lint:
	ruff check rehab_os/ tests/
	mypy rehab_os/

format:
	ruff check --fix rehab_os/ tests/
	ruff format rehab_os/ tests/

docker-up:
	docker compose up -d --build

docker-down:
	docker compose down

clean:
	rm -rf .mypy_cache .pytest_cache .ruff_cache dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
