# invoice-iq task runner (Unix / macOS / CI).
# Windows users: use the identical PowerShell tasks in tasks.ps1, e.g. `.\tasks.ps1 test`.
.DEFAULT_GOAL := help
PY := python

.PHONY: help install lint format type test check serve docker-build docker-up sync-metrics clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Create venv deps: install project + dev tooling
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]"

lint:  ## Ruff lint
	ruff check src tests

format:  ## Ruff auto-format + fix
	ruff format src tests
	ruff check --fix src tests

type:  ## mypy strict type-check
	mypy

test:  ## Run pytest
	pytest

check: lint type test  ## Lint + type + test (the full CI gate)

serve:  ## Run the FastAPI app locally
	$(PY) -m uvicorn invoice_iq.serving.app:create_app --factory --host 0.0.0.0 --port 8000 --reload

docker-build:  ## Build the API image (trains a local classifier inside the image)
	docker build -t invoice-iq:local .

docker-up:  ## Start the API with docker compose
	docker compose up --build

sync-metrics:  ## Inject models/metrics.json into the README (no-op until Phase 3)
	$(PY) scripts/sync_metrics_readme.py

clean:  ## Remove caches and build artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov build dist *.egg-info
