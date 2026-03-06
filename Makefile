.PHONY: setup clean test lint ingest query sources help

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
CLI := $(VENV)/bin/legacylens

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup: ## Create venv and install package (use after clone or when stuck)
	rm -rf $(VENV)
	python3 -m venv $(VENV)
	$(PIP) install -e ".[dev]"
	@echo ""
	@echo "Done. Run commands with: make test, make query, or directly via .venv/bin/legacylens"

clean: ## Remove venv and build artifacts
	rm -rf $(VENV) build/ dist/ src/*.egg-info .pytest_cache
	find . -path ./data -prune -o -type d -name __pycache__ -print -exec rm -rf {} + 2>/dev/null || true

test: $(VENV) ## Run all tests
	$(PYTHON) -m pytest tests/ -v

test-python: $(VENV) ## Run Python chunker + source tests only
	$(PYTHON) -m pytest tests/test_python_chunker.py tests/test_sources.py tests/test_retrieve.py -v

sources: $(VENV) ## List all registered sources
	$(CLI) sources list

$(VENV):
	@echo "No venv found. Run 'make setup' first."
	@exit 1
