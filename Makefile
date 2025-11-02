.PHONY: help install test lint format clean train api dashboard docs

help:  ## Show this help message
	@echo "T-CRIS - Temporal Cancer Recurrence Intelligence System"
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies with poetry
	poetry install

install-dev:  ## Install development dependencies
	poetry install --with dev

test:  ## Run tests with coverage
	poetry run pytest

test-quick:  ## Run tests without coverage
	poetry run pytest --no-cov

lint:  ## Run linters (flake8, mypy)
	poetry run flake8 src/ tests/
	poetry run mypy src/

format:  ## Format code with black and isort
	poetry run black src/ tests/ scripts/ dashboard/
	poetry run isort src/ tests/ scripts/ dashboard/

format-check:  ## Check code formatting
	poetry run black --check src/ tests/ scripts/ dashboard/
	poetry run isort --check src/ tests/ scripts/ dashboard/

clean:  ## Clean generated files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .coverage htmlcov .mypy_cache dist build

validate-data:  ## Validate data quality
	poetry run python scripts/validate_data.py

train:  ## Train all models
	poetry run python scripts/train_models.py

api:  ## Run FastAPI server
	poetry run uvicorn tcris.api.main:app --reload --host 0.0.0.0 --port 8000

dashboard:  ## Run Streamlit dashboard
	poetry run streamlit run dashboard/app.py

docs:  ## Build documentation
	cd docs && poetry run mkdocs build

docs-serve:  ## Serve documentation locally
	cd docs && poetry run mkdocs serve

notebook:  ## Start Jupyter notebook
	poetry run jupyter notebook notebooks/

setup:  ## Initial project setup
	cp .env.example .env
	poetry install
	@echo "Setup complete! Edit .env file with your configuration."

all: format lint test  ## Run formatting, linting, and tests
