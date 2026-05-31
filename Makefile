.PHONY: install test test-fast test-coverage lint lint-fix pre-commit-install docker docker-run clean

install:
	pip install -e .
	pip install -e ".[dev]"
	pip install pre-commit==3.7.1 ruff==0.4.0

test:
	python -m pytest tests/ -q --tb=short

test-fast:
	python -m pytest tests/ -q --tb=short -k "not backtest and not integration"

test-coverage:
	python -m pytest tests/ -q --tb=short --cov=kairos --cov-report=term-missing

lint:
	ruff check kairos/ tests/
	ruff format --check kairos/ tests/

lint-fix:
	ruff check --fix kairos/ tests/
	ruff format kairos/ tests/

pre-commit-install:
	pre-commit install

docker:
	test -f Dockerfile
	docker build -t kairos .

docker-run:
	docker run --rm kairos

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete
	rm -rf .pytest_cache
	rm -rf *.egg-info
