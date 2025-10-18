.PHONY: help install test lint format clean docker-build docker-up docker-down logs

help:
	@echo "Available commands:"
	@echo "  make install       - Install dependencies"
	@echo "  make test          - Run tests"
	@echo "  make test-cov      - Run tests with coverage"
	@echo "  make lint          - Run linters"
	@echo "  make format        - Format code"
	@echo "  make security      - Run security checks"
	@echo "  make clean         - Clean temporary files"
	@echo "  make docker-build  - Build Docker image"
	@echo "  make docker-up     - Start Docker containers"
	@echo "  make docker-down   - Stop Docker containers"
	@echo "  make logs          - Show Docker logs"
	@echo "  make run           - Run the application locally"

install:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install black isort flake8 pylint bandit safety

test:
	PYTHONPATH=. pytest -v

test-cov:
	PYTHONPATH=. pytest --cov=src --cov-report=html --cov-report=term -v
	@echo "Coverage report generated in htmlcov/index.html"

lint:
	@echo "Running flake8..."
	-flake8 src/ tests/ app.py config.py logging_config.py main.py
	@echo "Running pylint..."
	-pylint src/ --exit-zero

format:
	@echo "Running black..."
	black src/ tests/ app.py config.py logging_config.py main.py
	@echo "Running isort..."
	isort src/ tests/ app.py config.py logging_config.py main.py
	@echo "✅ Code formatted successfully"

format-check:
	@echo "Checking black..."
	black --check src/ tests/ app.py config.py logging_config.py main.py
	@echo "Checking isort..."
	isort --check-only src/ tests/ app.py config.py logging_config.py main.py

security:
	@echo "Running bandit..."
	-bandit -r src/ -f json -o bandit-report.json
	@echo "Running safety..."
	-safety check

clean:
	@echo "Cleaning temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.log" -delete
	rm -rf .pytest_cache .coverage htmlcov/ *.egg-info build/ dist/
	rm -rf tests/data tests/chroma_db_test tests/test_memory.db
	@echo "Clean complete"

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d
	@echo "Waiting for service to be healthy..."
	@timeout 60 bash -c 'until curl -f http://localhost:8000/health 2>/dev/null; do sleep 2; done' || echo "Service may not be ready yet"
	@echo "Service is running at http://localhost:8000"
	@echo "API docs: http://localhost:8000/docs"

docker-down:
	docker-compose down

docker-restart:
	docker-compose restart

logs:
	docker-compose logs -f

run:
	python app.py

setup-dirs:
	mkdir -p data logs vector_store

init: setup-dirs
	cp .env.example .env
	@echo "Created .env file. Please edit it with your API keys."
	@echo "Then run: make install"