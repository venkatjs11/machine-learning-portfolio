.PHONY: install train test lint format serve docker clean

install:
	pip install -r requirements.txt

train:
	python scripts/train.py --config configs/config.yaml --diagnostics

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	ruff check src/ tests/
	mypy src/ --ignore-missing-imports

format:
	black src/ tests/ scripts/
	ruff check src/ tests/ --fix

serve:
	uvicorn src.api.app:app --reload --port 8000

docker:
	docker build -t linear-regression .
	docker run -p 8000:8000 linear-regression

clean:
	rm -rf outputs/ __pycache__ .pytest_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
