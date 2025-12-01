.PHONY: help install install-dev clean run-cli run-web test lint format check venv venv-activate

help:
	@echo "Dialectica - Philosophy Q&A Platform"
	@echo ""
	@echo "Available commands:"
	@echo "  make venv             Create a virtual environment"
	@echo "  make venv-activate    Show commands to activate virtual environment"
	@echo "  make install          Install dependencies"
	@echo "  make install-dev      Install dependencies including dev tools"
	@echo "  make run-cli          Run the CLI tool"
	@echo "  make run-web          Run the Streamlit web interface"
	@echo "  make test             Run tests with pytest"
	@echo "  make lint             Run flake8 linter"
	@echo "  make format           Format code with black"
	@echo "  make check            Run lint and test"
	@echo "  make clean            Remove cache and temporary files"
	@echo "  make help             Show this help message"

venv:
	python3 -m venv philenv

venv-activate:
	@echo "To activate the virtual environment, run:"
	@echo "  source philenv/bin/activate"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install pytest black flake8

run-cli:
	python philosophyqa_cli.py

run-web:
	streamlit run philosopher_sl.py

test:
	pytest test_llm.py -v

lint:
	flake8 *.py --max-line-length=120

format:
	black *.py

check: lint test
	@echo "All checks passed!"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	rm -rf .coverage
