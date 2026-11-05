# ManualMind Makefile
.PHONY: help setup run eval test docker-build docker-run clean

help:
	@echo "ManualMind Build and Run Automation"
	@echo "==================================="
	@echo "make setup        - Install dependencies and pre-download all NLP models"
	@echo "make run          - Launch ManualMind FastAPI server on port 7860"
	@echo "make eval         - Run full evaluation benchmark and regenerate figures"
	@echo "make test         - Run 100+ pytest test suite"
	@echo "make docker-build - Build self-contained Docker container image"
	@echo "make docker-run   - Run Docker container on http://localhost:7860"
	@echo "make clean        - Remove temporary pycache and build artifacts"

setup:
	pip install --upgrade pip
	pip install torch --index-url https://download.pytorch.org/whl/cpu
	pip install -r requirements.txt
	python scripts/download_models.py

run:
	uvicorn app.main:app --host 0.0.0.0 --port 7860 --reload

eval:
	python eval/run_all_evals.py

test:
	pytest -v

docker-build:
	docker build -t manualmind:latest .

docker-run:
	docker run -p 7860:7860 --name manualmind manualmind:latest

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
