install:
	uv sync --all-extras
	uv run pre-commit install

fix:
	uv run pre-commit run --all-files

test:
	uv run pytest --cov=src --cov-report=term-missing

clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null
	rm -rf .pytest_cache .mypy_cache .coverage htmlcov dist build *.egg-info

download-data:
	chmod +x scripts/download_dataset.sh
	./scripts/download_dataset.sh

train:
	uv run python3 -W ignore ./scripts/train_vit_clip.py

submit:
	chmod +x scripts/submit.sh
	./scripts/submit.sh
