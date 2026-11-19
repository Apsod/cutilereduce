.PHONY: lint test quality typecheck

lint:
	PYTHONPATH=. uv run ruff check ./src ./tests ./examples

test:
	PYTHONPATH=. uv run pytest

typecheck:
	PYTHONPATH=. uv run ty check ./src

quality: lint test typecheck
