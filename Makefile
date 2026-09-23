lint:
	PYTHONPATH=. uv run ruff check ./src ./tests ./examples

test:
	PYTHONPATH=. uv run pytest
