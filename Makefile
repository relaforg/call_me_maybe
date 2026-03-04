MYPY_FLAGS := --warn-return-any \
			 --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs \
			 --check-untyped-defs
SRC_DIRECTORY := src

run:
	uv run python -m src $(ARGS)

install:
	uv sync

debug:
	uv run python -m pdb -m src

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	rm -rf .mypy_cache

lint:
	uv run python -m flake8 $(SRC_DIRECTORY)
	uv run python -m mypy $(MYPY_FLAGS)

.PHONY: run install debug clean lint
