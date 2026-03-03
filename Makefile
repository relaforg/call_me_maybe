MYPY_FLAGS = --warn-return-any \
			 --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs \
			 --check-untyped-defs

run:
	uv run python -m src $(ARGS)

install:
	uv pip install -r requirements.txt

debug:
	uv run python -m pdb -m src

clean:
	rm -fdr  __pycache__ .mypy_cache

lint:
	uv run python -m flake8 src
	uv run python -m mypy src $(MYPY_FLAGS)
