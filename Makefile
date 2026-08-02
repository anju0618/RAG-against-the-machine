.PHONY: install run debug clean fclean lint lint-strict test api

SRC_DIR = src
# TEST_DIR = tests/

install:
	uv sync

run:
	uv run python -m $(SRC_DIR)

debug:
	uv run python -m pdb -m $(SRC_DIR)

clean:
	@echo "Cleaning cache and temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".DS_Store" -delete
	find . -type f -name "*~" -delete
	rm -rf .mypy_cache .ruff_cache .pytest_cache .coverage htmlcov $(SRC_DIR)/__pycache__

fclean: clean
	@echo "Performing full clean..."
	rm -rf .venv
	find data/output -type f -name "*.json" -delete
	find data/processed -type f -not -name ".gitkeep" -delete

lint:
	uv run flake8 $(SRC_DIR)
	uv run mypy --explicit-package-bases --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs $(SRC_DIR)

lint-strict:
	uv run flake8 $(SRC_DIR)
	uv run mypy --strict $(SRC_DIR)

test:
	@echo "Running test suite with pytest..."
	# PYTHONPATH=. uv run python -m pytest $(TEST_DIR) -v

# Bonus 5: Local HTTP API
api:
	uv run python -m $(SRC_DIR) api
