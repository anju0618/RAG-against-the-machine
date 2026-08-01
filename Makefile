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
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf .pytest_cache
	rm -rf $(SRC_DIR)/__pycache__

fclean: clean
	rm -rf .venv
	# .gitkeep は残しつつ、生成された JSON ファイルだけを削除する
	find data/output -type f -name "*.json" -delete

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