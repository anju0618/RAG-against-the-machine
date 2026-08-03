.PHONY: install run debug clean fclean lint lint-strict test api index index-fast

SRC_DIR = src
# TEST_DIR = tests/

install:
	uv sync

run:
	uv run python -m $(SRC_DIR)

debug:
	uv run python -m pdb -m $(SRC_DIR)

# --- Indexing shortcuts -------------------------------------------------
# フルインデックス作成（BM25 + セマンティック埋め込み）。
# 採点・提出に使う最終的なインデックスは必ずこちらで作ること。
# 所要時間の内訳（Lexical / Vector / Total）が標準出力に表示される。
index:
	uv run python -m $(SRC_DIR) index --max_chunk_size 2000

# BM25のみの高速インデックス作成（セマンティック埋め込みをスキップ）。
# チャンキングやCLIの動作を素早く試したいときの開発用モード。
# recall@5の合格基準はBM25単体では届かないため、採点用インデックスの
# 代わりにこれを使わないこと。`make index` を後で実行すれば、
# 埋め込み未計算のチャンクだけを自動的に補完できる。
index-fast:
	uv run python -m $(SRC_DIR) index --max_chunk_size 2000 --skip_vector

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
	@if [ -d data/output ]; then \
		find data/output -type f -name "*.json" -delete; \
	fi
	@if [ -d data/processed ]; then \
		find data/processed -type f -not -name ".gitkeep" -delete; \
	fi

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
# api.py は FastAPI アプリなので、CLI (`python -m src`) 経由ではなく
# uvicorn から直接起動する。
# 以前の `uv run python -m $(SRC_DIR) api` は、RAGCLI に存在しない
# "api" サブコマンドを呼び出そうとしており、実際には動作しなかった。
api:
	uv run uvicorn $(SRC_DIR).api:app --reload
