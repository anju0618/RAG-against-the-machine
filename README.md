*This project has been created as part of the 42 curriculum by amakino.*

# RAG against the machine

## Description
This project aims to build a custom Retrieval-Augmented Generation (RAG) system from scratch, targeting a massive software codebase (`vllm-0.10.1`). When asking an AI model (specifically, the lightweight `Qwen3-0.6B` model) about how the code works, it lacks inherent knowledge of it. Therefore, this project implements a complete pipeline that first retrieves relevant code snippets using custom search strategies, then passes them to the AI to generate accurate, grounded answers.

---

## Instructions

### Installation & Dependencies
This project uses `uv` as the package manager.
```bash
uv venv && uv sync
```

### Makefile Rules
- `make install`: Installs dependencies using `uv`.
- `make run`: Executes the main script.
- `make debug`: Runs the main script in debug mode.
- `make clean`: Removes temporary files and caches.
- `make lint`: Runs `flake8` and `mypy` checks.
- `make lint-strict`: Runs strict static analysis.

### CLI Commands
- Index the codebase:
  ```bash
  uv run python -m src index --max_chunk_size 2000
  ```
- Search for a single query:
  ```bash
  uv run python -m src search "How to configure the OpenAI server?" --k 5
  ```
- Search over a dataset:
  ```bash
  uv run python -m src search_dataset --dataset_path data/datasets/AnsweredQuestions/dataset_code_public.json --k 10 --save_directory data/output/search_results
  ```
- Answer a single query:
  ```bash
  uv run python -m src answer "How does the retriever handle tokenization?" --k 5
  ```
- Answer a dataset:
  ```bash
  uv run python -m src answer_dataset --student_search_results_path data/output/search_results/dataset_code_public.json --save_directory data/output/answers
  ```

---

## Resources & AI Usage

### References
- [What is RAG? (AWS)](https://aws.amazon.com/jp/what-is/retrieval-augmented-generation/)
- [Introduction to RAG (Qiita)](https://qiita.com/Junpei_Takagi/items/f82d31323f00ad895579)
- [RAG Implementation Strategies](https://qiita.com/jw-automation/items/045917be7b558509fdf2)
- [Python UUID Guide](https://qiita.com/shimajiri/items/315d458d3796ed4a60da)
- [Hugging Face Transformers](https://qiita.com/ski2_1116/items/f74e7b97008663d0702d)
- [PyTorch Beginner Guide](https://qiita.com/wooooo/items/f7d439e166ff664ad47c)

### AI Usage Description
AI tools were used during the development of this project to assist with drafting boilerplate code for Pydantic models, optimizing regular expressions for chunking strategies, and structuring prompt templates for the local LLM. All generated code and logic have been thoroughly reviewed, tested, and understood by the author.

---

## System Architecture & Components

### 1. `models.py`
Defines data structures using Pydantic for clean type safety across all pipeline stages:
- **`MinimalSource`**: Represents the minimum unit of a retrieved source, holding `file_path`, `first_character_index`, and `last_character_index`.
- **`UnansweredQuestion` / `AnsweredQuestion`**: Represents input questions and questions paired with ground-truth sources and answers.
- **`MinimalSearchResults` / `MinimalAnswer`**: Combines questions with retrieved sources and generated answers.
- **`StudentSearchResults` / `StudentSearchResultsAndAnswer`**: Defines the overall JSON structure for batch processing outputs.

### 2. `indexer.py`
Builds and persists the search index from the raw corpus under `data/processed/chunks.json`.
- **Chunking Strategies**: Implements `chunk_python_code` (splitting by lines for `.py` files) and `chunk_markdown_text` (splitting by paragraphs `\n\n` for Markdown).
- **Safety Guard (`_safe_append`)**: Ensures chunks never exceed `max_chunk_size` (default: 2000 characters).
- **BM25 Indexing**: Removes stopwords (`a`, `the`, `is`, etc.) and builds a `BM25Okapi` sparse retrieval engine.

### 3. `retriever.py`
Loads the pre-built index and searches for the most relevant source locations given a query.
- **`load_index`**: Initializes the `BM25Okapi` engine from `chunks.json`.
- **`search`**: Tokenizes the query, removes stopwords, and fetches the top-$k$ matching chunks using BM25.
- **`search_dataset`**: Processes a JSON dataset of multiple questions in batch.

### 4. `generator.py`
Loads a local causal language model (`Qwen/Qwen2.5-0.5B-Instruct`) and generates grounded natural language answers.
- **`_load_chunk`**: Pinpoint-extracts text slices from physical files using character indices.
- **`_generate_prompt`**: Constructs system and user prompts instructing the model to rely *only* on the provided context without hallucinating.
- **`generate_answers`**: Batches prompt tokens, calls model generation, strips input prompts from token outputs, and decodes clean text responses.

### 5. `__main__.py`
Exposes the CLI via Python Fire, wiring together Indexer, Retriever, and Generator components.

---
---

# RAG against the machine (日本語版)

## Description
このプロジェクトは、配布された `vllm-0.10.1` という大規模なソフトウェアのソースコード一式を対象に、コードの仕様や内容について正確に答えられるRAGシステム（Retrieval-Augmented Generation）を自作することです。

AI（小型モデルの `Qwen3-0.6B` を使用）に直接「このコードはどう動くの？」と聞いても内部のコードを知らないため、まずは自分で書いたプログラムで関連するコードの断片を検索（Retrieve）し、それをAIに読ませて回答（Generate）させるパイプラインを構築します。

---

## Instructions

### インストールと依存関係
パッケージマネージャーとして `uv` を使用します。
```bash
uv venv && uv sync
```

### Makefile ルール
- `install`: `uv` を使って依存関係をインストールする。
- `run`: メインスクリプトを実行する。
- `debug`: デバッグモードで実行する。
- `clean`: 一時ファイルやキャッシュを削除する。
- `lint`: `flake8` と `mypy` による静的解析を実行する。
- `lint-strict`: strictモードで静的解析を実行する。

### CLI コマンド
- インデックスの構築:
  ```bash
  uv run python -m src index --max_chunk_size 2000
  ```
- 単一クエリの検索:
  ```bash
  uv run python -m src search "How to configure the OpenAI server?" --k 5
  ```
- データセットの一括検索:
  ```bash
  uv run python -m src search_dataset --dataset_path data/datasets/AnsweredQuestions/dataset_code_public.json --k 10 --save_directory data/output/search_results
  ```
- 単一クエリの回答生成:
  ```bash
  uv run python -m src answer "How does the retriever handle tokenization?" --k 5
  ```
- データセットの一括回答生成:
  ```bash
  uv run python -m src answer_dataset --student_search_results_path data/output/search_results/dataset_code_public.json --save_directory data/output/answers
  ```

---

## Resources & AI Usage

### 参考文献
- [RAG (検索拡張生成) とは何ですか? (AWS)](https://aws.amazon.com/jp/what-is/retrieval-augmented-generation/)
- [【生成AI入門】「RAG」をできるだけわかりやすく解説してみる](https://qiita.com/Junpei_Takagi/items/f82d31323f00ad895579)
- [RAGの実装戦略まとめ](https://qiita.com/jw-automation/items/045917be7b558509fdf2)
- [Python3でUUIDを生成する](https://qiita.com/shimajiri/items/315d458d3796ed4a60da)
- [HuggingFaceのTransformerライブラリを使ってみよう](https://qiita.com/ski2_1116/items/f74e7b97008663d0702d)
- [PyTorch初心者ガイド](https://qiita.com/wooooo/items/f7d439e166ff664ad47c)
- [エンジニアのための AI 基礎 - ベクトル埋め込みと仲良くなりたい！](https://qiita.com/yuji-arakawa/items/14a26f038740e7b89f3c)

### AIの利用について
本プロジェクトの開発にあたり、AIツールはPydanticモデルのボイラープレートコードの記述、チャンキング処理における正規表現の最適化、およびローカルLLMへのプロンプトテンプレートの構成補助として活用しました。生成されたコードやロジックはすべて著者自身で詳細にレビュー、テストしました

---

## システムアーキテクチャと各コンポーネント

### 1. `models.py`
すべてのコンポーネント間でやり取りされるデータの形を、Pydanticを使って厳密に型安全に定義しているファイルです。
- **`MinimalSource`**: 検索された根拠（ソース）の最小単位です。ファイルのパス（`file_path`）と文字インデックスの範囲（`first_character_index`, `last_character_index`）を保持し、評価システム（Moulinette）との共通言語になります。
- **`UnansweredQuestion` / `AnsweredQuestion`**: 入力用質問と、正解ソース・模範解答がセットになった質問データの構造です。
- **`MinimalSearchResults` / `MinimalAnswer`**: 質問に対する検索結果と、LLMによる回答を組み合わせたモデルです。
- **`StudentSearchResults` / `StudentSearchResultsAndAnswer`**: データセット全体を処理したときに出力される最終的なJSONの全体構造を規定します。

### 2. `indexer.py`
大量のvLLMのソースコードやドキュメントを読み込み、高速に検索できる「インデックス」を構築してディスク（`data/processed/chunks.json`）に保存するクラスです。
- **2つのチャンキング戦略**: 
  - `chunk_python_code`: Pythonコード（`.py`）は行（改行）単位で意味のまとまりを意識して分割します。
  - `chunk_markdown_text`: Markdownドキュメントは段落（空行 `\n\n`）単位で自然に分割します。
- **安全装置（`_safe_append`）**: どんなに長いファイルであっても、1チャンクの最大文字数（`max_chunk_size`、デフォルト2000文字）を絶対に超えないように強制分割し、位置情報を正確に記録します。
- **BM25インデックスの構築**: ストップワード（`a`, `the`, `is` など）を除去した上で、`BM25Okapi` を使ってキーワード検索用の数理モデルを構築します。

### 3. `retriever.py`
作成されたインデックスをメモリ上に読み込み、ユーザーの質問に対して最も関連性の高いソースの場所を探し出すクラスです。
- **`load_index`**: 保存されたインデックスを読み込み、BM25検索エンジンを再初期化します。
- **`search`**: 質問文からストップワードを除去してトークン化し、`BM25Okapi` を使って上位 $k$ 件のチャンクインデックスを取得、`MinimalSource` に詰めて返します。
- **`search_dataset`**: 複数問の質問が含まれるJSONファイルを読み込み、一括で検索結果（`StudentSearchResults`）を組み立てます。

### 4. `generator.py`
検索されたソースの場所をもとに実ファイルをピンポイントで読み込み、ローカルLLM（`Qwen/Qwen2.5-0.5B-Instruct`）に渡して自然言語の回答を生成するクラスです。
- **`_load_chunk`**: ファイルパスと文字インデックスを使って実ファイルから該当部分のテキストを切り出します。
- **`_generate_prompt`**: 「コンテキストにのみ基づいて答えよ」「外部知識や嘘は使ってはならない」という厳格なシステムプロンプトを設定し、検索結果と質問を結合してLLMへの入力を作成します。
- **`generate_answers`**: バッチ単位でLLMの `generate` を呼び出し、プロンプト部分を除外して純粋な回答テキストのみをデコードします。

### 5. `__main__.py`
Python Fire を用いてターミナルからのコマンド入力を受け付け、上記の各クラス（Indexer, Retriever, Generator）を適切な順序で動かすエントリーポイントです。