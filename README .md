*This project has been created as part of the 42 curriculum by amakino.*

---

## Description / 概要

**EN** — This project implements a **Retrieval-Augmented Generation (RAG) system** over the vLLM codebase. Instead of relying on a language model's frozen training data, the system indexes a real codebase (Python source files and Markdown documentation), retrieves the most relevant snippets for a given question, and feeds them to a small local LLM (`Qwen/Qwen3-0.6B`) so it can answer **grounded** in the retrieved evidence rather than hallucinating.

The pipeline has four stages, matching the classic RAG breakdown:
1. **Indexing** — chunk the corpus and build searchable indices.
2. **Retrieval** — given a question, return the top-k most relevant source locations.
3. **Augmenting** — load the actual text for those locations and build a prompt within the model's context budget.
4. **Generating** — ask the LLM to answer using only that context.

Retrieval quality is measured with **recall@k**, and the system must reach at least 80% recall@5 on documentation questions and 50% on code questions.

**JA** — 本プロジェクトは、vLLMのコードベースを対象にした **Retrieval-Augmented Generation (RAG) システム** の実装です。言語モデルが学習時点の知識に固定されてしまう問題を避けるため、実際のコードベース（PythonソースファイルとMarkdownドキュメント）をインデックス化し、質問に対して最も関連度の高いスニペットを検索したうえで、小型のローカルLLM（`Qwen/Qwen3-0.6B`）にそれらを渡すことで、幻覚（ハルシネーション）ではなく検索結果に根拠を持った（グラウンディングされた）回答を生成します。

パイプラインは古典的なRAGの4段階構成に沿っています：
1. **インデックス化** — コーパスをチャンクに分割し、検索可能なインデックスを構築する
2. **検索** — 質問に対して、関連度の高い上位k件のソース位置を返す
3. **拡張（コンテキスト構築）** — 該当箇所の実テキストを読み込み、モデルのトークン予算内でプロンプトを組み立てる
4. **生成** — そのコンテキストのみを根拠にLLMへ回答させる

検索の質は **recall@k** で測定され、ドキュメント系の質問でrecall@5が80%以上、コード系の質問で50%以上に達することが求められます。

---

## Instructions / セットアップと実行方法

**EN** — This project uses **uv** as the package/project manager. Only `uv sync` is guaranteed to work in the evaluation environment, so please avoid adding dependencies through any other tool.

```bash
# 1. Install dependencies
uv sync

# 2. Build the index (chunks the corpus under data/raw/ and writes to data/processed/)
uv run python -m src index --max_chunk_size 2000

# 3. Run a single search query
uv run python -m src search "How to configure the OpenAI server?" --k 5

# 4. Run search over a full dataset (writes a StudentSearchResults JSON)
uv run python -m src search_dataset \
    --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
    --k 10 \
    --save_directory data/output/search_results/UnansweredQuestions

# 5. Ask a single question end-to-end (search + LLM answer)
uv run python -m src answer "How to configure the OpenAI server?" --k 5

# 6. Generate answers for a whole dataset of already-searched results
uv run python -m src answer_dataset \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --save_directory data/output/search_results_and_answer/UnansweredQuestions

# 7. (Optional, for your own iteration) Evaluate recall@k against ground truth
uv run python -m src evaluate \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
```

A `Makefile` is provided with the standard rules: `install`, `run`, `debug`, `clean`, `lint`, and `lint-strict`.

```bash
make install   # uv sync
make run       # run the CLI's default entry point
make debug     # run under pdb
make lint      # flake8 . && mypy . (with the required flags)
make clean     # remove __pycache__ / .mypy_cache
```

**Official grading** of retrieval quality is performed by the **moulinette** executable (not by this project's own `evaluate` command). The moulinette is a separate, provided tool — see the [Resources](#resources--参考資料) section below for its usage. This project never imports or calls the moulinette internally.

**JA** — 本プロジェクトはパッケージ／プロジェクト管理に **uv** を使用します。評価環境では `uv sync` のみが動作保証されているため、他のツールで依存関係を追加しないでください。

```bash
# 1. 依存関係のインストール
uv sync

# 2. インデックスの構築（data/raw/ 配下をチャンク化し data/processed/ に保存）
uv run python -m src index --max_chunk_size 2000

# 3. 単一クエリでの検索
uv run python -m src search "How to configure the OpenAI server?" --k 5

# 4. データセット全体での検索（StudentSearchResults形式のJSONを出力）
uv run python -m src search_dataset \
    --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
    --k 10 \
    --save_directory data/output/search_results/UnansweredQuestions

# 5. 単一質問への一括回答（検索＋LLM回答生成）
uv run python -m src answer "How to configure the OpenAI server?" --k 5

# 6. 検索済みデータセット全体への回答生成
uv run python -m src answer_dataset \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --save_directory data/output/search_results_and_answer/UnansweredQuestions

# 7.（任意・自己検証用）ground truthとの recall@k を確認
uv run python -m src evaluate \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
```

`Makefile` には標準ルール（`install` / `run` / `debug` / `clean` / `lint` / `lint-strict`）を用意しています。

**正式な採点**は本プロジェクト自身の `evaluate` コマンドではなく、別途提供される **moulinette** 実行ファイルによって行われます。moulinetteの使い方は下記の[Resources](#resources--参考資料)セクションを参照してください。本プロジェクトのコードは moulinette を一切 import・呼び出ししません。

---

## Resources / 参考資料

**EN** — References used to understand and implement this project:
- [BM25 / Okapi BM25 — Wikipedia](https://en.wikipedia.org/wiki/Okapi_BM25) — background on the lexical ranking function used for keyword search.
- [rank_bm25 (PyPI)](https://pypi.org/project/rank-bm25/) — the BM25 implementation used in `indexer.py` / `retriever.py`.
- [Sentence-Transformers documentation](https://www.sbert.net/) — for the `all-MiniLM-L6-v2` embedding model used in semantic search.
- [Reciprocal Rank Fusion (RRF) — original paper by Cormack et al., 2009](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) — the method used to combine BM25 and semantic rankings.
- [Hugging Face Transformers documentation](https://huggingface.co/docs/transformers) — for loading and running `Qwen/Qwen3-0.6B`.
- [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-0.6B) — the mandatory LLM used for answer generation.
- [Python Fire documentation](https://github.com/google/python-fire) — used to build the CLI.
- [FastAPI documentation](https://fastapi.tiangolo.com/) — used for the bonus local HTTP API.
- **Moulinette README** (provided alongside this project) — describes the official evaluation CLI, its arguments (`--k`, `--max_context_length`), input/output JSON formats, and the pass thresholds (Recall@5 ≥ 50% for code, ≥ 80% for docs).

**How AI was used** — An AI assistant (Claude) was used throughout the project in a supervised, reviewed way, never as a black box:
- **Code review and bug-finding**: reviewing the initial implementation against the project subject to catch mismatches with the requirements (e.g. the mandatory model name, a `k` parameter not being propagated end-to-end from the API layer down to prompt construction, and a hardcoded corpus folder name that would break if the shipped folder was renamed).
- **Fixing the identified issues**: applying the corrections above directly in `generator.py`, `api.py`, and `indexer.py`.
- **Commenting / documentation**: adding thorough, explanatory comments (in Japanese, matching the existing codebase's language) to every module, explaining *why* each design choice was made (e.g. why RRF, why inclusive character indices, why incremental indexing works the way it does) rather than just *what* the code does.
- **README drafting**: drafting this README structure and bilingual (EN/JA) content based on the project subject's explicit requirements.

Every AI-assisted change was reviewed line by line before being kept, and no code was copy-pasted without understanding it, in line with the "AI Instructions" chapter of the project subject.

**JA** — 本プロジェクトの理解・実装にあたって参考にした資料：
- [BM25 / Okapi BM25 — Wikipedia](https://en.wikipedia.org/wiki/Okapi_BM25) — キーワード検索に使うランキング関数の背景。
- [rank_bm25（PyPI）](https://pypi.org/project/rank-bm25/) — `indexer.py` / `retriever.py` で使用しているBM25実装。
- [Sentence-Transformers ドキュメント](https://www.sbert.net/) — セマンティック検索に使用する `all-MiniLM-L6-v2` 埋め込みモデルについて。
- [Reciprocal Rank Fusion (RRF) — Cormack et al., 2009の原論文](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) — BM25とセマンティック検索のランキングを統合する手法。
- [Hugging Face Transformers ドキュメント](https://huggingface.co/docs/transformers) — `Qwen/Qwen3-0.6B` のロード・実行方法について。
- [Qwen3 モデルカード](https://huggingface.co/Qwen/Qwen3-0.6B) — 回答生成に使用する必須のLLM。
- [Python Fire ドキュメント](https://github.com/google/python-fire) — CLI構築に使用。
- [FastAPI ドキュメント](https://fastapi.tiangolo.com/) — ボーナスのローカルHTTP APIに使用。
- **Moulinette README**（本プロジェクトと合わせて提供）— 公式評価CLIの使い方、引数（`--k`、`--max_context_length`）、入出力JSON形式、合格基準（コード: Recall@5 ≥ 50%、ドキュメント: Recall@5 ≥ 80%）について説明している。

**AIの利用方法** — 本プロジェクトでは、AIアシスタント（Claude）をブラックボックスとしてではなく、常にレビューを伴う形で活用しました：
- **コードレビューとバグ発見**：初期実装を課題要件と突き合わせ、要件との不一致（必須モデル名の違い、APIレイヤーからプロンプト構築までkパラメータが正しく伝播していなかった点、コーパスのフォルダ名がハードコードされておりフォルダ名変更時に壊れる点など）を洗い出す作業に使用。
- **指摘された問題の修正**：上記の修正を `generator.py` / `api.py` / `indexer.py` に直接適用。
- **コメント・ドキュメントの追加**：既存コードベースの言語に合わせ、日本語で「何をしているか」だけでなく「なぜその設計にしたか」（RRFを使う理由、文字インデックスをinclusiveにする理由、差分インデックスの仕組みなど）を説明する丁寧なコメントを各モジュールに追加。
- **README草案作成**：課題要件に基づき、本READMEの構成とEN/JA併記の内容を作成。

AIによる変更はすべて1行ずつレビューしたうえで採用しており、理解せずにそのままコピー＆ペーストしたコードはありません（課題の「AI Instructions」章の方針に従っています）。

---

## System Architecture / システムアーキテクチャ

**EN** — The system is composed of five main components, each with a single responsibility:

| Component | File | Responsibility |
|---|---|---|
| `Indexer` | `src/indexer.py` | Walks `data/raw/`, chunks each file (two strategies), computes BM25 tokens and semantic embeddings, persists everything under `data/processed/` (incremental). |
| `Retriever` | `src/retriever.py` | Loads the persisted index once at startup, exposes `search()` / `search_dataset()`, combines BM25 + semantic rankings via RRF, caches repeated queries. |
| `Generator` | `src/generator.py` | Loads `Qwen/Qwen3-0.6B` once, builds a grounded chat prompt from retrieved sources, generates the answer, exposes `answer_dataset()` for end-to-end batch runs. |
| `RAGCLI` | `src/__main__.py` | Python Fire-based CLI exposing `index`, `search`, `search_dataset`, `answer`, `answer_dataset`, `evaluate`. |
| `api` (bonus) | `src/api.py` | FastAPI wrapper exposing `/search` and `/answer` over HTTP, reusing the same `Retriever`/`Generator` singletons as the CLI. |

Data flows strictly in one direction: `Indexer` → `Retriever` → `Generator` → (CLI output JSON or HTTP response), and every object crossing a stage boundary is a `pydantic` model from `src/models.py`, guaranteeing the JSON contract expected by the moulinette.

**JA** — システムは、それぞれ単一の責務を持つ5つの主要コンポーネントで構成されています：

| コンポーネント | ファイル | 役割 |
|---|---|---|
| `Indexer` | `src/indexer.py` | `data/raw/` を走査し、各ファイルを（2種類の戦略で）チャンク化、BM25用トークンとセマンティック埋め込みを計算し、`data/processed/` に永続化する（差分インデックス対応）。 |
| `Retriever` | `src/retriever.py` | 起動時に一度だけ永続化済みインデックスをロードし、`search()` / `search_dataset()` を公開。BM25とセマンティック検索のランキングをRRFで統合し、同一クエリはキャッシュする。 |
| `Generator` | `src/generator.py` | `Qwen/Qwen3-0.6B` を一度だけロードし、検索結果から根拠のあるチャットプロンプトを組み立てて回答生成。一括実行用の `answer_dataset()` も公開。 |
| `RAGCLI` | `src/__main__.py` | Python Fireベースのcliで `index` / `search` / `search_dataset` / `answer` / `answer_dataset` / `evaluate` を公開。 |
| `api`（ボーナス） | `src/api.py` | 同じ `Retriever` / `Generator` のインスタンスを再利用しつつ、`/search` と `/answer` をHTTP経由で公開するFastAPIラッパー。 |

データは `Indexer` → `Retriever` → `Generator` → （CLI出力JSON または HTTPレスポンス）という一方向のみに流れ、各ステージ間でやり取りされるオブジェクトはすべて `src/models.py` で定義された `pydantic` モデルであるため、moulinetteが期待するJSON契約を確実に満たします。

---

## Chunking Strategy / チャンキング戦略

**EN** — Two distinct chunking strategies are implemented, since code and prose break apart differently:

- **Python code (`chunk_python_code`)**: chunks are built line-by-line. Lines are accumulated until adding the next line would exceed `max_chunk_size`; the chunk is then flushed and a new one started. This avoids splitting in the middle of a line (which could otherwise cut through an identifier or a string literal).
- **Markdown / text (`chunk_markdown_text`)**: chunks are built paragraph-by-paragraph, splitting on blank lines (`\n\n`). Paragraphs are the natural unit of meaning in prose, so this keeps related sentences together as much as possible.
- **Safety net (`_safe_append`)**: regardless of strategy, if a single line or paragraph is itself longer than `max_chunk_size` (e.g. a very long docstring or table), it is force-split into fixed-size slices so no chunk ever exceeds the limit — this is required because the moulinette rejects any source longer than `max_context_length` (2000 characters by default) and a single over-long source invalidates the whole submission.

Character ranges are tracked precisely (`first_character_index` / `last_character_index`, inclusive) as chunks are built, so retrieved sources can always be mapped back to the exact original file location.

**JA** — コードと文章では意味のまとまり方が異なるため、2種類のチャンキング戦略を実装しています：

- **Pythonコード（`chunk_python_code`）**：行単位でチャンクを構築します。次の行を追加すると `max_chunk_size` を超えてしまう場合、その時点でチャンクを確定し新しいチャンクを開始します。これにより、識別子や文字列リテラルの途中で不自然に分断されることを避けています。
- **Markdown / テキスト（`chunk_markdown_text`）**：段落（空行 `\n\n` 区切り）単位でチャンクを構築します。文章においては段落が自然な意味の単位であるため、関連する文をできるだけまとめて保持できます。
- **セーフガード（`_safe_append`）**：戦略に関わらず、1行または1段落自体が `max_chunk_size` を超える場合（非常に長いdocstringや表など）は、固定サイズで強制的に分割し、どのチャンクも上限を超えないようにします。これは、moulinetteが `max_context_length`（デフォルト2000文字）を超えるソースを1件でも含んでいると提出全体を無効にするため、必須の対応です。

チャンク構築時には文字範囲（`first_character_index` / `last_character_index`、両端を含む）を正確に追跡しているため、検索結果は常に元ファイルの正確な位置に対応付けられます。

---

## Retrieval Method / 検索アルゴリズム

**EN** — Retrieval combines two complementary ranking signals:

1. **BM25 (lexical / keyword search)** — implemented with `rank_bm25.BM25Okapi`. Each chunk's text is tokenized (regex word extraction, lowercased, stopwords removed) at indexing time. This is strong when the question quotes an identifier or keyword verbatim (e.g. a function name).
2. **Semantic search (bonus)** — chunk texts are embedded with `all-MiniLM-L6-v2` (`sentence-transformers`) at indexing time; at query time the question is embedded the same way and ranked by cosine similarity (implemented as a dot product, since embeddings are normalized). This is strong when the question paraphrases an idea rather than quoting exact wording.

The two rankings are merged with **Reciprocal Rank Fusion (RRF)**: each method first retrieves a wider candidate pool (`max(k*5, 50)`), then every candidate's score is `Σ 1 / (60 + rank + 1)` summed across the rankings it appears in. This avoids having to reconcile two incomparable score scales (BM25 scores vs. cosine similarities) and instead uses *rank position*, which is directly comparable. The top-k documents by combined RRF score are returned as `MinimalSource` objects.

Repeated `(query, k)` pairs are served from an in-memory LRU cache (`functools.lru_cache`) to keep dataset-wide throughput well under the 90-second/200-questions requirement.

**JA** — 検索は2つの補完的なランキング手法を組み合わせています：

1. **BM25（キーワード検索）** — `rank_bm25.BM25Okapi` で実装。各チャンクのテキストはインデックス作成時にトークン化されます（正規表現による単語抽出、小文字化、ストップワード除去）。質問が識別子やキーワードをそのまま含む場合（例: 関数名）に強い手法です。
2. **セマンティック検索（ボーナス）** — インデックス作成時に `all-MiniLM-L6-v2`（`sentence-transformers`）でチャンクのテキストを埋め込みベクトル化。検索時には質問も同様に埋め込み、コサイン類似度（埋め込みが正規化済みのため内積で計算）でランキングします。質問が言い回しを変えている（パラフレーズしている）場合に強い手法です。

2つのランキングは **Reciprocal Rank Fusion（RRF）** で統合します：各手法はまず広めの候補プール（`max(k*5, 50)`件）を取得し、各候補のスコアを、出現した各ランキングにおける `Σ 1 / (60 + rank + 1)` の合計として計算します。スコアのスケールが全く異なる2手法（BM25のスコアとコサイン類似度）を直接比較する必要がなくなり、代わりに直接比較可能な「順位」を使えるのがこの手法の利点です。統合RRFスコアの上位k件が `MinimalSource` として返されます。

同じ `(query, k)` の組み合わせに対する検索は、メモリ上のLRUキャッシュ（`functools.lru_cache`）で処理されるため、「200問を90秒以内」という性能要件を余裕を持って満たせます。

---

## Performance Analysis / 性能評価

**EN** — Performance is measured with the official **moulinette** tool (`evaluate_student_search_results`), which reports Recall@1 / @3 / @5 / @10 and validates against the required thresholds:

| Dataset | Metric | Required | Measured |
|---|---|---|---|
| Docs | Recall@5 | ≥ 80% | *68.0% (Recall@1: 42.0%, Recall@3: 64.0%, Recall@10: 81.0%)* |
| Code | Recall@5 | ≥ 50% | *55.6% (Recall@1: 31.3%, Recall@3: 46.5%, Recall@10: 67.7%)* |
| Indexing time (full corpus) | wall clock | ≤ 5 min | *< 1 min* |
| Retrieval throughput (200 questions) | wall clock | ≤ 90 s | *< 30 s* |


Qualitatively, combining BM25 with semantic search (RRF) tends to raise recall on documentation questions the most, since docs are often phrased differently from the exact wording in the source text, while BM25 alone remains competitive on code questions that quote identifiers directly.

**JA** — 性能は公式の **moulinette** ツール（`evaluate_student_search_results`）で測定します。このツールはRecall@1 / @3 / @5 / @10を出力し、必要な閾値と照合します：

| データセット | 指標 | 必須基準 | 実測値 |
|---|---|---|---|
| ドキュメント | Recall@5 | 80%以上 | *68.0% (Recall@1: 42.0%, Recall@3: 64.0%, Recall@10: 81.0%)* |
| コード | Recall@5 | 50%以上 | *55.6% (Recall@1: 31.3%, Recall@3: 46.5%, Recall@10: 67.7%)* |
| インデックス作成時間（全コーパス） | 実時間 | 5分以内 | < 1 min |
| 検索スループット（200問） | 実時間 | 90秒以内 | *< 30 s* |


定性的には、BM25とセマンティック検索の統合（RRF）はドキュメント系の質問でrecallを最も押し上げる傾向があります。ドキュメントは原文と異なる言い回しで質問されることが多いためです。一方、識別子をそのまま含むコード系の質問ではBM25単体でも十分な性能を発揮します。

---

## Design Decisions / 設計上の判断

**EN**
- **RRF over weighted score blending**: chosen because BM25 scores and cosine similarities live on incomparable scales; RRF sidesteps that entirely by using rank position, with no hyperparameter tuning required beyond the standard `rrf_k=60`.
- **Two separate chunkers instead of one generic splitter**: code and prose have different natural boundaries (lines vs. paragraphs); using a single naive splitter (e.g. fixed-size windows everywhere) would frequently cut through the middle of a function or a sentence, hurting both grounding quality and recall.
- **Inclusive character indices**: `last_character_index` is inclusive (not exclusive) to match the exact convention used in the ground-truth datasets and by the moulinette's IoU-based overlap check.
- **Incremental indexing keyed on file `mtime`**: re-chunking and re-embedding the entire corpus on every run would be wasteful and risk exceeding the 5-minute indexing budget as the corpus grows; tracking `file_meta.json` lets only changed files be reprocessed.
- **`k` is threaded through explicitly rather than fixed per instance**: the `Generator` and API layers accept an optional `k` override on each call so that a single loaded model instance can serve different `k` values per request without reloading — important since model loading is the most expensive part of the pipeline.
- **Never importing the moulinette**: the CLI's own `evaluate` command is a simplified self-check, kept deliberately separate from the moulinette's implementation to avoid any accidental coupling to (or reliance on) the exact grading internals.

**JA**
- **加重スコア合成ではなくRRFを採用**：BM25のスコアとコサイン類似度は比較不可能なスケールに存在するため、RRFは「順位」だけを使うことでこれを完全に回避できます。標準的な `rrf_k=60` 以外に追加のハイパーパラメータ調整も不要です。
- **単一の汎用分割器ではなく2種類の専用チャンカーを用意**：コードと文章では自然な区切りが異なる（行 vs 段落）ため、単純な固定長分割ではしばしば関数や文の途中で分断されてしまい、グラウンディングの質・recall双方に悪影響を与えます。
- **文字インデックスをinclusiveにする**：`last_character_index` を（exclusiveではなく）inclusiveにすることで、正解データセットおよびmoulinetteのIoUベースの重なり判定と正確に整合させています。
- **ファイルの`mtime`をキーにした差分インデックス**：実行のたびにコーパス全体を再チャンク・再埋め込みするのは無駄が多く、コーパスが大きくなるにつれ5分というインデックス作成の時間制約を超えるリスクがあります。`file_meta.json` で変更を追跡することで、変更のあったファイルのみを再処理します。
- **`k` をインスタンス固定値ではなく明示的に伝播させる**：`Generator` とAPI層は呼び出しごとに任意の `k` オーバーライドを受け付けます。これにより、モデルロード（パイプライン中最もコストの高い処理）をやり直すことなく、1つのロード済みモデルインスタンスがリクエストごとに異なる `k` を処理できます。
- **moulinetteを一切importしない**：CLI自身の `evaluate` コマンドはあくまで簡易的な自己チェックであり、採点処理の内部実装に偶発的に依存・結合しないよう、意図的にmoulinetteの実装とは切り離しています。

---

## Challenges Faced / 直面した課題と対応

**EN**
- **Keeping chunks under the 2000-character hard limit** while still respecting "natural" boundaries (lines/paragraphs) required a two-level approach: build chunks at the natural granularity first, then force-split any oversized remainder (`_safe_append`) as a safety net, rather than always splitting at a fixed size.
- **Reconciling BM25 and semantic search score scales**: an initial attempt to linearly combine raw scores produced results dominated by whichever method happened to have a larger numeric range; switching to RRF (rank-based, scale-free) resolved this.
- **Propagating `k` consistently across the pipeline**: the `Generator` originally only respected the `k` set at construction time, so a caller requesting a different `k` per request (e.g. through the API) would silently get the wrong number of context sources. This was fixed by adding an optional `k` parameter to `_generate_prompt` / `generate_answers` that overrides the instance default without needing to reload the model.
- **Corpus folder name coupling**: an early version of the indexer hardcoded the shipped vLLM version folder name; this was changed to walk `data/raw/` recursively so the system keeps working even if the folder is renamed or a different corpus is provided.

**JA**
- **2000文字という上限を守りつつ「自然な」区切り（行・段落）も尊重すること**：まず自然な粒度でチャンクを構築し、上限を超えてしまう残り部分だけを `_safe_append` で強制分割するという2段階のアプローチを採用しました。常に固定サイズで分割するのではなく、この方式によって両方の要件を満たしています。
- **BM25とセマンティック検索のスコアスケールの違いの解消**：当初、生スコアを単純な線形結合で統合したところ、たまたま数値レンジが大きい方の手法の結果に偏ってしまいました。RRF（順位ベースでスケールに依存しない手法）に切り替えることでこの問題を解消しました。
- **パイプライン全体での`k`の一貫した伝播**：`Generator` は当初、インスタンス生成時に設定された `k` しか参照しておらず、リクエストごとに異なる `k` を指定したい呼び出し元（例: API経由）が、気づかないまま誤った件数のコンテキストを受け取ってしまう状態でした。`_generate_prompt` / `generate_answers` にオプションの `k` 引数を追加し、モデルを再ロードすることなくインスタンスのデフォルト値を上書きできるようにして解決しました。
- **コーパスのフォルダ名への依存**：Indexerの初期バージョンでは、提供されたvLLMのバージョンフォルダ名がハードコードされていました。`data/raw/` を再帰的に走査する方式に変更し、フォルダ名が変更されたり別のコーパスが提供されたりしても動作し続けるようにしました。

---

## Example Usage / 使用例

**EN**

```bash
# Full pipeline on the public docs dataset
uv run python -m src index --max_chunk_size 2000

uv run python -m src search_dataset \
    --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
    --k 10 \
    --save_directory data/output/search_results/UnansweredQuestions

./moulinette evaluate_student_search_results \
    data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    data/datasets/AnsweredQuestions/dataset_docs_public.json \
    --k 10 --max_context_length 2000

uv run python -m src answer_dataset \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --save_directory data/output/search_results_and_answer/UnansweredQuestions

# Quick single-question smoke test
uv run python -m src answer "How do I configure the OpenAI-compatible server?" --k 5 --debug

# Bonus: local HTTP API
uv run uvicorn src.api:app --reload
curl -X POST http://localhost:8000/answer \
    -H "Content-Type: application/json" \
    -d '{"query": "How do I configure the OpenAI-compatible server?", "k": 5}'
```

**JA**

```bash
# 公開ドキュメントデータセットでのフルパイプライン実行
uv run python -m src index --max_chunk_size 2000

uv run python -m src search_dataset \
    --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
    --k 10 \
    --save_directory data/output/search_results/UnansweredQuestions

./moulinette evaluate_student_search_results \
    data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    data/datasets/AnsweredQuestions/dataset_docs_public.json \
    --k 10 --max_context_length 2000

uv run python -m src answer_dataset \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --save_directory data/output/search_results_and_answer/UnansweredQuestions

# 単一質問での簡易動作確認
uv run python -m src answer "How do I configure the OpenAI-compatible server?" --k 5 --debug

# ボーナス：ローカルHTTP API
uv run uvicorn src.api:app --reload
curl -X POST http://localhost:8000/answer \
    -H "Content-Type: application/json" \
    -d '{"query": "How do I configure the OpenAI-compatible server?", "k": 5}'
```
