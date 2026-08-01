*This project has been created as part of the 42 curriculum by amakino.*

# RAG-against-the-machine

## Description
このプロジェクトは、配布された vllm-0.10.1 という大規模なソフトウェアのソースコード一式を対象に、コードの仕様や内容について正確に答えられるRAGシステム(Retrieval-Augmented Generation)を自作することです。

AI（Qwen3-0.6Bという小型モデルを使います）に直接「このコードはどう動くの？」と聞いても知らないので、まずは既に書いたプログラムで関連するコードの断片を検索（Retrieve）し、それをAIに読ませて回答（Generate）させるパイプラインを作ります。
## Instruction

## Resources

### RAGについて
[RAG (検索拡張生成) とは何ですか?](https://aws.amazon.com/jp/what-is/retrieval-augmented-generation/)

[【生成AI入門】「RAG」をできるだけわかりやすく解説してみる](https://qiita.com/Junpei_Takagi/items/f82d31323f00ad895579)

[RAGの実装戦略まとめ](https://qiita.com/jw-automation/items/045917be7b558509fdf2)

### UUID
[Python3でUUIDを生成する](https://qiita.com/shimajiri/items/315d458d3796ed4a60da)

### Transformerライブラリ
[HuggingFaceのTransformerライブラリを使ってみよう](https://qiita.com/ski2_1116/items/f74e7b97008663d0702d)

[LLMでよく見る関数についての解説](https://qiita.com/ilovebooks0618/items/0292ec6ad09a6340f64b)

### Pytorch
[【2024年最新版】Python🐍PyTorch初心者ガイド](https://qiita.com/wooooo/items/f7d439e166ff664ad47c)

[PytorchによるLLMの高速化](https://zenn.dev/umeko/articles/fe961fda3148d1)

## memo

### **RAGとは**:
一般的なLLMに質問するとき、通常AIは過去に学習した「記憶」だけを頼りに答えます。しかし、それでは学習した時点以降の最新情報や、学習データに含まれないプライベートな情報（個人のソースコードなど）には答えられず、ハルシネーションをしてしまう問題があります。
RAG（Retrieval-Augmented Generation：検索拡張生成）は、これを解決するためにAIに「外部資料の検索」を許可する仕組みです。
具体的には以下の4つのステップを踏みます
1. **Indexing（インデックス作成）:**
    検索しやすいようにデータを整理・分割する
2. **Retrieving（検索）:**
    質問に関連する文章をデータの中から探し出す
3. **Augmenting（拡張）:**
    探し出した文章をAIに渡す準備をする。
4. **Generating（生成）:**
    AIがその文章を読んで、最終的な回答を作る。

### **BM250kapi**:
BM25Okapi とは、検索クエリと文書の関連性をスコア化するアルゴリズム「Okapi BM25」をPythonで簡単に使えるようにした、rank_bm25 ライブラリの代表的なクラス

Okapi BM25 の仕組み
* 単語の出現頻度 (TF) と 逆文書頻度 (IDF) を組み合わせて、文書の関連度を計算します。
* 文書の長さによる偏りを調整するパラメータ（k₁ や b）を備えています
* キーワード（語句）がどれだけ一致しているかを調べる、いわゆる「Lexical Search（語彙ベース検索）」で広く使われています。

### /data/raw/vllm-0.10.1　の中身
* 検索、参照用のコーパス
    展開して data/raw/vllm-0.10.1 として配置し、何千ものPythonファイルやMarkdownドキュメントをインデックス化します。
* AIの外部知識源
    LLMが直接記憶していないコードベースや公式ドキュメントの詳細情報を、この中から正確に検索（Retrieval）して回答の根拠（コンテキスト）として与えるために使用します。
### datasets_public.zip の中身
* **UnansweredQuestions**

    質問ID（question_id）と質問文（question）のみが含まれた、入力用の質問リスト（JSON形式）

    例: dataset_docs_public.json（ドキュメントに関する質問群）など。

    自分のRAGシステムにこのファイルを読み込ませて、検索結果や回答を出力させるためのテストデータとして使用します。

* **AnsweredQuestions**

    同じ質問に対して、あらかじめ人間や正解システムが特定した正しいソース位置（sources）や模範解答（answer）がセットになったデータが入っています。

    評価ツール（moulinette）が、出力した検索結果のJSONとこの正解ファイルを突き合わせ、Recall@k（検索精度）を自動算出するために使用します。
---
Docs（ドキュメント関連）: vLLMの公式ドキュメントやMarkdownファイルの内容を問う質問（目標：Recall@5で 80%以上）。

Code（ソースコード関連）: vLLMのPythonコードの仕組みや実装を問う質問（目標：Recall@5で 50%以上）。

### /srcの中身
#### models.py
すべてのコンポーネント（検索や生成）の間でやり取りされるデータの形を、Pydanticを使って定義しているファイル

* **`MinimalSource:`** 検索された根拠（ソース）の最小単位です。ファイルのパス（file_path）と、何文字目から何文字目までか（first_character_index, last_character_index）を保持します。これが評価システム（Moulinette）との共通言語になります。
* **`UnansweredQuestion` / `AnsweredQuestion`:** 質問データの構造です。前者は「質問IDと質問文」、後者はそこに「正解のソースと模範解答」が加わった形です。
* **`MinimalSearchResults` / `MinimalAnswer`:** 1つの質問に対して「どのソースが見つかったか（検索結果）」、およびそれに「LLMが生成した回答（answer）」を組み合わせたモデルです。
* **`StudentSearchResults` / `StudentSearchResultsAndAnswer`:** 100問などのデータセット全体を処理したときに出力される最終的なJSONの全体構造を規定します。
---
#### indexer.py
大量のvLLMのソースコードやドキュメントを読み込み、高速に検索できる「**インデックス**」を構築してディスクに保存するクラス
* 2つのチャンキング戦略:

    `chunk_python_code`: Pythonコード（.py）は行（改行）単位で意味のまとまりを意識して分割します

    `chunk_markdown_text`: Markdownやドキュメントは段落（空行 \n\n）単位で自然に分割します
* 安全装置（_safe_append）: どんなに長いファイルや段落であっても、1チャンクの最大文字数（max_chunk_size、デフォルト2000文字）を絶対に超えないように強制分割し、位置情報（開始・終了インデックス）を正確に記録します。
* BM25インデックスの構築: ストップワード（`a`, `the`, `is` などの検索ノイズになる単語）を除去した上で、`BM25Okapi` を使ってキーワード検索用の数理モデルを構築し、data/processed/chunks.json としてファイルに保存
---
#### retriever.py
作成されたインデックス（`chunks.json`）をメモリ上に読み込み、ユーザーの質問に対して最も関連性の高いソースの場所を探し出すクラスです。
* load_index: 保存されたインデックスを読み込み、BM25検索エンジンを再初期化します

    1. `search（単一クエリ検索）`:ユーザーの質問文からストップワードを除去してトークン化します.
    2. `BM25Okapi` の機能を使って、コーパス全体から関連度の高い上位 $k$ 件のチャンクのインデックス（top_k_indices）を取得します
    3. 該当するチャンクの `file_path` と文字インデックス範囲を `MinimalSource` オブジェクトに詰めて返します。
* search_dataset（データセット一括検索）: 複数問の質問が含まれるJSONファイルを読み込み、1問ずつ search を実行して一括の検索結果（`StudentSearchResults`）を組み立てます。
---
#### generator.py
検索されたソースの場所（ファイルと文字範囲）をもとに実ファイルを読み込み、ローカルLLM（Qwen）に渡して自然言語の回答を生成するクラスです。
* `__init__`: 指定された軽量LLM（`Qwen/Qwen2.5-0.5B-Instruct` など）とトークナイザーを`Hugging Face`からロードします。
* `_load_chunk`: 検索結果が指し示すファイルのパスと文字インデックスを使って、実ファイルから該当部分のテキストをピンポイントで切り出します
* `_generate_prompt`:

    * `[SYSTEM]` プロンプトで「コンテキスト（検索されたテキスト）にのみ基づいて答えよ」「外部知識や嘘は使ってはならない」という厳格なルールを指示します。
    * 切り出した複数のソーステキストとユーザーの質問を結合し、LLMへの入力プロンプトを組み立てます。
* `generate_answers`: バッチ単位でLLMの generate を呼び出し、コンテキストに基づいた回答文字列を生成・デコードします。進捗がわかるように `tqdm` も組み込まれています。
---
#### \_\_main\_\_.py
Python Fire を用いて、ターミナルからのコマンド入力を受け付け、上記のクラスたち（Indexer, Retriever, Generator）を適切な順序で動かすmainファイルです
* `index`:
    * Indexer を呼び出してコーパスの分割とBM25インデックスの構築を行います。
* `search`: Retriever を呼び出して単一の質問に対する上位 $k$ 件のソース位置を表示（またはデバッグ表示）します
* `search_dataset`: 質問データセット全体を検索し、指定ディレクトリにJSONとして保存します。
* `answer / answer_dataset`: 検索からLLMによる回答生成までを繋げ、最終的な `StudentSearchResultsAndAnswer` 形式のJSONファイルを出力します。デバッグモード時には、プロンプトがどのように組み立てられているかの過程を可視化します。
