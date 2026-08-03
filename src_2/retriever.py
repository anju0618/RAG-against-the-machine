"""
retriever.py
============

Indexerが作成したインデックス（BM25用のチャンク + セマンティック埋め込み）を
使って、質問に最も関連するソーステキストの場所（file_path と文字範囲）を
検索するモジュール。

検索方式は2種類を組み合わせている：
  - BM25（キーワードベースのスコアリング）
  - セマンティック検索（SentenceTransformerによるコサイン類似度）

この2つのランキングを RRF (Reciprocal Rank Fusion) で統合することで、
「識別子をそのまま含む質問」にも「意味は同じだが単語が違う質問」にも
ある程度強いハイブリッド検索を実現している
（ボーナス1: セマンティック埋め込み／ボーナス2: ハイブリッド検索）。

また、同じクエリ・k の組み合わせに対しては lru_cache で結果を
メモリ上にキャッシュし、同一クエリが繰り返された場合の応答を高速化する
（ボーナス4: キャッシング の一部）。

--- 精度改善について ---
BM25のトークン化には indexer.py の tokenize_text() / build_search_text() を
そのまま共有利用している。これにより、
  - Indexer側でインデックスを作った際のトークン化方法
  - Retriever側で検索クエリをトークン化する方法
が完全に一致し、「同じ単語のはずなのにインデックス側とクエリ側で
トークン化の結果がずれてマッチしない」という事態を防いでいる。
tokenize_text() 自体も snake_case / camelCase の識別子分割に対応した
ものになっているため、単純な \\w+ 抽出だけの場合よりも
質問とチャンクの語彙が一致しやすくなっている。
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, cast
from functools import lru_cache
from rank_bm25 import BM25Okapi
import numpy as np
from sentence_transformers import SentenceTransformer
from src.models import (
    MinimalSource,
    MinimalSearchResults,
    StudentSearchResults,
)
from src.indexer import EMBEDDING_MODEL_NAME, build_search_text, tokenize_text

# BGE系の埋め込みモデルは、検索（asymmetric retrieval）タスクにおいて
# 「クエリ側にのみ」この指示文を付与すると精度が上がることが知られている
# （パッセージ/チャンク側には付与しない）。indexer.py 側でチャンクの
# 埋め込みを計算する際にはこの指示文を使っていないことに注意。
QUERY_INSTRUCTION_PREFIX = (
    "Represent this sentence for searching relevant passages: "
)


class Retriever:
    """
    Indexerが作った index (chunks.json / embeddings.npy) を使って、
    質問に一番関連するソーステキストを探し出すクラス。

    インスタンス生成時に一度だけインデックスをメモリ上にロードし、
    以降の検索呼び出しではディスクアクセスを行わない設計になっている。
    """

    def __init__(self) -> None:
        """
        初期化。インスタンス生成と同時にインデックス
        （BM25用チャンク + セマンティック埋め込み）をメモリ上にロードする。

        Raises:
            FileNotFoundError: chunks.json が見つからない場合
                （事前に `index` コマンドを実行しておく必要がある）
        """
        self.processed_dir = Path("data/processed")
        self.chunks: List[Dict[str, Any]] = []
        self.bm25: BM25Okapi | None = None
        self.embeddings: np.ndarray | None = None
        self.embed_model: SentenceTransformer | None = None

        # BM25用のインデックスをロード（必須）
        self.load_index()
        # セマンティック検索用の埋め込みをロード（あれば使う。
        # 存在しない/読み込み失敗時はBM25のみのフォールバック動作になる）
        self.load_semantic_index()

    def load_index(self) -> None:
        """
        chunks.json を読み込み、ストップワードを除去した状態で
        BM25エンジンを初期化する。

        Raises:
            FileNotFoundError: chunks.json が存在しない場合
        """
        chunks_path = self.processed_dir / "chunks.json"
        if not chunks_path.exists():
            raise FileNotFoundError("Index not found. Run 'index' first.")

        with open(chunks_path, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
            self.chunks = cast(List[Dict[str, Any]], loaded_data)

        # 各チャンクをファイル名ヒント付きの検索用テキストに変換してから
        # トークン化する。indexer.pyのload_index()と全く同じロジックを
        # 共有関数経由で呼んでいるため、インデックス作成時と検索時で
        # トークン化結果がずれない。
        tokenized_corpus = [
            tokenize_text(build_search_text(chunk["file_path"], chunk["text"]))
            for chunk in self.chunks
        ]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"Loaded {len(self.chunks)} chunks into Retriever.")

    def load_semantic_index(self) -> None:
        """
        保存されている embeddings.npy と
        SentenceTransformer モデル（EMBEDDING_MODEL_NAME、indexer.py と共有）
        をロードする。

        embeddings.npy が存在しない、またはロードに失敗した場合は
        警告を表示するのみで例外は投げない。この場合、hybrid_search()は
        自動的にBM25のみの結果にフォールバックする
        （semantic_search()が空リストを返すため）。
        """
        embeddings_path = self.processed_dir / "embeddings.npy"
        if embeddings_path.exists():
            try:
                self.embeddings = np.load(embeddings_path)
                self.embed_model = SentenceTransformer(
                    EMBEDDING_MODEL_NAME, device="cpu"
                )
                print(
                    "Loaded semantic embeddings shape: "
                    f"{self.embeddings.shape}"
                )
            except Exception as e:
                print(f"Warning: Could not load semantic embeddings: {e}")

    def search(self, query: str, k: int = 5) -> List[MinimalSource]:
        """
        【単一クエリの検索】
        キャッシュを活用してハイブリッド検索を実行する公開API。

        Args:
            query: 検索したい質問文字列
            k: 取得したい上位件数

        Returns:
            関連度が高い順に並んだ MinimalSource のリスト（最大k件）
        """
        return self._cached_search(query, k)

    @lru_cache(maxsize=1024)
    def _cached_search(self, query: str, k: int) -> List[MinimalSource]:
        """
        内部キャッシュ付き検索メソッド。

        lru_cache により、同一の (query, k) の組み合わせに対する
        検索結果を最大1024件までメモリ上にキャッシュする。
        評価用データセットには同じ質問が複数回投げられることもあるため、
        このキャッシュによって全体の検索スループットが向上する
        （仕様の「200問を90秒以内」という性能要件に寄与する）。

        Args:
            query: 検索クエリ文字列
            k: 取得したい上位件数

        Returns:
            hybrid_search() の結果をそのまま返す
        """
        return self.hybrid_search(query, k)

    def semantic_search(
        self, query: str, k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        【セマンティック検索（ベクトル検索）】
        クエリの埋め込みベクトルと、全チャンクの埋め込みベクトルの
        コサイン類似度を計算し、上位 k 件のインデックスとスコアを返す。

        埋め込みが正規化済み（normalize_embeddings=True で作成）の場合、
        コサイン類似度は単純な内積で計算できるため、np.dot を使っている。

        クエリ側にだけ QUERY_INSTRUCTION_PREFIX を付与している点に注意。
        BGE系の埋め込みモデルは、パッセージ（検索対象のチャンク）と
        クエリ（質問）の役割が非対称であることを前提に学習されており、
        クエリ側に検索用の指示文を付けることで実際に検索精度が上がる
        ことが確認されている（indexer.py側でチャンクを埋め込む際は
        この指示文を使っていない＝チャンク側は常にプレーンなテキスト）。

        Args:
            query: 検索クエリ文字列
            k: 取得したい上位件数

        Returns:
            (チャンクのインデックス, 類似度スコア) のタプルのリスト。
            埋め込みインデックスが未ロードの場合は空リストを返す
            （BM25のみのフォールバック動作になる）。
        """
        if self.embeddings is None or self.embed_model is None:
            return []

        query_embedding = self.embed_model.encode(
            QUERY_INSTRUCTION_PREFIX + query, normalize_embeddings=True
        )
        # 全チャンクとの内積（＝コサイン類似度）を一気に計算
        scores = np.dot(self.embeddings, query_embedding)
        # スコアの高い順にソートして上位k件のインデックスを取り出す
        top_indices = np.argsort(scores)[::-1][:k]

        return [(int(idx), float(scores[idx])) for idx in top_indices]

    def hybrid_search(self, query: str, k: int = 5) -> List[MinimalSource]:
        """
        【ハイブリッド検索】
        BM25によるキーワード検索とセマンティック検索の結果を
        RRF (Reciprocal Rank Fusion) を用いて統合し、上位 k 件を返す。

        RRFは「各ランキングにおける順位（rank）」だけを使ってスコアを
        合成する手法で、スコアのスケールが全く異なる2つの検索手法
        （BM25のスコアと埋め込みのコサイン類似度）を単純な重み付けなしで
        公平に統合できるのが利点。

        計算式: score(doc) = Σ 1 / (rrf_k + rank + 1)
        rrf_k は「上位付近の差を緩やかにする」ためのハイパーパラメータで、
        ここでは一般的によく使われる60を採用している。

        Args:
            query: 検索クエリ文字列
            k: 最終的に返す上位件数

        Returns:
            関連度が高い順に並んだ MinimalSource のリスト（最大k件）

        Raises:
            RuntimeError: BM25インデックスが未ロードの場合
                （通常は__init__でload_index()が呼ばれるため発生しない）
        """
        if self.bm25 is None:
            raise RuntimeError("BM25 index is not loaded.")

        # 最終的にk件に絞る前に、まず候補を広めに取得しておく
        # （fetch_k件）。こうすることで、BM25とセマンティックそれぞれの
        # 上位候補が十分に重なり合い、統合後の精度が安定しやすくなる。
        # 以前は max(k*5, 50) だったが、これだと候補プールが狭く、
        # BM25とセマンティックのどちらか一方にしか出てこない
        # 「惜しい」候補が最終結果から漏れやすかった。
        # 候補を広げても、この処理自体は検索時（クエリ処理時）にのみ
        # 発生し、インデックス作成時間（5分ルール）には影響しないため、
        # 精度向上のために少し余裕を持たせた値にしている。
        fetch_k = max(k * 8, 100)

        # 1. BM25によるランキング取得
        # クエリのトークン化も、インデックス作成時と全く同じ
        # tokenize_text() を使うことで、識別子分割などの語彙処理を一致させる
        tokenized_query = tokenize_text(query)
        bm25_indices = self.bm25.get_top_n(
            tokenized_query,
            range(len(self.chunks)),
            n=fetch_k
        )

        # 2. セマンティック検索によるランキング取得
        semantic_results = self.semantic_search(query, k=fetch_k)
        semantic_indices = [idx for idx, _ in semantic_results]

        # 3. RRF (Reciprocal Rank Fusion) によるスコア統合
        rrf_scores: Dict[int, float] = {}
        rrf_k = 60

        for rank, idx in enumerate(bm25_indices):
            rrf_scores[idx] = (
                rrf_scores.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)
            )

        for rank, idx in enumerate(semantic_indices):
            rrf_scores[idx] = (
                rrf_scores.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)
            )

        # 統合スコアの高い順に並べ替え、上位k件のインデックスだけを残す
        sorted_indices = sorted(
            rrf_scores.keys(),
            key=lambda x: rrf_scores[x],
            reverse=True
        )[:k]

        # インデックスから実際のソース情報（MinimalSource）へ変換する
        sources = []
        for idx in sorted_indices:
            chunk = self.chunks[idx]
            sources.append(
                MinimalSource(
                    file_path=chunk["file_path"],
                    first_character_index=chunk["first_character_index"],
                    last_character_index=chunk["last_character_index"]
                )
            )
        return sources

    def search_dataset(
        self, dataset_path: str, k: int = 5
    ) -> StudentSearchResults:
        """
        複数の質問が記載されたJSONファイル（データセット）を読み込み、
        それぞれの質問に対してバッチで検索を実行する。

        Args:
            dataset_path: 質問データセットのJSONファイルパス (str)。
                {"rag_questions": [{"question_id": ..., "question": ...}, ...]}
                という構造を想定。
            k: 各質問ごとに取得する上位件数 (int)

        Returns:
            StudentSearchResults: search_dataset コマンドの出力として
                そのままJSONに書き出せる、moulinette評価用の形式

        Raises:
            FileNotFoundError: dataset_path が存在しない場合
        """
        dataset_file = Path(dataset_path)
        if not dataset_file.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        with open(dataset_file, 'r', encoding='utf-8') as f:
            dataset_data = cast(Dict[str, Any], json.load(f))

        results = []
        questions = cast(
            List[Dict[str, Any]],
            dataset_data.get("rag_questions", [])
        )

        for q in questions:
            q_id = str(q["question_id"])
            q_text = str(q["question"])

            # 質問ごとに単一クエリ検索を呼び出す（内部でキャッシュも効く）
            sources = self.search(q_text, k)

            results.append(
                MinimalSearchResults(
                    question_id=q_id,
                    question=q_text,
                    retrieved_sources=sources
                )
            )

        return StudentSearchResults(search_results=results, k=k)
