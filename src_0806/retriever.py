"""
retriever.py
検索方式は2種類を組み合わせている：
  - BM25（キーワードベースのスコアリング）
  - セマンティック検索（SentenceTransformerによるコサイン類似度）

  RRF (Reciprocal Rank Fusion) で統合、
（ボーナス1: セマンティック埋め込み／ボーナス2: ハイブリッド検索）
lru_cache
（ボーナス4: キャッシング）
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
QUERY_INSTRUCTION_PREFIX = ""


class Retriever:
    """
    インスタンス生成時に一度だけインデックスをメモリ上にロードし、
    以降の検索呼び出しではディスクアクセスを行わない
    """

    def __init__(self) -> None:
        self.processed_dir = Path("data/processed")
        self.chunks: List[Dict[str, Any]] = []
        self.bm25: BM25Okapi | None = None
        self.embeddings: np.ndarray | None = None
        self.embed_model: SentenceTransformer | None = None

        self.load_index()
        self.load_semantic_index()

    def load_index(self) -> None:
        """
        chunks.json を読み込みBM25を初期化
        """
        chunks_path = self.processed_dir / "chunks.json"
        if not chunks_path.exists():
            raise FileNotFoundError("Index not found. Run 'index' first.")

        with open(chunks_path, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
            self.chunks = cast(List[Dict[str, Any]], loaded_data)

        tokenized_corpus = [
            tokenize_text(build_search_text(chunk["file_path"], chunk["text"]))
            for chunk in self.chunks
        ]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"Loaded {len(self.chunks)} chunks into Retriever.")

    def load_semantic_index(self) -> None:

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
        単一クエリの検索
        """
        return self._cached_search(query, k)

    @lru_cache(maxsize=1024)
    def _cached_search(self, query: str, k: int) -> List[MinimalSource]:
        """
        検索メソッド

        lru_cache により、同一の (query, k) の組み合わせに対する
        検索結果を最大1024件までメモリ上にキャッシュ

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
        Args:
            query: 検索クエリ文字列
            k: 取得したい上位件数

        Returns:
            (チャンクのインデックス, 類似度スコア) のタプルのリスト
            埋め込みインデックスが未ロードの場合は空リストを返す
        """
        if self.embeddings is None or self.embed_model is None:
            return []

        query_embedding = self.embed_model.encode(
            QUERY_INSTRUCTION_PREFIX + query, normalize_embeddings=True
        )
        # 全チャンクとの内積（＝コサイン類似度）を計算
        scores = np.dot(self.embeddings, query_embedding)
        top_indices = np.argsort(scores)[::-1][:k]

        return [(int(idx), float(scores[idx])) for idx in top_indices]

    def hybrid_search(self, query: str, k: int = 5) -> List[MinimalSource]:
        """
        ハイブリッド検索
        BM25によるキーワード検索とセマンティック検索の結果を
        RRF (Reciprocal Rank Fusion) を用いて統合し、上位 k 件を返す。
        計算式: score(doc) = Σ 1 / (rrf_k + rank + 1)

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

        fetch_k = max(k * 10, 150)

        # BM25
        tokenized_query = tokenize_text(query)
        bm25_indices = self.bm25.get_top_n(
            tokenized_query,
            range(len(self.chunks)),
            n=fetch_k
        )

        # セマンティック
        semantic_results = self.semantic_search(query, k=fetch_k)
        semantic_indices = [idx for idx, _ in semantic_results]

        # スコア統合
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

        sorted_indices = sorted(
            rrf_scores.keys(),
            key=lambda x: rrf_scores[x],
            reverse=True
        )[:k]

        # インデックスから実際のMinimalSourceへ変換
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
        それぞれの質問に対してバッチで検索を実行
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
            sources = self.search(q_text, k)

            results.append(
                MinimalSearchResults(
                    question_id=q_id,
                    question=q_text,
                    retrieved_sources=sources
                )
            )

        return StudentSearchResults(search_results=results, k=k)
