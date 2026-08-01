import json
import re
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
from src.indexer import STOPWORDS


class Retriever:
    """
    Indexerが作ったindexを使って、質問に一番関連するソーステキストを探し出します。
    キャッシュ機能により、繰り返しクエリを高速化します。
    """

    def __init__(self) -> None:
        """
        初期化。起動と同時にインデックスをメモリ上にロードします。
        """
        self.processed_dir = Path("data/processed")
        self.chunks: List[Dict[str, Any]] = []
        self.bm25: BM25Okapi | None = None
        self.embeddings: np.ndarray | None = None
        self.embed_model: SentenceTransformer | None = None

        self.load_index()
        self.load_semantic_index()

    def load_index(self) -> None:
        """
        chunks.json を読み込み、ストップワードを除去した状態でBM25エンジンを初期化します。
        """
        chunks_path = self.processed_dir / "chunks.json"
        if not chunks_path.exists():
            raise FileNotFoundError("Index not found. Run 'index' first.")

        with open(chunks_path, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
            self.chunks = cast(List[Dict[str, Any]], loaded_data)

        tokenized_corpus = [
            [
                w for w in re.findall(r'\w+', chunk["text"].lower())
                if w not in STOPWORDS
            ]
            for chunk in self.chunks
        ]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"Loaded {len(self.chunks)} chunks into Retriever.")

    def load_semantic_index(self) -> None:
        """
        保存されている embeddings.npy と
        SentenceTransformer モデルをロードします。
        """
        embeddings_path = self.processed_dir / "embeddings.npy"
        if embeddings_path.exists():
            try:
                self.embeddings = np.load(embeddings_path)
                self.embed_model = SentenceTransformer(
                    "all-MiniLM-L6-v2", device="cpu"
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
        キャッシュを活用してハイブリッド検索を実行します。
        """
        return self._cached_search(query, k)

    @lru_cache(maxsize=1024)
    def _cached_search(self, query: str, k: int) -> List[MinimalSource]:
        """
        内部キャッシュ付き検索メソッド（LRUキャッシュにより最大1024件のクエリ結果を保持）
        """
        return self.hybrid_search(query, k)

    def semantic_search(
        self, query: str, k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        【セマンティック検索（ベクトル検索）】
        クエリの埋め込みベクトルと、全チャンクの埋め込みベクトルの
        コサイン類似度を計算し、上位 k 件のインデックスを返します。
        """
        if self.embeddings is None or self.embed_model is None:
            return []

        query_embedding = self.embed_model.encode(
            query, normalize_embeddings=True
        )
        scores = np.dot(self.embeddings, query_embedding)
        top_indices = np.argsort(scores)[::-1][:k]

        return [(int(idx), float(scores[idx])) for idx in top_indices]

    def hybrid_search(self, query: str, k: int = 5) -> List[MinimalSource]:
        """
        【ハイブリッド検索】
        BM25によるキーワード検索とセマンティック検索の結果を
        RRF (Reciprocal Rank Fusion) を用いて統合し、上位 k 件を返します。
        """
        if self.bm25 is None:
            raise RuntimeError("BM25 index is not loaded.")

        fetch_k = max(k * 5, 50)

        # 1. BM25によるランキング取得
        tokenized_query = [
            w for w in re.findall(r'\w+', query.lower())
            if w not in STOPWORDS
        ]
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

        sorted_indices = sorted(
            rrf_scores.keys(),
            key=lambda x: rrf_scores[x],
            reverse=True
        )[:k]

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
        複数の質問が記載されたJSONファイルを読み込み、それぞれの質問に対して検索を実行

        Args:
            dataset_path: 質問データセットのJSONファイルパス (str)
            k: 各質問ごとに取得する上位件数 (int)

        Returns:
            StudentSearchResults: moulinetteに送るやつ
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
