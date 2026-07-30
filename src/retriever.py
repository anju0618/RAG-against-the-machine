import json
import re
from pathlib import Path
from typing import List, Dict, Any, cast
from rank_bm25 import BM25Okapi  # type: ignore
from src.models import (
    MinimalSource,
    MinimalSearchResults,
    StudentSearchResults,
)
from src.indexer import STOPWORDS


class Retriever:
    """
    【役割】検索ロボット
    Indexerが作った辞書を使って、質問に一番関連するソーステキストを探し出します。
    """

    def __init__(self) -> None:
        """
        リトリーバーの初期化。起動と同時にインデックスをメモリ上にロードします。
        """
        self.processed_dir = Path("data/processed")
        self.chunks: List[Dict[str, Any]] = []
        self.bm25: BM25Okapi | None = None

        self.load_index()

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

    def search(self, query: str, k: int = 5) -> List[MinimalSource]:
        """
        【単一クエリの検索】
        1つの質問文字列に対して、関連度の高い上位 k 件のソース位置を返します。

        Args:
            query: 検索クエリ文字列 (例: "How to configure the OpenAI server?")
            k: 取得する上位の件数 (int, デフォルト: 5)

        Returns:
            MinimalSourceオブジェクトのリスト (最大 k 件)
        """
        if self.bm25 is None:
            raise RuntimeError("BM25 index is not loaded.")

        # クエリ文からストップワードを除去し、小文字化してトークン化
        tokenized_query = [
            w for w in re.findall(r'\w+', query.lower())
            if w not in STOPWORDS
        ]

        top_k_indices = self.bm25.get_top_n(
            tokenized_query,
            range(len(self.chunks)),
            n=k
        )

        sources = []
        for idx in top_k_indices:
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
        【データセット一括検索】
        複数の質問が記載されたJSONファイル（例: 100問）を読み込み、それぞれの質問に対して検索を実行します。

        Args:
            dataset_path: 質問データセットのJSONファイルパス (str)
            k: 各質問ごとに取得する上位件数 (int)

        Returns:
            StudentSearchResults: 採点形式に準拠した検索結果オブジェクト
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
