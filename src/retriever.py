import json
from pathlib import Path
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from src.models import MinimalSource, MinimalSearchResults, StudentSearchResults

class Retriever:
    """
    作成されたインデックスを使用して、質問に対する関連ソースを検索するクラス
    """
    def __init__(self):
        self.processed_dir = Path("data/processed")
        self.chunks: List[Dict[str, Any]] = []
        self.bm25: BM25Okapi = None
        
        # 初期化時にインデックスをロードする
        self.load_index()

    def load_index(self) -> None:
        """
        Indexerが保存した chunks.json を読み込み、BM25を再構築します。
        """
        chunks_path = self.processed_dir / "chunks.json"
        if not chunks_path.exists():
            raise FileNotFoundError("Index not found. Please run 'index' command first.")

        with open(chunks_path, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
            
        # インデクサーと同じトークナイズ手法（空白分割）でBM25を構築
        tokenized_corpus = [chunk["text"].split(" ") for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"Loaded {len(self.chunks)} chunks into Retriever.")

    def search(self, query: str, k: int = 5) -> List[MinimalSource]:
        """
        単一のクエリに対して、上位k件の関連ソースを返します。
        """
        # クエリをトークナイズ
        tokenized_query = query.split(" ")
        
        # BM25でスコアを計算し、上位k件のインデックス（0, 1, 2...）を取得
        # ※コーパス自体ではなくインデックス番号のリストを渡すことで、元のchunksにアクセスしやすくする
        top_k_indices = self.bm25.get_top_n(tokenized_query, range(len(self.chunks)), n=k)
        
        sources = []
        for idx in top_k_indices:
            chunk = self.chunks[idx]
            # 抽出したデータを、必須要件であるPydanticモデルに流し込む
            sources.append(
                MinimalSource(
                    file_path=chunk["file_path"],
                    first_character_index=chunk["first_character_index"],
                    last_character_index=chunk["last_character_index"]
                )
            )
        return sources

    def search_dataset(self, dataset_path: str, k: int = 5) -> StudentSearchResults:
        """
        データセット（JSON）を読み込み、全質問に対して検索を実行します。
        最終的にMoulinetteに提出するフォーマット（StudentSearchResults）を返します。
        """
        dataset_file = Path(dataset_path)
        if not dataset_file.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        with open(dataset_file, 'r', encoding='utf-8') as f:
            dataset_data = json.load(f)
            
        results = []
        questions = dataset_data.get("rag_questions", [])
        
        for q in questions:
            q_id = q["question_id"]
            q_text = q["question"]
            
            # 各質問に対して検索を実行
            sources = self.search(q_text, k)
            
            results.append(
                MinimalSearchResults(
                    question_id=q_id,
                    question=q_text,
                    retrieved_sources=sources
                )
            )
            
        # 全結果をStudentSearchResultsモデルに格納して返す
        return StudentSearchResults(search_results=results, k=k)
