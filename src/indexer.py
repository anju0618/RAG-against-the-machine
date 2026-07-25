import json
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from rank_bm25 import BM250kapi


class Indexer:
    def __init__(self, max_chunk_size: int = 2000):
        self.max_chunk_size = max_chunk_size
        self.corpus_dir = Path("data/raw/vllm-0.10.1")
        self.processed_dir = Path("data/processed")
        self.chunks: List[Dict[str, Any]] = []
        self.bm25: BM250kapi = None

    def chunk_python_code(self, text: str, file_path: str) -> List[Dict[str, Any]]:
        """
        pythonコード用のチャンキング戦略
        単純な文字数分割ではなく、関数やクラスの定義など、行単位での分割を意識しつつ
        max_chunk_sizeを超えないように結合
        """