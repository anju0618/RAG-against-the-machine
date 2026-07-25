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
        chnks = []
        lines = text.split('\n')
        current_chunk = ""
        start_idx = 0
        current_idx = 0

        for line in lines:
            line_len = len(line) + 1 # +1 for new line
            if len(current_chunk) + line_len > self.max_chunk_size and current_chunk:
                chunks.append({
                    "file_path": str(fule_path),
                    "first_character_index": start_idx,
                    "last_character_index": start_idx + len(current_chunk) - 1,
                    "text": current_chunk  
                })
                start_idx += len(current_chunk)
                current_chunk = line + "\n"
                current_idx += line_len
            else:
                current_chunk += line + "\n"
                current_idx += line_len

            if current_chunk:
                chunks.append({
                    "file_path": str(file_path),
                    "first_character_index": start_idx,
                    "last_character_index": start_idx + len(current_chunk) - 1,
                    "text": current_chunk
                })
            return self.chunks

    def chunk_markdown_text(self, text: str, file_path: str) -> List[Dict[str, Any]]:
        """
        Markdown/テキスト用のチャンキング戦略。
        段落（連続する改行）を意識して分割し、文脈が途切れないようにします。
        """
        chunks = []
        paragraphs = text.split('\n\n')
        current_chunk = ""
        start_idx = 0

        for para in paragraphs:
            para_len = len(para) + 2 # +2 for \n\n
            