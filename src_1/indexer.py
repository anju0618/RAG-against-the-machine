import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
from rank_bm25 import BM25Okapi


class Indexer:
    def __init__(self, max_chunk_size: int = 2000):
        self.max_chunk_size = max_chunk_size
        self.corpus_dir = Path("data/raw/vllm-0.10.1")
        self.processed_dir = Path("data/processed")
        self.chunks: List[Dict[str, Any]] = []
        self.bm25: BM25Okapi = None

    def _safe_append(self, chunks: list, file_path: str,
                     start_idx: int, text: str):
        """
        method to append safely
        """
        idx = 0
        while idx < len(text):
            chunk_slice = text[idx:idx + self.max_chunk_size]
            chunks.append({
                "file_path": file_path,
                "first_character_index": start_idx + idx,
                "last_character_index": start_idx + idx + len(chunk_slice) - 1,
                "text": chunk_slice
            })
            idx += self.max_chunk_size

    def chunk_python_code(self, text: str, file_path: str) -> List[Dict[str, Any]]:
            chunks = []
            lines = text.split('\n')
            current_chunk = ""
            start_idx = 0
    
            for i, line in enumerate(lines):
                line_text = line + '\n' if i < len(lines) - 1 else line
                line_len = len(line_text)
                
                if len(current_chunk) + line_len > self.max_chunk_size and current_chunk:
                    self._safe_append(chunks, str(file_path), start_idx, current_chunk)
                    start_idx += len(current_chunk)
                    current_chunk = ""
                    
                current_chunk += line_text
    
            if current_chunk:
                self._safe_append(chunks, str(file_path), start_idx, current_chunk)
                
            return chunks

    def chunk_markdown_text(self, text: str, file_path: str) -> List[Dict[str, Any]]:
            chunks = []
            paragraphs = text.split('\n\n')
            current_chunk = ""
            start_idx = 0
    
            for i, para in enumerate(paragraphs):
                para_text = para + '\n\n' if i < len(paragraphs) - 1 else para
                para_len = len(para_text)
                
                if len(current_chunk) + para_len > self.max_chunk_size and current_chunk:
                    self._safe_append(chunks, str(file_path), start_idx, current_chunk)
                    start_idx += len(current_chunk)
                    current_chunk = ""
                    
                current_chunk += para_text
    
            if current_chunk:
                 self._safe_append(chunks, str(file_path), start_idx, current_chunk)
                 
            return chunks

    def index_corpus(self) -> None:
            if not self.corpus_dir.exists():
                raise FileNotFoundError(f"Corpus directory not found: {self.corpus_dir}")
    
            self.processed_dir.mkdir(parents=True, exist_ok=True)
            target_extensions = {'.py', '.md', '.txt', '.rst'}
            
            filepaths = []
            for root, _, files in os.walk(self.corpus_dir):
                for file in files:
                    if Path(file).suffix in target_extensions:
                        filepaths.append(Path(root) / file)
    
            print(f"Found {len(filepaths)} files to index.")
            
            for file_path in tqdm(filepaths, desc="Chunking"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                    
                    if file_path.suffix == '.py':
                        file_chunks = self.chunk_python_code(text, file_path)
                    else:
                        file_chunks = self.chunk_markdown_text(text, file_path)
                    
                    self.chunks.extend(file_chunks)
                except Exception as e:
                    continue
    
            print(f"Created {len(self.chunks)} chunks. Building BM25 index...")
            
            # 修正箇所: split(" ") を re.findall(r'\w+', ...) に変更して記号を除外＆小文字化
            tokenized_corpus = [re.findall(r'\w+', chunk["text"].lower()) for chunk in tqdm(self.chunks, desc="Tokenizing")]
            self.bm25 = BM25Okapi(tokenized_corpus)
    
            self.save_index()
            print(f"Ingestion complete! Indexed {len(self.chunks)} chunks under {self.processed_dir}/")
            