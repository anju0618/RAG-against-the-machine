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
            if len(current_chunk) + para_len > self.max_chunk_size and current_chunk:
                 chunks.append({
                    "file_path": str(file_path),
                    "first_character_index": start_idx,
                    "last_character_index": start_idx + len(current_chunk) - 1,
                    "text": current_chunk
                })
                 start_idx += len(current_chunk)
                 current_chunk = para + "\n\n"
            else:
                current_chunk += para + "\n\n"

        if current_chunk:
             chunks.append({
                "file_path": str(file_path),
                "first_character_index": start_idx,
                "last_character_index": start_idx + len(current_chunk) - 1,
                "text": current_chunk
            })
        return chunks

    def index_corpus(self) -> None:
        """
        コーパスを読み込み、チャンキングしてインデックスを作成します。
        """
        if not self.corpus_dir.exists():
            raise FileNotFoundError(f"Corpus directory not found: {self.corpus_dir}")

        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # インデックス対象のファイル拡張子
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
                # バイナリファイルやエンコーディングエラーをスキップ
                continue

        print(f"Created {len(self.chunks)} chunks. Building BM25 index...")
        
        # BM25インデックスの構築
        # 単純な空白分割によるトークナイズ
        tokenized_corpus = [chunk["text"].split(" ") for chunk in tqdm(self.chunks, desc="Tokenizing")]
        self.bm25 = BM25Okapi(tokenized_corpus)

        self.save_index()
        print(f"Ingestion complete! Indexed {len(self.chunks)} chunks under {self.processed_dir}/")

    def save_index(self) -> None:
        """
        チャンク情報とコーパスを保存します（実際のBM25オブジェクトは再計算が速いためデータのみ保存）
        """
        chunks_path = self.processed_dir / "chunks.json"
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

    def load_index(self) -> None:
        """
        保存されたチャンク情報を読み込み、BM25を再構築します。
        """
        chunks_path = self.processed_dir / "chunks.json"
        if not chunks_path.exists():
            raise FileNotFoundError("Index not found. Please run indexing first.")

        with open(chunks_path, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
            
        tokenized_corpus = [chunk["text"].split(" ") for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
