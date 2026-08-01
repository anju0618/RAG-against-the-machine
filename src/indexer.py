import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, cast
from tqdm import tqdm
from rank_bm25 import BM25Okapi
import numpy as np
from sentence_transformers import SentenceTransformer

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "can", "could",
    "did", "do", "does", "doing", "down", "during", "each", "few", "for",
    "from", "further", "had", "has", "have", "having", "he", "her", "here",
    "hers", "herself", "him", "himself", "his", "how", "i", "if", "in",
    "into", "is", "it", "its", "itself", "just", "me", "more", "most", "my",
    "myself", "no", "nor", "not", "now", "of", "off", "on", "once", "only",
    "or", "other", "our", "ours", "ourselves", "out", "over", "own", "s",
    "same", "she", "should", "so", "some", "such", "t", "than", "that",
    "the", "their", "theirs", "them", "themselves", "then", "there", "these",
    "they", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "we", "were", "what", "when", "where", "which", "while",
    "who", "whom", "why", "will", "with", "you", "your", "yours", "yourself",
    "yourselves"
}


class Indexer:
    """
    Indexer using BM25Okapi and SentenceTransformers with Incremental Indexing
    """

    def __init__(self, max_chunk_size: int = 2000) -> None:
        """
        Args:
            max_chunk_size: max chunk size(default: 2000)
        """
        self.max_chunk_size = max_chunk_size
        self.corpus_dir = Path("data/raw/vllm-0.10.1")
        self.processed_dir = Path("data/processed")

        # chunks
        # ex:
        # [
        #     {
        #         "file_path": "data/raw/vllm-0.10.1/README.md",
        #         "first_character_index": 0,
        #         "last_character_index": 1999,
        #         "text": "..."
        #     }
        # ]
        self.chunks: List[Dict[str, Any]] = []
        # BM25 instance
        self.bm25: Optional[BM25Okapi] = None

    def _safe_append(
        self,
        chunks: List[Dict[str, Any]],
        file_path: str,
        start_idx: int,
        text: str
    ) -> None:
        """
        safe appender (not mehr 2000)
        Args:
            chunks: 追加先のチャンクリーンリスト
            file_path: 対象ファイルのパス (str)
            start_idx: ファイル内での開始文字インデックス (int)
            text: 分割対象のテキスト (str)
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

    def chunk_python_code(
        self, text: str, file_path: str
    ) -> List[Dict[str, Any]]:
        """
        【Python】行ごとに分割してチャンクを作ります。
        """
        chunks: List[Dict[str, Any]] = []
        lines = text.split('\n')
        current_chunk = ""
        start_idx = 0

        for i, line in enumerate(lines):
            line_text = line + '\n' if i < len(lines) - 1 else line
            line_len = len(line_text)

            if len(current_chunk) + line_len > self.max_chunk_size \
               and current_chunk:
                self._safe_append(
                    chunks, str(file_path), start_idx, current_chunk
                )
                start_idx += len(current_chunk)
                current_chunk = ""

            current_chunk += line_text

        if current_chunk:
            self._safe_append(chunks, str(file_path), start_idx, current_chunk)

        return chunks

    def chunk_markdown_text(
        self, text: str, file_path: str
    ) -> List[Dict[str, Any]]:
        """
        【Markdown】
        ドキュメントは「段落（空行）」単位で意味を持つため、\n\nで分割してチャンクを作ります。
        """
        chunks: List[Dict[str, Any]] = []
        paragraphs = text.split('\n\n')
        current_chunk = ""
        start_idx = 0

        for i, para in enumerate(paragraphs):
            para_text = para + '\n\n' if i < len(paragraphs) - 1 else para
            para_len = len(para_text)

            if len(current_chunk) + para_len > self.max_chunk_size \
               and current_chunk:
                self._safe_append(
                    chunks, str(file_path), start_idx, current_chunk
                )
                start_idx += len(current_chunk)
                current_chunk = ""

            current_chunk += para_text

        if current_chunk:
            self._safe_append(chunks, str(file_path), start_idx, current_chunk)

        return chunks

    def index_corpus(self) -> None:
        """
        差分インデックス（Incremental indexing）を用いて、
        変更・新規があったファイルのみを再インデックスします。
        """
        if not self.corpus_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.corpus_dir}")

        self.processed_dir.mkdir(parents=True, exist_ok=True)
        target_extensions = {'.py', '.md', '.txt', '.rst'}

        chunks_path = self.processed_dir / "chunks.json"
        embeddings_path = self.processed_dir / "embeddings.npy"
        meta_path = self.processed_dir / "file_meta.json"

        # 既存データとメタデータのロード
        existing_chunks: List[Dict[str, Any]] = []
        existing_embeddings: Optional[np.ndarray] = None
        file_meta: Dict[str, float] = {}

        if chunks_path.exists() and meta_path.exists():
            with open(chunks_path, "r", encoding="utf-8") as f:
                existing_chunks = json.load(f)
            with open(meta_path, "r", encoding="utf-8") as f:
                file_meta = json.load(f)
            if embeddings_path.exists():
                existing_embeddings = np.load(embeddings_path)

        # 現在のファイル一覧と更新日時（mtime）を取得
        current_files: Dict[str, float] = {}
        filepaths_to_process = []

        for root, _, files in os.walk(self.corpus_dir):
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix in target_extensions:
                    str_path = str(file_path)
                    mtime = file_path.stat().st_mtime
                    current_files[str_path] = mtime
                    if (
                        str_path not in file_meta
                        or file_meta[str_path] != mtime
                    ):
                        filepaths_to_process.append(file_path)

        deleted_files = set(file_meta.keys()) - set(current_files.keys())

        print(
            "Incremental Indexing: "
            f"{len(filepaths_to_process)} files to update, "
            f"{len(deleted_files)} files deleted."
        )

        # 変更・追加ファイルの処理とチャンク作成
        new_chunks_by_file: Dict[str, List[Dict[str, Any]]] = {}
        embed_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

        for file_path in tqdm(
                filepaths_to_process,
                desc="Incremental Chunking"
                ):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()

                str_path = str(file_path)
                if file_path.suffix == '.py':
                    file_chunks = self.chunk_python_code(text, str_path)
                else:
                    file_chunks = self.chunk_markdown_text(text, str_path)

                new_chunks_by_file[str_path] = file_chunks
            except Exception:
                continue

        # 新規・変更ファイルの埋め込み計算
        new_embeddings_by_file: Dict[str, np.ndarray] = {}
        for str_path, chunks in new_chunks_by_file.items():
            if chunks:
                texts = [c["text"] for c in chunks]
                embs = embed_model.encode(
                    texts, batch_size=64, normalize_embeddings=True
                )
                new_embeddings_by_file[str_path] = embs

        # 既存データと新規データの結合・再構築
        final_chunks: List[Dict[str, Any]] = []
        final_embeddings_list: List[np.ndarray] = []

        if (
            existing_chunks
            and existing_embeddings is not None
            and len(existing_chunks) == len(existing_embeddings)
        ):
            file_data: Dict[str, List[Tuple[Dict[str, Any], np.ndarray]]] = {}
            for i, chunk in enumerate(existing_chunks):
                fpath = chunk["file_path"]
                if fpath not in file_data:
                    file_data[fpath] = []
                file_data[fpath].append((chunk, existing_embeddings[i]))

            for fpath, pairs in file_data.items():
                if fpath in deleted_files or fpath in new_chunks_by_file:
                    continue
                for chunk, emb in pairs:
                    final_chunks.append(chunk)
                    final_embeddings_list.append(emb)

        for str_path, chunks in new_chunks_by_file.items():
            if chunks and str_path in new_embeddings_by_file:
                embs = new_embeddings_by_file[str_path]
                for i, chunk in enumerate(chunks):
                    final_chunks.append(chunk)
                    final_embeddings_list.append(embs[i])

        self.chunks = final_chunks
        if final_embeddings_list:
            final_embeddings = np.array(final_embeddings_list)
        else:
            final_embeddings = np.zeros((0, 384))

        self.save_index()
        np.save(embeddings_path, final_embeddings)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(current_files, f, ensure_ascii=False, indent=2)

        print(
            f"Incremental indexing complete. "
            f"Total chunks: {len(self.chunks)}"
        )

    def save_index(self) -> None:
        """
        作成したチャンクデータを JSON ファイル (data/processed/chunks.json) としてディスクに保存します。
        """
        chunks_path = self.processed_dir / "chunks.json"
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

    def load_index(self) -> None:
        """
        保存されている chunks.json を読み込み、BM25検索エンジンを再構築します。
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
