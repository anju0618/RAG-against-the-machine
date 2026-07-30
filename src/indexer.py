import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, cast
from tqdm import tqdm  # type: ignore
from rank_bm25 import BM25Okapi  # type: ignore

# 検索ノイズを排除するためのストップワード（冠詞や前置詞など）の定義
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
    【役割】図書館の司書
    大量のソースコードやドキュメントを読み込み、
    検索しやすい状態（インデックス）に整理して保存するクラスです。
    """

    def __init__(self, max_chunk_size: int = 2000) -> None:
        """
        システムの初期設定を行います。

        Args:
            max_chunk_size: 1つのチャンクが許容する最大文字数（デフォルト: 2000文字）
        """
        self.max_chunk_size = max_chunk_size
        self.corpus_dir = Path("data/raw/vllm-0.10.1")
        self.processed_dir = Path("data/processed")

        # チャンクのリスト
        # 形の例:
        # [
        #     {
        #         "file_path": "data/raw/vllm-0.10.1/README.md",
        #         "first_character_index": 0,
        #         "last_character_index": 1999,
        #         "text": "..."
        #     }
        # ]
        self.chunks: List[Dict[str, Any]] = []

        # BM25検索エンジンのインスタンス
        self.bm25: Optional[BM25Okapi] = None

    def _safe_append(
        self,
        chunks: List[Dict[str, Any]],
        file_path: str,
        start_idx: int,
        text: str
    ) -> None:
        """
        【安全装置】
        2000文字の絶対上限を守るため、超過したテキストを強制的に分割して追加します。

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
        【Pythonファイル用の分割戦略】
        コードは「行（改行）」単位で意味を持つため、行ごとに分割してチャンクを作ります。
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
        【Markdownファイル用の分割戦略】
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
        【メイン処理】すべてのファイルを走査・分割し、ストップワードを除去した上でBM25インデックスを構築・保存します。
        """
        if not self.corpus_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.corpus_dir}")

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
                    file_chunks = self.chunk_python_code(text, str(file_path))
                else:
                    file_chunks = self.chunk_markdown_text(
                        text, str(file_path)
                    )

                self.chunks.extend(file_chunks)
            except Exception:
                continue

        print(f"Created {len(self.chunks)} chunks. Building BM25 index...")

        # トークン化の形・例:
        # [["how", "configure", "openai", "server"], ["vllm", "is", ...], ...]
        tokenized_corpus = [
            [
                w for w in re.findall(r'\w+', chunk["text"].lower())
                if w not in STOPWORDS
            ]
            for chunk in tqdm(self.chunks, desc="Tokenizing")
        ]

        self.bm25 = BM25Okapi(tokenized_corpus)
        self.save_index()
        print(f"Indexed {len(self.chunks)} chunks to {self.processed_dir}/")

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
