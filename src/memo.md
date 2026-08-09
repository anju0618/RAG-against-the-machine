# RAG
## 1. データモデル定義 (`models.py`)
```python
import uuid
from typing import List
from pydantic import BaseModel, Field

class MinimalSource(BaseModel):
    """
    【クラスの役割】
    検索された情報源（チャンク）の位置情報を正確に特定するためのモデル。
    実ファイルパスと、そのファイル内における文字インデックスの開始・終了位置を保持します。
    """
    file_path: str                  # 該当ファイルのパス（例: "data/raw/.../README.md"）
    first_character_index: int      # ファイル内でのチャンク開始文字インデックス（0始まり）
    last_character_index: int       # ファイル内でのチャンク終了文字インデックス

class UnansweredQuestion(BaseModel):
    """
    【クラスの役割】
    まだ回答が生成されていない、質問データセット内の1件の質問を表すモデル。
    """
    # question_id: なぜ UUID を使うのか？一意性を確実に担保し、異なるデータセット間や分散処理環境でも重複しないIDを自動生成するため。
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str                   # ユーザーからの質問文

class AnsweredQuestion(UnansweredQuestion):
    """
    【クラスの役割】
    質問に対して、参照すべきソース（MinimalSource）と生成された回答（answer）が紐づいた状態のモデル。
    """
    sources: List[MinimalSource]    # 根拠となったソースのリスト
    answer: str                     # LLMが生成した回答文

class RagDataset(BaseModel):
    """
    【クラスの役割】
    評価用データセットのJSONファイル全体の構造（複数の質問リスト）を表現するモデル。
    """
    rag_questions: List[AnsweredQuestion | UnansweredQuestion]

class MinimalSearchResults(BaseModel):
    """
    【クラスの役割】
    1つの質問に対して、検索エンジンが導き出した上位 $k$ 件の検索結果を格納するモデル。
    """
    question_id: str                # 質問ID
    question: str                   # 質問文
    retrieved_sources: List[MinimalSource]  # 検索された上位ソースのリスト

class MinimalAnswer(MinimalSearchResults):
    """
    【クラスの役割】
    MinimalSearchResults を継承し、LLMが生成した自然言語の回答文字列をフィールドとして追加したモデル。
    """
    answer: str                     # 生成された回答

class StudentSearchResults(BaseModel):
    """
    【クラスの役割】
    検索コマンド (`search_dataset`) がファイルとして出力する構造を表現するモデル。
    """
    search_results: List[MinimalSearchResults]
    k: int                          # 取得件数

class StudentSearchResultsAndAnswer(BaseModel):
    """
    【クラスの役割】
    回答生成コマンド (`answer_dataset`) がファイルとして出力する最終的なJSONフォーマット。
    """
    search_results: List[MinimalAnswer]
    k: int

class SearchRequest(BaseModel):
    """【クラスの役割】FastAPIの POST /search エンドポイント受信用リクエストボディモデル。"""
    query: str
    k: int = 5

class AnswerRequest(BaseModel):
    """【クラスの役割】FastAPIの POST /answer エンドポイント受信用リクエストボディモデル。"""
    query: str
    k: int = 5
```
## 2. インデクサーとチャンク分割 (`indexer.py`)
```py
import json
import os
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, cast
from tqdm import tqdm
from rank_bm25 import BM25Okapi
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# 検索精度を向上させるため、意味を持たない一般的な英単語を排除するストップワードセット
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

# コーパス探索時にスキップすべきシステム・キャッシュ用ディレクトリ名
SKIP_DIR_NAMES = {
    ".git", "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "node_modules", ".venv", "venv", "env", "dist", "build",
    ".idea", ".vscode", ".tox", "site-packages", ".eggs",
}

_CAMEL_BOUNDARY_1 = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_CAMEL_BOUNDARY_2 = re.compile(r"(?<=[A-Z])(?=[A-Z][a-z])")

# 意味ベクトル検索で使用するHugging Face上のモデル名
# なぜこのモデルか？ "all-MiniLM-L6-v2" は軽量かつ高速でありながら、高品質な文脈ベクトルを生成できるため、CPU環境でも実用的な速度で動作する業界標準モデルであるため。
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# 埋め込みベクトルの次元数
# なぜ 384 なのか？ "all-MiniLM-L6-v2" モデルのアーキテクチャ仕様として出力次元数が 384 次元に固定されているため。
EMBEDDING_DIM = 384

_MARKDOWN_HEADING_RE = re.compile(r'(?m)^#{1,6} .*$')
_WORD_RE = re.compile(r"\w+")
QUERY_INSTRUCTION_PREFIX = ""

def _split_identifier(word: str) -> List[str]:
    """キャメルケースやスネークケースの識別子を単語単位に分割するヘルパー関数"""
    s = _CAMEL_BOUNDARY_1.sub("_", word)
    s = _CAMEL_BOUNDARY_2.sub("_", s)
    return [p for p in s.split("_") if p]

def tokenize_text(text: str) -> List[str]:
    """生テキストを小文字化し、ストップワードを除去した上でBM25用のトークンに分割する"""
    tokens: List[str] = []
    for raw in _WORD_RE.findall(text):
        lower_whole = raw.lower()
        if lower_whole not in STOPWORDS:
            tokens.append(lower_whole)

        parts = _split_identifier(raw)
        if len(parts) > 1:
            for part in parts:
                lower_part = part.lower()
                if lower_part and lower_part not in STOPWORDS:
                    tokens.append(lower_part)

    return tokens

def build_search_text(file_path: str, text: str) -> str:
    """ファイルパス自体を検索用テキストのヒントとしてプレフィックスに付与する"""
    path_hint = re.sub(r'[/\\_.-]', ' ', file_path)
    return f"{path_hint}\n{text}"

def _encode_texts(
    embed_model: SentenceTransformer,
    texts: List[str],
    cpu_count: Optional[int],
    use_multiprocess: bool = True,
    batch_size: int = 32,
) -> np.ndarray:
    """テキストリストをSentenceTransformerを用いて高次元のセマンティック埋め込みベクトルに変換する"""
    if not texts:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

    if use_multiprocess and cpu_count and cpu_count > 1:
        try:
            torch.set_num_threads(1)
            pool = embed_model.start_multi_process_pool(
                target_devices=["cpu"] * cpu_count
            )
            try:
                raw_embeddings = embed_model.encode_multi_process(
                    texts, pool, batch_size=batch_size,
                )
            finally:
                embed_model.stop_multi_process_pool(pool)

            mp_embeddings: np.ndarray = np.asarray(raw_embeddings)
            norms = np.linalg.norm(mp_embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            result: np.ndarray = (mp_embeddings / norms).astype(np.float32)
            return result
        except Exception as e:
            print(f"Warning: multi-process embedding failed ({e}); falling back.")

    if cpu_count:
        torch.set_num_threads(cpu_count)
    raw_single = embed_model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return np.asarray(raw_single, dtype=np.float32)

class Indexer:
    """ローカルのソースコードやドキュメントを走査し、チャンク化とインデックス構築を行うクラス。"""
    def __init__(
        self,
        max_chunk_size: int = 2000,
        skip_vector: bool = False,
        use_multiprocess: bool = True,
        target_chunk_size: Optional[int] = None,
        chunk_overlap: int = 150,
    ) -> None:
        """
        【パラメータの選定理由解説】
        - max_chunk_size=2000: RAGの評価システムの制約（max_context_length）に確実に収まるハードキャップ。
        - target_chunk_size: 指定がない場合 max(400, max_chunk_size // 2)。トピックのまとまりを維持しつつ適切なサイズにするため。
        - chunk_overlap=150: 境界をまたぐ正解情報の分断による検索漏れを防ぐため、150文字を次のチャンクに重複させる。
        """
        self.max_chunk_size = max_chunk_size
        self.skip_vector = skip_vector
        self.use_multiprocess = use_multiprocess
        self.target_chunk_size = (
            target_chunk_size
            if target_chunk_size is not None
            else max(400, max_chunk_size // 2)
        )
        self.chunk_overlap = max(0, chunk_overlap)
        self.corpus_dir = Path("data/raw")
        self.processed_dir = Path("data/processed")
        self.chunks: List[Dict[str, Any]] = []
        self.bm25: Optional[BM25Okapi] = None

    def _safe_append(
        self,
        chunks: List[Dict[str, Any]],
        file_path: str,
        start_idx: int,
        text: str
    ) -> None:
        """テキストを指定された max_chunk_size 以内に確実に収めるように分割し、メタデータと共にリストに追加する"""
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
        """Pythonコードファイルを行単位で走査し、論理的な行の区切りを意識しながらチャンクに分割する"""
        chunks: List[Dict[str, Any]] = []
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
        """Markdownやテキストファイルを段落単位で分割し、目標サイズを超えた段階でオーバーラップを考慮して区切る"""
        chunks: List[Dict[str, Any]] = []
        paragraphs = text.split('\n\n')
        current_chunk = ""
        start_idx = 0

        for i, para in enumerate(paragraphs):
            para_text = para + '\n\n' if i < len(paragraphs) - 1 else para
            para_len = len(para_text)

            if len(current_chunk) + para_len > self.target_chunk_size and current_chunk:
                self._safe_append(chunks, str(file_path), start_idx, current_chunk)
                overlap_text = (
                    current_chunk[-self.chunk_overlap:]
                    if self.chunk_overlap > 0 else ""
                )
                start_idx = start_idx + len(current_chunk) - len(overlap_text)
                current_chunk = overlap_text

            current_chunk += para_text

        if current_chunk:
            self._safe_append(chunks, str(file_path), start_idx, current_chunk)

        return chunks

    def chunk_markdown_by_headings(self, text: str, file_path: str) -> List[Dict[str, Any]]:
        """Markdownファイルをヘッダー位置に基づいてセクションごとに分割する"""
        heading_starts = [m.start() for m in _MARKDOWN_HEADING_RE.finditer(text)]
        if not heading_starts:
            return self.chunk_markdown_text(text, file_path)

        boundaries = sorted(set([0] + heading_starts + [len(text)]))
        sections: List[Tuple[int, str]] = []
        for i in range(len(boundaries) - 1):
            s, e = boundaries[i], boundaries[i + 1]
            if e > s:
                sections.append((s, text[s:e]))

        chunks: List[Dict[str, Any]] = []
        current_chunk = ""
        current_start = 0

        for section_start, section_text in sections:
            if not current_chunk:
                current_start = section_start

            section_len = len(section_text)
            if current_chunk and len(current_chunk) + section_len > self.target_chunk_size:
                self._safe_append(chunks, str(file_path), current_start, current_chunk)
                overlap_text = (
                    current_chunk[-self.chunk_overlap:]
                    if self.chunk_overlap > 0 else ""
                )
                current_start = current_start + len(current_chunk) - len(overlap_text)
                current_chunk = overlap_text

            current_chunk += section_text

        if current_chunk:
            self._safe_append(chunks, str(file_path), current_start, current_chunk)

        return chunks

    def _walk_corpus_files(self, target_extensions: set[str]) -> List[Path]:
        """対象ディレクトリ内を再帰的に走査し、ファイルパスのリストを取得する"""
        matched_files: List[Path] = []
        for root, dirs, files in os.walk(self.corpus_dir):
            dirs[:] = [d for d in dirs if d not in SKIP_DIR_NAMES and not d.startswith(".")]
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix in target_extensions:
                    matched_files.append(file_path)
        return matched_files

    def index_corpus(self) -> None:
        """増分インデックス更新に対応したコーパス全体のインデックス構築を実行する"""
        start_time = time.time()
        if not self.corpus_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.corpus_dir}")

        self.processed_dir.mkdir(parents=True, exist_ok=True)
        target_extensions = {'.py', '.md', '.txt', '.rst'}
        chunks_path = self.processed_dir / "chunks.json"
        embeddings_path = self.processed_dir / "embeddings.npy"
        meta_path = self.processed_dir / "file_meta.json"
        
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

        current_files: Dict[str, float] = {}
        filepaths_to_process = []

        for file_path in self._walk_corpus_files(target_extensions):
            str_path = str(file_path)
            mtime = file_path.stat().st_mtime
            current_files[str_path] = mtime
            if str_path not in file_meta or file_meta[str_path] != mtime:
                filepaths_to_process.append(file_path)

        deleted_files = set(file_meta.keys()) - set(current_files.keys())
        new_chunks_by_file: Dict[str, List[Dict[str, Any]]] = {}

        for file_path in tqdm(filepaths_to_process, desc="Chunking"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                str_path = str(file_path)
                if file_path.suffix == '.py':
                    file_chunks = self.chunk_python_code(text, str_path)
                elif file_path.suffix == '.md':
                    file_chunks = self.chunk_markdown_by_headings(text, str_path)
                else:
                    file_chunks = self.chunk_markdown_text(text, str_path)
                new_chunks_by_file[str_path] = file_chunks
            except Exception:
                continue

        final_chunks: List[Dict[str, Any]] = []
        embeddings_for_chunks: List[Optional[np.ndarray]] = []

        if existing_chunks and existing_embeddings is not None and len(existing_chunks) == len(existing_embeddings):
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
                    embeddings_for_chunks.append(emb if emb is not None and np.any(emb) else None)
        elif existing_chunks:
            for chunk in existing_chunks:
                fpath = chunk["file_path"]
                if fpath in deleted_files or fpath in new_chunks_by_file:
                    continue
                final_chunks.append(chunk)
                embeddings_for_chunks.append(None)

        for str_path, chunks in new_chunks_by_file.items():
            for chunk in chunks:
                final_chunks.append(chunk)
                embeddings_for_chunks.append(None)

        self.chunks = final_chunks
        missing_indices = [i for i, e in enumerate(embeddings_for_chunks) if e is None]
        vector_time = 0.0

        if self.skip_vector:
            pass
        elif missing_indices:
            embed_start_time = time.time()
            embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
            texts_to_embed = [
                build_search_text(final_chunks[i]["file_path"], final_chunks[i]["text"])
                for i in missing_indices
            ]
            new_embs = _encode_texts(embed_model, texts_to_embed, os.cpu_count(), use_multiprocess=self.use_multiprocess)
            for pos, i in enumerate(missing_indices):
                embeddings_for_chunks[i] = new_embs[pos]
            vector_time = time.time() - embed_start_time

        final_embeddings = np.array([
            e if e is not None else np.zeros(EMBEDDING_DIM, dtype=np.float32)
            for e in embeddings_for_chunks
        ], dtype=np.float32) if final_chunks else np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

        self.save_index()
        np.save(embeddings_path, final_embeddings)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(current_files, f, ensure_ascii=False, indent=2)

    def save_index(self) -> None:
        """チャンクメタデータをJSONファイルとして保存する"""
        chunks_path = self.processed_dir / "chunks.json"
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

    def load_index(self) -> None:
        """chunks.jsonを読み込み、BM25Okapiインスタンスを初期化する"""
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
```
3. リトリーバー(`retriever.py`)
```py
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, cast
from functools import lru_cache
from rank_bm25 import BM25Okapi
import numpy as np
from sentence_transformers import SentenceTransformer
from src.models import MinimalSource, MinimalSearchResults, StudentSearchResults
from src.indexer import EMBEDDING_MODEL_NAME, build_search_text, tokenize_text

class Retriever:
    """インスタンス生成時に一度だけインデックスをメモリ上にロードし、高速なハイブリッド検索を提供するクラス。"""
    def __init__(self) -> None:
        self.processed_dir = Path("data/processed")
        self.chunks: List[Dict[str, Any]] = []
        self.bm25: BM25Okapi | None = None
        self.embeddings: np.ndarray | None = None
        self.embed_model: SentenceTransformer | None = None

        self.load_index()
        self.load_semantic_index()

    def load_index(self) -> None:
        """chunks.jsonを読み込み、BM25Okapiインスタンスを初期化する"""
        chunks_path = self.processed_dir / "chunks.json"
        if not chunks_path.exists():
            raise FileNotFoundError("Index not found. Run 'index' first.")

        with open(chunks_path, 'r', encoding='utf-8') as f:
            self.chunks = cast(List[Dict[str, Any]], json.load(f))

        tokenized_corpus = [
            tokenize_text(build_search_text(chunk["file_path"], chunk["text"]))
            for chunk in self.chunks
        ]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def load_semantic_index(self) -> None:
        """embeddings.npyをロードし、埋め込みモデルを準備する"""
        embeddings_path = self.processed_dir / "embeddings.npy"
        if embeddings_path.exists():
            try:
                self.embeddings = np.load(embeddings_path)
                self.embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
            except Exception as e:
                print(f"Warning: Could not load semantic embeddings: {e}")

    def search(self, query: str, k: int = 5) -> List[MinimalSource]:
        """キャッシュ機構を経由して単一クエリの検索を実行する"""
        return self._cached_search(query, k)

    @lru_cache(maxsize=1024)
    def _cached_search(self, query: str, k: int) -> List[MinimalSource]:
        """同一の (query, k) ペアに対する検索結果を最大1024件までメモリ上にキャッシュする"""
        return self.hybrid_search(query, k)

    def semantic_search(self, query: str, k: int = 5) -> List[Tuple[int, float]]:
        """クエリをベクトル化し、コサイン類似度を計算して上位 k 件を返す"""
        if self.embeddings is None or self.embed_model is None:
            return []

        query_embedding = self.embed_model.encode(
            QUERY_INSTRUCTION_PREFIX + query, normalize_embeddings=True
        )
        scores = np.dot(self.embeddings, query_embedding)
        top_indices = np.argsort(scores)[::-1][:k]

        return [(int(idx), float(scores[idx])) for idx in top_indices]

    def hybrid_search(self, query: str, k: int = 5) -> List[MinimalSource]:
        """
        BM25とセマンティック検索の結果を RRF (Reciprocal Rank Fusion) で統合する。
        【なぜ RRF と rrf_k = 60 なのか？】
        - 異なるスコアスケールの検索手法を順位ベースで公平に統合できるため。
        - rrf_k = 60 は情報検索ベンチマークにおいて最も頑健で優れたパフォーマンスを発揮する標準的な定数値。
        """
        if self.bm25 is None:
            raise RuntimeError("BM25 index is not loaded.")

        fetch_k = max(k * 10, 150)
        tokenized_query = tokenize_text(query)
        bm25_indices = self.bm25.get_top_n(tokenized_query, range(len(self.chunks)), n=fetch_k)

        semantic_results = self.semantic_search(query, k=fetch_k)
        semantic_indices = [idx for idx, _ in semantic_results]

        rrf_scores: Dict[int, float] = {}
        rrf_k = 60

        for rank, idx in enumerate(bm25_indices):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)

        for rank, idx in enumerate(semantic_indices):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)

        sorted_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:k]

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

    def search_dataset(self, dataset_path: str, k: int = 5) -> StudentSearchResults:
        """データセットファイル（JSON）を読み込み、各質問に対して一括で検索を実行する"""
        dataset_file = Path(dataset_path)
        if not dataset_file.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        with open(dataset_file, 'r', encoding='utf-8') as f:
            dataset_data = cast(Dict[str, Any], json.load(f))

        results = []
        questions = cast(List[Dict[str, Any]], dataset_data.get("rag_questions", []))

        for q in questions:
            q_id = str(q["question_id"])
            q_text = str(q["question"])
            sources = self.search(q_text, k)
            results.append(MinimalSearchResults(question_id=q_id, question=q_text, retrieved_sources=sources))

        return StudentSearchResults(search_results=results, k=k)
```
## 4. ジェネレーター(`generator.py`)
```py
import itertools
from pathlib import Path
from typing import List, Dict, Any, cast
import torch
from more_itertools import batched
from transformers import AutoModelForCausalLM, AutoTokenizer, BatchEncoding
from src.models import MinimalSource, MinimalSearchResults, MinimalAnswer, StudentSearchResultsAndAnswer
from src.retriever import Retriever

class Generator:
    """ローカルLLMを用いて検索結果に基づく正確な回答を生成するクラス"""
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        batch_size: int = 1,
        max_new_tokens: int = 256,
        k: int = 5,
    ) -> None:
        """
        【パラメータの選定理由】
        - model_name="Qwen/Qwen3-0.6B": 軽量かつ高性能でローカル環境でもスムーズに動作するサイズ。
        - batch_size=1: スループットとメモリ使用量のトレードオフを考慮した設定。
        - max_new_tokens=256: 必要な説明や根拠を示すのに十分でありつつ、生成の暴走を防ぐ上限値。
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.k = k
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto" if self.device == "cuda" else "cpu",
        )

    def _load_chunk(self, source: MinimalSource) -> str:
        """ソース情報に基づき、実ファイルから該当部分の文字列を切り出して返す"""
        file_path = Path(source.file_path)
        if not file_path.exists():
            return f"[Error: File not found {source.file_path}]"
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            return content[source.first_character_index:source.last_character_index + 1]
        except Exception as e:
            return f"[Error loading chunk: {e}]"

    def _generate_prompt(self, result: MinimalSearchResults, k: int | None = None) -> List[Dict[str, str]]:
        """提供されたコンテキストのみに基づいて回答させるための厳格なシステムプロンプトを構築する"""
        effective_k = self.k if k is None else k
        system_prompt = {
            "role": "system",
            "content": (
                "You are a precise and helpful assistant. Answer the user's "
                "question using ONLY the retrieved context provided below. "
                "Follow these rules strictly:\n"
                "- If the answer is not in the context, say: \"I don't have "
                'enough information to answer that."\n'
                "- Do not use outside knowledge or make up information.\n"
                "- Keep answers concise and grounded in the provided text.\n"
                "- When possible, cite which document/source supports your answer."
            ),
        }
        chunks = [
            (source.file_path, self._load_chunk(source))
            for source in itertools.islice(result.retrieved_sources, effective_k)
        ]
        formatted_sources = [
            f"[Source {i}] File: {file_path}\nContent: {content}\n"
            for i, (file_path, content) in enumerate(chunks, start=1)
        ]
        context_str = "\n".join(formatted_sources)

        user_prompt = {
            "role": "user",
            "content": (
                "Retrieved Context:\n---\n"
                f"{context_str}"
                "---\n\n"
                f"Question: {result.question}\n\n"
                "Answer based only on the retrieved context above."
            ),
        }
        return [system_prompt, user_prompt]

    @torch.inference_mode()
    def generate_answers(self, search_results: List[MinimalSearchResults], k: int | None = None) -> List[str]:
        """勾配計算を無効化した推論モードで、バッチ処理を活用して回答テキストを生成する"""
        prompt_messages = [self._generate_prompt(res, k=k) for res in search_results]
        outputs: List[str] = []

        with torch.no_grad():
            for batch in batched(prompt_messages, self.batch_size):
                prompt_tokens = self.tokenizer.apply_chat_template(
                    list(batch), tokenize=True, padding=True, add_generation_prompt=True, return_tensors="pt"
                )
                assert isinstance(prompt_tokens, BatchEncoding)
                prompt_tokens = prompt_tokens.to(self.device)
                input_length = prompt_tokens["input_ids"].shape[1]
                generated_ids = cast(Any, self.model).generate(**prompt_tokens, max_new_tokens=self.max_new_tokens)
                new_tokens = generated_ids[:, input_length:]
                decoded = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
                outputs.extend(decoded)

        return outputs

    def answer_dataset(self, dataset_path: str, k: int = 5) -> StudentSearchResultsAndAnswer:
        """データセット全体の質問に対して検索とLLM回答生成を一気通貫で実行する"""
        retriever = Retriever()
        search_results_obj = retriever.search_dataset(dataset_path, k=k)
        answers = self.generate_answers(search_results_obj.search_results, k=k)

        minimal_answers: List[MinimalAnswer] = []
        for search_res, ans in zip(search_results_obj.search_results, answers):
            minimal_answers.append(
                MinimalAnswer(
                    question_id=search_res.question_id,
                    question=search_res.question,
                    retrieved_sources=search_res.retrieved_sources,
                    answer=ans,
                )
            )
        return StudentSearchResultsAndAnswer(search_results=minimal_answers, k=k)
```