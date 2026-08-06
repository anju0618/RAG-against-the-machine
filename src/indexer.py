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
SKIP_DIR_NAMES = {
    ".git", "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "node_modules", ".venv", "venv", "env", "dist", "build",
    ".idea", ".vscode", ".tox", "site-packages", ".eggs",
}
_CAMEL_BOUNDARY_1 = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_CAMEL_BOUNDARY_2 = re.compile(r"(?<=[A-Z])(?=[A-Z][a-z])")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
_MARKDOWN_HEADING_RE = re.compile(r'(?m)^#{1,6} .*$')
_WORD_RE = re.compile(r"\w+")


def _split_identifier(word: str) -> List[str]:

    s = _CAMEL_BOUNDARY_1.sub("_", word)
    s = _CAMEL_BOUNDARY_2.sub("_", s)
    return [p for p in s.split("_") if p]


def tokenize_text(text: str) -> List[str]:

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
    """
    Args:
        file_path: チャンクの元になったファイルのパス
        text: チャンク本文

    Returns:
        ファイル名ヒントを付与した検索用テキスト
    """
    path_hint = re.sub(r'[/\\_.-]', ' ', file_path)

    return f"{path_hint}\n{text}"


def _encode_texts(
    embed_model: SentenceTransformer,
    texts: List[str],
    cpu_count: Optional[int],
    use_multiprocess: bool = True,
    batch_size: int = 32,
) -> np.ndarray:
    """
    embed_model.encode() 呼び出し
    Returns:
        ベクトル配列
    """
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
            print(
                "Warning: multi-process embedding failed "
                f"({e}); falling back to single-process encoding."
            )

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

    def __init__(
            self,
            max_chunk_size: int = 2000,
            skip_vector: bool = False,
            use_multiprocess: bool = True,
            target_chunk_size: Optional[int] = None,
            chunk_overlap: int = 150,
            ) -> None:
        """
        Args:
            max_chunk_size: 1チャンクの絶対的な上限文字数（ハードキャップ）。
                moulinette側のmax_context_length(2000)を超えると評価が
                無効になるため、_safe_appendはこの値を常に守る。
            skip_vector: Trueの場合、埋め込み計算をスキップする
                （開発用の高速モード。採点用インデックスには使わない）。
            use_multiprocess: 埋め込み計算をマルチプロセス化するか
                （失敗時は単一プロセスに自動フォールバック）。
            target_chunk_size: Markdown（.md/.txt/.rst）チャンクを
                積み上げる際の「目標」サイズ（ソフトな閾値）。
                Noneの場合は max(400, max_chunk_size // 2) を使う

            chunk_overlap: Markdownチャンクを確定する際に、直前の
                チャンク末尾からこの文字数だけ次のチャンクの先頭に
                重複して含める（オーバーラップ）。
                段落や見出しの境界にちょうど正解範囲がまたがっている
                場合でも、どちらか一方のチャンクに完全に収まる
                可能性を上げるための工夫。
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

        # [
        #     {
        #         "file_path": "data/raw/vllm-0.10.1/README.md",
        #         "first_character_index": 0,
        #         "last_character_index": 1999,
        #         "text": "..."
        #     },
        #     ...
        # ]
        self.chunks: List[Dict[str, Any]] = []
        self.bm25: Optional[BM25Okapi] = None

    def _safe_append(
        self,
        chunks: List[Dict[str, Any]],
        file_path: str,
        start_idx: int,
        text: str
    ) -> None:

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

        chunks: List[Dict[str, Any]] = []
        paragraphs = text.split('\n\n')
        current_chunk = ""
        start_idx = 0

        for i, para in enumerate(paragraphs):
            para_text = para + '\n\n' if i < len(paragraphs) - 1 else para
            para_len = len(para_text)

            # ハードな上限(max_chunk_size)ではなく、ソフトな目標サイズ
            # (target_chunk_size)で早めに区切ることで、chunkをより
            # トピックに集中させる
            if len(current_chunk) + para_len > self.target_chunk_size \
               and current_chunk:
                self._safe_append(
                    chunks, str(file_path), start_idx, current_chunk
                )
                # オーバーラップ: 直前チャンクの末尾 chunk_overlap 文字を
                # 次のチャンクの先頭に重複させて含める段落境界に
                # ちょうど正解範囲がまたがっているケースでも、
                # どちらかのチャンクに収まりやすくなる
                overlap_text = (
                    current_chunk[-self.chunk_overlap:]
                    if self.chunk_overlap > 0 else ""
                )
                start_idx = (
                    start_idx + len(current_chunk) - len(overlap_text)
                )
                current_chunk = overlap_text

            current_chunk += para_text

        if current_chunk:
            self._safe_append(chunks, str(file_path), start_idx, current_chunk)

        return chunks

    def chunk_markdown_by_headings(
        self, text: str, file_path: str
    ) -> List[Dict[str, Any]]:

        heading_starts = [
            m.start() for m in _MARKDOWN_HEADING_RE.finditer(text)
        ]

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
            # ハードな上限ではなく、ソフトな目標サイズ(target_chunk_size)
            # で区切る
            if (
                current_chunk
                and len(current_chunk) + section_len
                > self.target_chunk_size
            ):
                self._safe_append(
                    chunks, str(file_path), current_start, current_chunk
                )
                # オーバーラップ: 見出しセクションの境界をまたぐ内容も
                # 拾えるよう、直前チャンクの末尾を次のチャンクの先頭に
                # 重複させて含める。
                overlap_text = (
                    current_chunk[-self.chunk_overlap:]
                    if self.chunk_overlap > 0 else ""
                )
                current_start = (
                    current_start + len(current_chunk) - len(overlap_text)
                )
                current_chunk = overlap_text

            current_chunk += section_text

        if current_chunk:
            self._safe_append(
                chunks, str(file_path), current_start, current_chunk
            )

        return chunks

    def _walk_corpus_files(self, target_extensions: set[str]) -> List[Path]:
        """
        Args:
            target_extensions: 対象とするファイル拡張子の集合

        Returns:
            条件に合致するファイルパスのリスト
        """
        matched_files: List[Path] = []
        for root, dirs, files in os.walk(self.corpus_dir):
            dirs[:] = [
                d for d in dirs
                if d not in SKIP_DIR_NAMES and not d.startswith(".")
            ]
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix in target_extensions:
                    matched_files.append(file_path)
        return matched_files

    def index_corpus(self) -> None:
        """
        コーパス全体をインデックス化
        Raises:
            FileNotFoundError: self.corpus_dir が存在しない場合
        """
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

        # 現在のファイル一覧と更新日時
        current_files: Dict[str, float] = {}
        filepaths_to_process = []

        for file_path in self._walk_corpus_files(target_extensions):
            str_path = str(file_path)
            mtime = file_path.stat().st_mtime
            current_files[str_path] = mtime
            if (
                str_path not in file_meta
                or file_meta[str_path] != mtime
            ):
                filepaths_to_process.append(file_path)
        # 前回はあったが今回は見当たらないファイル＝削除されたファイル
        deleted_files = set(file_meta.keys()) - set(current_files.keys())

        print(
            "Incremental Indexing: "
            f"{len(filepaths_to_process)} files to update, "
            f"{len(deleted_files)} files deleted."
        )

        # 変更・追加ファイルの処理とチャンク作成
        new_chunks_by_file: Dict[str, List[Dict[str, Any]]] = {}

        for file_path in tqdm(
                filepaths_to_process,
                desc="Chunking"
                ):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()

                str_path = str(file_path)
                if file_path.suffix == '.py':
                    file_chunks = self.chunk_python_code(text, str_path)
                elif file_path.suffix == '.md':
                    file_chunks = self.chunk_markdown_by_headings(
                        text, str_path
                    )
                else:
                    file_chunks = self.chunk_markdown_text(text, str_path)

                new_chunks_by_file[str_path] = file_chunks
            except Exception:
                continue

        chunking_end_time = time.time()
        lexical_time = chunking_end_time - start_time
        final_chunks: List[Dict[str, Any]] = []
        embeddings_for_chunks: List[Optional[np.ndarray]] = []

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
                    if emb is not None and np.any(emb):
                        embeddings_for_chunks.append(emb)
                    else:
                        embeddings_for_chunks.append(None)
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
        missing_indices = [
            i for i, e in enumerate(embeddings_for_chunks) if e is None
        ]
        vector_time = 0.0

        if self.skip_vector:
            if missing_indices:
                print(
                    "Skipping semantic embeddings (Vector) computation "
                    f"for {len(missing_indices)} new/changed chunk(s) "
                    "(--skip_vector). Run `index` without --skip_vector "
                    "later to compute them."
                )
            else:
                print(
                    "Skipping semantic embeddings (Vector) computation "
                    "(--skip_vector; nothing was missing anyway)."
                )
        elif missing_indices:
            print(
                "Computing semantic embeddings (Vector) for "
                f"{len(missing_indices)} / {len(final_chunks)} chunks..."
            )
            embed_start_time = time.time()
            embed_model = SentenceTransformer(
                EMBEDDING_MODEL_NAME, device="cpu"
            )
            texts_to_embed = [
                build_search_text(
                    final_chunks[i]["file_path"], final_chunks[i]["text"]
                )
                for i in missing_indices
            ]
            cpu_count = os.cpu_count()
            new_embs = _encode_texts(
                embed_model, texts_to_embed, cpu_count,
                use_multiprocess=self.use_multiprocess,
            )
            for pos, i in enumerate(missing_indices):
                embeddings_for_chunks[i] = new_embs[pos]

            vector_time = time.time() - embed_start_time
        else:
            print(
                "All chunks already have up-to-date embeddings; "
                "nothing to compute."
            )

        if final_chunks:
            final_embeddings = np.array([
                e if e is not None
                else np.zeros(EMBEDDING_DIM, dtype=np.float32)
                for e in embeddings_for_chunks
            ], dtype=np.float32)
        else:
            final_embeddings = np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

        # ディスク
        self.save_index()
        np.save(embeddings_path, final_embeddings)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(current_files, f, ensure_ascii=False, indent=2)

        total_time = time.time() - start_time

        print(
            f"Incremental indexing complete. "
            f"Total chunks: {len(self.chunks)}"
        )

        print("\n⏱️ Indexing Time Breakdown:")
        print(f"  - Lexical (Chunking & Setup): {lexical_time:.2f} seconds")
        if self.skip_vector:
            print("  - Vector (Embedding): skipped (--skip_vector)")
        else:
            print(f"  - Vector (Embedding): {vector_time:.2f} seconds")
        print(f"  - Total Time: {total_time:.2f} seconds\n")

        remaining_missing = sum(
            1 for e in embeddings_for_chunks if e is None
        )
        if remaining_missing:
            print(
                f"⚠️  {remaining_missing} chunk(s) still have no semantic "
                "embedding (BM25-only for those until the next `index` "
                "run without --skip_vector)."
            )

    def save_index(self) -> None:
        """
        現在保持しているチャンクデータを
        JSON ファイル (data/processed/chunks.json) としてディスクに保存
        """
        chunks_path = self.processed_dir / "chunks.json"
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

    def load_index(self) -> None:
        """
        保存されている chunks.jsonから
        BM25検索エンジンを再構築
        """
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
