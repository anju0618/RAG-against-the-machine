"""
indexer.py
==========

コーパス（vLLMリポジトリなど）を読み込み、検索可能な形に変換するモジュール。

全体の流れは次の3ステップ：
  1. コーパス配下のファイルを走査し、拡張子でフィルタリングする
  2. ファイルの種類（Python / Markdown等）に応じて異なる方法でチャンク分割する
  3. チャンクごとにBM25用のトークン化とSentenceTransformerによる埋め込みを作り、
     data/processed/ 以下に永続化する

「差分インデックス（Incremental Indexing）」にも対応しており、前回実行時から
更新・追加・削除されたファイルだけを再計算することで、2回目以降の実行を
高速化している（ボーナス3: Incremental indexing）。
"""

import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, cast
from tqdm import tqdm
from rank_bm25 import BM25Okapi
import numpy as np
from sentence_transformers import SentenceTransformer

# BM25検索で使うストップワード（英語の一般的な機能語）一覧。
# これらの単語は情報量が少なく、キーワード検索のノイズになるため、
# トークン化の際に除去する。indexer / retriever の両方で共有利用する。
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
    BM25Okapi（キーワード検索）と SentenceTransformer（埋め込みベクトル）を
    組み合わせてコーパスをインデックス化するクラス。

    差分インデックス（Incremental Indexing）に対応しているため、
    2回目以降の index_corpus() 実行では変更されたファイルのみを
    再チャンク・再埋め込みし、既存のチャンク/埋め込みは使い回す。
    """

    def __init__(self, max_chunk_size: int = 2000) -> None:
        """
        Indexerを初期化する。

        Args:
            max_chunk_size: 1チャンクあたりの最大文字数（デフォルト: 2000）。
                仕様書の制約により、moulinette側の max_context_length と
                同じ 2000 文字を超えるチャンクを作ってはいけない
                （超過すると評価結果が丸ごと無効になる）。
        """
        self.max_chunk_size = max_chunk_size

        # コーパスのルートディレクトリ。
        # 特定のバージョン名（例: "vllm-0.10.1"）をハードコードすると、
        # 評価環境でフォルダ名が変わった瞬間にインデックス作成が失敗する。
        # そのため data/raw/ 配下を再帰的に os.walk() で走査する設計にし、
        # 実際のサブフォルダ名に依存しないようにしている。
        # file_path はこの走査で見つかった実パス
        # （例: "data/raw/vllm-0.10.1/README.md"）がそのまま使われるため、
        # 「file_pathは検索対象コーパスのパスと完全一致する」という
        # 仕様要件も自然に満たされる。
        self.corpus_dir = Path("data/raw")
        self.processed_dir = Path("data/processed")

        # 生成されたチャンクを保持するリスト。
        # 各要素は以下のような辞書：
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

        # BM25インデックス本体。load_index() が呼ばれるまではNoneのまま。
        self.bm25: Optional[BM25Okapi] = None

    def _safe_append(
        self,
        chunks: List[Dict[str, Any]],
        file_path: str,
        start_idx: int,
        text: str
    ) -> None:
        """
        テキストを max_chunk_size 以下の断片に強制的に分割してから
        chunks リストへ追加する安全弁（セーフガード）。

        chunk_python_code / chunk_markdown_text は「行」や「段落」を
        単位に積み上げていくが、1行・1段落自体が max_chunk_size を
        超える場合（長いコード行や長大な段落）がありうる。
        このメソッドはそのケースでも必ず max_chunk_size 以下の
        チャンクだけが出力されるように、文字単位でスライスし直す。

        Args:
            chunks: 追加先のチャンクリスト（呼び出し元が保持するリストを
                そのまま書き換える＝副作用あり）
            file_path: 対象ファイルのパス (str)
            start_idx: ファイル全体における、このtext断片の開始文字位置
            text: 分割対象のテキスト (str)
        """
        idx = 0
        while idx < len(text):
            # [idx : idx+max_chunk_size) の範囲を切り出す。
            # Pythonのスライスは範囲外でもエラーにならないため、
            # 末尾でも安全に扱える。
            chunk_slice = text[idx:idx + self.max_chunk_size]
            chunks.append({
                "file_path": file_path,
                "first_character_index": start_idx + idx,
                # inclusive（両端を含む）なインデックスにするため -1 する。
                # 例: 2000文字のチャンクなら [0, 1999] になる。
                "last_character_index": start_idx + idx + len(chunk_slice) - 1,
                "text": chunk_slice
            })
            idx += self.max_chunk_size

    def chunk_python_code(
        self, text: str, file_path: str
    ) -> List[Dict[str, Any]]:
        """
        Pythonソースコード用のチャンキング戦略。

        コードは「行」を境界にして意味のまとまりが崩れにくいため、
        改行単位で行を積み上げていき、max_chunk_size を超えそうに
        なったところで一区切りにする（行の途中では切らない）。
        こうすることで、関数やクラスの途中で不自然に分断される
        可能性を減らしつつ、チャンクサイズの上限も守る。

        Args:
            text: ファイル全体のテキスト
            file_path: このファイルのパス（保存用）

        Returns:
            {"file_path", "first_character_index",
             "last_character_index", "text"} を持つ辞書のリスト
        """
        chunks: List[Dict[str, Any]] = []
        lines = text.split('\n')
        current_chunk = ""
        # current_chunk の先頭がファイル全体の何文字目から始まるか
        start_idx = 0

        for i, line in enumerate(lines):
            # split('\n') で失われる改行文字を、最後の行以外には復元する
            line_text = line + '\n' if i < len(lines) - 1 else line
            line_len = len(line_text)

            # この行を足すと上限を超えてしまう場合は、
            # 現在たまっている current_chunk を先に確定させる
            if len(current_chunk) + line_len > self.max_chunk_size \
               and current_chunk:
                self._safe_append(
                    chunks, str(file_path), start_idx, current_chunk
                )
                # 次のチャンクの開始位置を更新してからリセット
                start_idx += len(current_chunk)
                current_chunk = ""

            current_chunk += line_text

        # ループを抜けた時点でまだ確定していない残り分を最後に追加する
        if current_chunk:
            self._safe_append(chunks, str(file_path), start_idx, current_chunk)

        return chunks

    def chunk_markdown_text(
        self, text: str, file_path: str
    ) -> List[Dict[str, Any]]:
        """
        Markdown / プレーンテキスト用のチャンキング戦略。

        ドキュメントは「段落（空行区切り）」が意味のまとまりの単位に
        なることが多いため、"\\n\\n" で段落に分割し、段落単位で
        積み上げていく。Pythonコード用のchunk_python_codeと構造は
        similar だが、分割の粒度が「行」ではなく「段落」である点が異なる。

        Args:
            text: ファイル全体のテキスト
            file_path: このファイルのパス（保存用）

        Returns:
            {"file_path", "first_character_index",
             "last_character_index", "text"} を持つ辞書のリスト
        """
        chunks: List[Dict[str, Any]] = []
        paragraphs = text.split('\n\n')
        current_chunk = ""
        start_idx = 0

        for i, para in enumerate(paragraphs):
            # split で失われる "\n\n" を、最後の段落以外には復元する
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
        コーパス全体をインデックス化するメインの処理。

        差分インデックス（Incremental Indexing）を実装しており、
        前回のインデックス作成時からのファイルの mtime（更新日時）を
        file_meta.json に保存しておくことで、2回目以降の実行では：
          - 新規追加されたファイル
          - 内容が更新されたファイル（mtimeが変化）
        だけをチャンク分割・埋め込み計算し直す。
        変更のないファイルは既存の chunks.json / embeddings.npy から
        そのまま引き継ぐため、コーパス全体を毎回舐め直すより高速。

        削除されたファイルのチャンクは最終結果から除外される。

        Raises:
            FileNotFoundError: self.corpus_dir が存在しない場合
        """
        if not self.corpus_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.corpus_dir}")

        # 出力先ディレクトリを用意（既にあってもエラーにしない）
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # インデックス対象とする拡張子。
        # コード(.py)とドキュメント(.md/.txt/.rst)の両方を扱う。
        target_extensions = {'.py', '.md', '.txt', '.rst'}

        chunks_path = self.processed_dir / "chunks.json"
        embeddings_path = self.processed_dir / "embeddings.npy"
        meta_path = self.processed_dir / "file_meta.json"

        # --- 既存データとメタデータのロード ---
        # 初回実行時はこれらのファイルがまだ存在しないため、
        # 空のリスト/辞書のままで問題なく動く。
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

        # --- 現在のファイル一覧と更新日時（mtime）を取得 ---
        # ここで「今このコーパスに存在するファイルとその更新時刻」を
        # 全部集め、file_meta（前回の記録）と突き合わせて
        # 「今回処理すべきファイル」を洗い出す。
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
                        # 新規ファイル、またはmtimeが変わった＝更新された
                        # ファイルなので再処理の対象にする
                        filepaths_to_process.append(file_path)

        # 前回はあったが今回は見当たらないファイル＝削除されたファイル
        deleted_files = set(file_meta.keys()) - set(current_files.keys())

        print(
            "Incremental Indexing: "
            f"{len(filepaths_to_process)} files to update, "
            f"{len(deleted_files)} files deleted."
        )

        # --- 変更・追加ファイルの処理とチャンク作成 ---
        # ファイルの種類ごとにチャンキング戦略を切り替える。
        new_chunks_by_file: Dict[str, List[Dict[str, Any]]] = {}
        # 埋め込み計算用の軽量モデル（CPUで動くもの）をロード。
        # 全体をロードするのはここだけで、以降は使い回す。
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
                    # .md / .txt / .rst はすべて段落ベースのチャンキングを使う
                    file_chunks = self.chunk_markdown_text(text, str_path)

                new_chunks_by_file[str_path] = file_chunks
            except Exception:
                # 文字コードの問題などで読めないファイルはスキップし、
                # インデックス作成全体を止めないようにする
                continue

        # --- 新規・変更ファイルの埋め込み計算 ---
        # ファイル単位でまとめてバッチエンコードすることで、
        # モデル呼び出し回数を減らし高速化する。
        new_embeddings_by_file: Dict[str, np.ndarray] = {}
        for str_path, chunks in new_chunks_by_file.items():
            if chunks:
                texts = [c["text"] for c in chunks]
                embs = embed_model.encode(
                    texts, batch_size=64, normalize_embeddings=True
                )
                new_embeddings_by_file[str_path] = embs

        # --- 既存データと新規データの結合・再構築 ---
        # 最終的な chunks / embeddings を1つのリストにまとめ直す。
        final_chunks: List[Dict[str, Any]] = []
        final_embeddings_list: List[np.ndarray] = []

        if (
            existing_chunks
            and existing_embeddings is not None
            and len(existing_chunks) == len(existing_embeddings)
        ):
            # 既存チャンクをファイルパスごとにグルーピングしておく。
            # こうすることで「削除されたファイル」「更新されたファイル」を
            # 除外しつつ、変更のなかったファイルのチャンクだけを
            # そのまま引き継げる。
            file_data: Dict[str, List[Tuple[Dict[str, Any], np.ndarray]]] = {}
            for i, chunk in enumerate(existing_chunks):
                fpath = chunk["file_path"]
                if fpath not in file_data:
                    file_data[fpath] = []
                file_data[fpath].append((chunk, existing_embeddings[i]))

            for fpath, pairs in file_data.items():
                if fpath in deleted_files or fpath in new_chunks_by_file:
                    # 削除された、または今回再計算されたファイルは
                    # 古いチャンクを引き継がない（二重登録を防ぐ）
                    continue
                for chunk, emb in pairs:
                    final_chunks.append(chunk)
                    final_embeddings_list.append(emb)

        # 新規・更新分を追加
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
            # チャンクが1件もない場合でも、埋め込み次元(384)だけ
            # 合わせた空配列を保存しておくことで、後続のロード処理が
            # shape不一致で落ちないようにする。
            final_embeddings = np.zeros((0, 384))

        # ディスクへ永続化
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
        現在保持しているチャンクデータを
        JSON ファイル (data/processed/chunks.json) としてディスクに保存する。
        """
        chunks_path = self.processed_dir / "chunks.json"
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

    def load_index(self) -> None:
        """
        保存されている chunks.json を読み込み、
        BM25検索エンジンを再構築する。

        主にIndexerを単体で使ってBM25の状態を再現したい場合に使う
        （通常の検索フローでは Retriever が同等の処理を持つ）。

        Raises:
            FileNotFoundError: chunks.json が見つからない場合
                （まだ index_corpus() が一度も実行されていない状態）
        """
        chunks_path = self.processed_dir / "chunks.json"
        if not chunks_path.exists():
            raise FileNotFoundError("Index not found. Run 'index' first.")

        with open(chunks_path, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
            self.chunks = cast(List[Dict[str, Any]], loaded_data)

        # BM25はトークン化されたコーパス（単語のリストのリスト）を必要とする。
        # 正規表現 \w+ で単語を抽出し、ストップワードを除去してから渡す。
        tokenized_corpus = [
            [
                w for w in re.findall(r'\w+', chunk["text"].lower())
                if w not in STOPWORDS
            ]
            for chunk in self.chunks
        ]
        self.bm25 = BM25Okapi(tokenized_corpus)
