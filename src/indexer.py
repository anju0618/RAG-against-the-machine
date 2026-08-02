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

--- 高速化・精度改善について（重要） ---
このファイルには、以下の2つの問題に対応するための実装が含まれている：

1. インデックス作成が遅い（5分ルールに抵触する）:
   - 埋め込み計算をファイルごとに何度も小さく呼び出すのではなく、
     今回処理する全ファイル分のテキストを1回にまとめてから
     SentenceTransformer.encode() を呼ぶことで、モデル呼び出しの
     オーバーヘッドを大幅に削減している（詳細は index_corpus 内のコメント参照）。
   - .git / __pycache__ / node_modules など、検索に無関係で
     かつ数が非常に多くなりがちなディレクトリを os.walk() の探索対象から
     あらかじめ除外し、無駄なファイルシステム走査を減らしている。
   - torch のスレッド数をCPUコア数に合わせて明示的に設定し、
     CPU環境でのバッチ推論を高速化している。

2. recall@kが基準を下回る（精度が足りない）:
   - BM25・埋め込みの両方で使うトークン化を、単純な単語抽出だけでなく
     snake_case / camelCase の識別子を分割するロジックに強化した
     （例: "openai_serving_chat" は "openai" "serving" "chat" としても
     マッチできるようになる）。これにより、質問がコードの識別子を
     そのまま引用する場合も、言い回しを変えている場合も拾いやすくなる。
   - 検索対象テキストに軽いファイル名ヒントを付加してからトークン化・
     埋め込みすることで、「ファイル名やモジュール名に関するキーワード」を
     含む質問への再現率(recall)を高めている
     （chunks.json に保存される "text" 自体は変更しない。
     このヒントはBM25/埋め込みの検索用表現にのみ使われる）。
   - Markdown（.md）ファイルのチャンキングを、単純な空行区切りの
     段落ベースから、見出し（"#" 〜 "######"）単位を優先する方式
     （chunk_markdown_by_headings）に変更した。ドキュメント系の質問は
     「特定の見出し（機能・設定項目）について」尋ねることが多いため、
     見出し単位でチャンクの内容を揃えた方が埋め込みベクトルが
     1つの話題に集中し、セマンティック検索の精度が上がりやすい。
   - 埋め込みモデルを all-MiniLM-L6-v2 から BAAI/bge-small-en-v1.5 に
     変更した。同程度の速度でCPU上でも実用的に動きつつ、検索タスク向けに
     チューニングされているため、一般的にMiniLMより高い検索精度を示す。
     クエリ側にのみ検索用の指示文を付与する（retriever.py参照）。
"""

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

# --- インデックス作成をスキップするディレクトリ名 -----------------------
# バージョン管理メタデータ、キャッシュ、仮想環境、依存パッケージなどは
# 検索対象として無意味なうえ、ファイル数が非常に多くなりがちで
# os.walk() の実行時間を無駄に伸ばす原因になる。
# os.walk() が返す dirs リストをその場で書き換えることで、
# これらのディレクトリ配下には一切降りていかないようにする。
SKIP_DIR_NAMES = {
    ".git", "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "node_modules", ".venv", "venv", "env", "dist", "build",
    ".idea", ".vscode", ".tox", "site-packages", ".eggs",
}

# 単語抽出用の正規表現。モジュールレベルで一度だけコンパイルしておくことで、
# 大量のチャンク・クエリに対して繰り返し呼び出す際の再解析コストを避ける。
_WORD_RE = re.compile(r"\w+")

# snake_case / camelCase の識別子を単語境界で分割するための正規表現。
# 例: "AsyncLLMEngine" -> "Async_LLM_Engine" -> ["Async", "LLM", "Engine"]
#     "openai_serving_chat" は元々アンダースコアがあるのでそのまま分割対象になる
_CAMEL_BOUNDARY_1 = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_CAMEL_BOUNDARY_2 = re.compile(r"(?<=[A-Z])(?=[A-Z][a-z])")

# 埋め込みモデル名。indexer.py（インデックス作成時）と retriever.py
# （検索時のクエリ埋め込み）の両方でこの定数を共有することで、
# 2つのモジュールが異なるモデルの埋め込みベクトルを混同して比較してしまう
# 事故を防いでいる。
#
# all-MiniLM-L6-v2 から BAAI/bge-small-en-v1.5 に変更した。どちらもCPUで
# 実用的な速度で動く軽量モデル（パラメータ数は同程度）だが、BGE系のモデルは
# 検索（retrieval）タスク向けに追加学習されており、一般的なベンチマークで
# MiniLMよりも高い検索精度を示すことが多い。出力次元は384で一致しているため、
# embeddings.npy 側の形状（shape）変更は不要。
#
# 注意: BGEモデルは「クエリ側にだけ」検索用の指示文（instruction）を
# 付与することを推奨している（パッセージ/チャンク側には付与しない）。
# この指示文は retriever.py の QUERY_INSTRUCTION_PREFIX で付与される。
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# 見出し（ATX形式の "#" 〜 "######"）を検出するための正規表現。
# Markdownのチャンキングを「空行区切りの段落」だけでなく
# 「見出しセクション」単位でも行えるようにするために使う
# （chunk_markdown_by_headings 参照）。
_MARKDOWN_HEADING_RE = re.compile(r'(?m)^#{1,6} .*$')


def _split_identifier(word: str) -> List[str]:
    """
    識別子らしき単語を、snake_case のアンダースコアと
    camelCase の大文字境界の両方で分割する。

    通常の英単語（"server" など）を渡した場合は分割が発生せず
    そのまま1要素のリストが返るため、副作用なく安全に使える。

    Args:
        word: 分割対象の単語（識別子でなくても良い）

    Returns:
        分割済みの部分文字列のリスト（空文字は含まれない）
    """
    # 小文字/数字の直後に大文字が来る境界にアンダースコアを挿入する
    # （例: "asyncLLM" の "c" と "L" の間）
    s = _CAMEL_BOUNDARY_1.sub("_", word)
    # 大文字の連続（略語）の直後に「大文字+小文字」の単語が続く境界にも
    # アンダースコアを挿入する（例: "LLMEngine" -> "LLM_Engine"）
    s = _CAMEL_BOUNDARY_2.sub("_", s)
    return [p for p in s.split("_") if p]


def tokenize_text(text: str) -> List[str]:
    """
    BM25・検索クエリの両方で共有するトークン化関数。

    単純な単語抽出（正規表現 \\w+）に加えて、snake_case / camelCase の
    識別子をサブワードに分割したトークンも追加することで、
    「質問がコードの識別子をそのまま引用している場合」だけでなく
    「質問が識別子を単語に分解して言い換えている場合」もヒットしやすくする。

    例:
        "AsyncLLMEngine" というコード上の識別子に対して、
        - "AsyncLLMEngine" というそのままの質問
        - "async llm engine" のように分解した言い回しの質問
        の両方が、同じチャンクにマッチできるようになる。

    Args:
        text: トークン化したい元のテキスト（チャンク本文 or 検索クエリ）

    Returns:
        ストップワードを除去した後のトークンのリスト
        （小文字化済み。重複除去はしない＝BM25の頻度計算に任せる）
    """
    tokens: List[str] = []
    # 大文字/小文字の情報が必要なため、lower() する前に単語を抽出する
    for raw in _WORD_RE.findall(text):
        lower_whole = raw.lower()
        if lower_whole not in STOPWORDS:
            tokens.append(lower_whole)

        # 識別子らしく分割できる場合のみサブトークンを追加する。
        # 通常の英単語は1要素のまま返ってくるため、ここでの追加が
        # 二重登録にならないようチェックしている。
        parts = _split_identifier(raw)
        if len(parts) > 1:
            for part in parts:
                lower_part = part.lower()
                if lower_part and lower_part not in STOPWORDS:
                    tokens.append(lower_part)

    return tokens


def build_search_text(file_path: str, text: str) -> str:
    """
    BM25のトークン化とセマンティック埋め込みの計算「にのみ」使う、
    検索用のテキスト表現を組み立てる。

    ファイル名（拡張子を除いたもの）を軽いヒントとして先頭に付与することで、
    「このファイル・モジュールについて」という形で質問された場合にも
    ヒットしやすくする（例: ファイル名が "lora.md" のチャンクは、
    "LoRA" というキーワードを含む質問に少し有利になる）。

    重要: この関数の戻り値は chunks.json に保存される "text" フィールド
    （= 実際にmoulinette/Generatorが参照する原文そのもの）には一切影響しない。
    あくまでBM25トークン化・埋め込み計算という「検索のためだけの表現」に
    使われる一時的な文字列である。

    Args:
        file_path: チャンクの元になったファイルのパス
        text: チャンク本文（原文そのまま）

    Returns:
        ファイル名ヒントを付与した検索用テキスト
    """
    name_hint = Path(file_path).stem.replace("_", " ").replace("-", " ")
    return f"{name_hint}\n{text}"


class Indexer:
    """
    BM25Okapi（キーワード検索）と SentenceTransformer（埋め込みベクトル）を
    組み合わせてコーパスをインデックス化するクラス。

    差分インデックス（Incremental Indexing）に対応しているため、
    2回目以降の index_corpus() 実行では変更されたファイルのみを
    再チャンク・再埋め込みし、既存のチャンク/埋め込みは使い回す。
    """

    def __init__(
            self,
            max_chunk_size: int = 2000,
            skip_vector: bool = False
            ) -> None:
        """
        Indexerを初期化する。

        Args:
            max_chunk_size: 1チャンクあたりの最大文字数（デフォルト: 2000）。
                仕様書の制約により、moulinette側の max_context_length と
                同じ 2000 文字を超えるチャンクを作ってはいけない
                （超過すると評価結果が丸ごと無効になる）。
                なお、チャンクサイズを大きく保つほど総チャンク数が減り、
                埋め込み計算量も減るため、精度が許す限り上限値
                （2000文字）を使うのがインデックス作成速度の観点でも有利。
            skip_vector: Trueの場合、Vector（埋め込み）計算をスキップする。
                必須パート（Lexicalのみ）の評価時に5分ルールを安全に満たすためのフラグ。
        """
        self.max_chunk_size = max_chunk_size
        self.skip_vector = skip_vector
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
        chunks リストへ追加する安全弁

        chunk_python_code / chunk_markdown_text は「行」や「段落」を
        単位に積み上げていくが、1行・1段落自体が max_chunk_size を
        超える場合（長いコード行や長大な段落）がありうる
        このメソッドはそのケースでも必ず max_chunk_size 以下の
        チャンクだけが出力されるように、文字単位でスライスし直す

        Args:
            chunks: 追加先のチャンクリスト（呼び出し元が保持するリストを
                そのまま書き換える＝副作用あり）
            file_path: 対象ファイルのパス (str)
            start_idx: ファイル全体における、このtext断片の開始文字位置
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

    def chunk_markdown_by_headings(
        self, text: str, file_path: str
    ) -> List[Dict[str, Any]]:
        """
        Markdown（.md）専用の、見出し（ATX heading, "#" 〜 "######"）を
        優先的な境界とするチャンキング戦略。

        なぜ必要か:
        chunk_markdown_text() は単純に空行（段落）で分割するため、
        「見出し行」とその直後の説明文がたまたま別の"段落"として
        バラバラに扱われたり、逆に全く無関係な複数のセクションが
        1つのチャンクに混在してしまうことがある。ドキュメントの
        質問は多くの場合「特定の見出し（機能・設定項目）について」
        尋ねるものなので、見出し単位でチャンクを揃えた方が、
        埋め込みベクトルが1つの話題に集中し、意味的な検索
        （semantic search）の精度が上がりやすい。

        アルゴリズム:
        1. 正規表現でMarkdown見出し行の開始位置を全て見つける
        2. その位置を境界として、テキスト全体を「見出し＋本文」の
           セクション単位に分割する（見出しが1つもない場合は
           chunk_markdown_text() にフォールバックする）
        3. 各セクションを、chunk_python_code/chunk_markdown_text と
           同じ「max_chunk_sizeを超えそうになったら確定する」
           積み上げロジックで結合し、最終的に _safe_append で
           2000文字の上限を必ず守る

        Args:
            text: ファイル全体のテキスト
            file_path: このファイルのパス（保存用）

        Returns:
            {"file_path", "first_character_index",
             "last_character_index", "text"} を持つ辞書のリスト
        """
        heading_starts = [
            m.start() for m in _MARKDOWN_HEADING_RE.finditer(text)
        ]

        if not heading_starts:
            return self.chunk_markdown_text(text, file_path)

        # テキスト全体を、見出し開始位置を境界とした連続区間
        # （隙間なく [0, len(text)) をちょうど覆う区間）のリストに変換する。
        boundaries = sorted(set([0] + heading_starts + [len(text)]))
        sections: List[Tuple[int, str]] = []
        for i in range(len(boundaries) - 1):
            s, e = boundaries[i], boundaries[i + 1]
            if e > s:
                sections.append((s, text[s:e]))

        chunks: List[Dict[str, Any]] = []
        current_chunk = ""
        # current_chunk の先頭がファイル全体の何文字目から始まるか。
        # セクションは常に隙間なく連続しているため、複数セクションを
        # またいで文字列連結しても、実ファイル内の位置とずれることはない。
        current_start = 0

        for section_start, section_text in sections:
            if not current_chunk:
                # 新しいチャンクの先頭を、このセクションの実際の開始位置に合わせる
                current_start = section_start

            section_len = len(section_text)
            if (
                current_chunk
                and len(current_chunk) + section_len > self.max_chunk_size
            ):
                self._safe_append(
                    chunks, str(file_path), current_start, current_chunk
                )
                current_chunk = ""
                current_start = section_start

            current_chunk += section_text

        if current_chunk:
            self._safe_append(
                chunks, str(file_path), current_start, current_chunk
            )

        return chunks

    def _walk_corpus_files(self, target_extensions: set[str]) -> List[Path]:
        """
        コーパスディレクトリを走査し、対象拡張子を持つファイルのパスの
        リストを返す。

        SKIP_DIR_NAMES に含まれるディレクトリ（.git, node_modules,
        仮想環境フォルダ等）には一切降りていかない。os.walk() が返す
        dirs リストをその場で（in-place で）フィルタリングすることで、
        os.walk 自身がそのサブディレクトリを再帰的に辿らないようにする
        仕組みを利用している。これにより、検索に無関係な大量のファイルを
        スキャンする無駄な時間を削減できる（インデックス作成速度の改善）。

        Args:
            target_extensions: 対象とするファイル拡張子の集合

        Returns:
            条件に合致するファイルパスのリスト
        """
        matched_files: List[Path] = []
        for root, dirs, files in os.walk(self.corpus_dir):
            # dirs をその場で書き換えることで、os.walk に
            # 「このディレクトリ配下には降りない」ことを伝える。
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

        速度面の工夫（5分ルール対策）:
          - .git 等の無関係なディレクトリを走査しない
            （_walk_corpus_files 参照）
          - 埋め込み計算を「ファイルごとに何度も」ではなく
            「今回処理する全ファイル分をまとめて1回」呼び出す
            （SentenceTransformer.encode() の呼び出しオーバーヘッドは
            バッチが小さいほど相対的に大きくなるため、まとめることで
            大幅に高速化できる）
          - torch のスレッド数をCPUコア数に合わせて設定し、
            CPU上でのバッチ推論の並列度を上げる

        Raises:
            FileNotFoundError: self.corpus_dir が存在しない場合
        """
        start_time = time.time()

        if not self.corpus_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.corpus_dir}")

        # CPU環境でのバッチ推論を高速化するため、利用可能な全コアを
        # torchに使わせる。GPUがある環境では影響しない設定なので、
        # 常に呼び出しておいて問題ない。
        cpu_count = os.cpu_count()
        if cpu_count:
            torch.set_num_threads(cpu_count)

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
        # 無関係なディレクトリ（.git等）は _walk_corpus_files が
        # あらかじめ除外してくれているため、ここでは対象拡張子の
        # ファイルだけを効率的に走査できる。
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
                    # Markdownは見出し単位のチャンキングを優先する
                    # （chunk_markdown_by_headings参照。見出しが
                    # 見つからない場合は内部で段落ベースにフォールバックする）
                    file_chunks = self.chunk_markdown_by_headings(
                        text, str_path
                    )
                else:
                    # .txt / .rst はMarkdownの見出し記法(#)を前提にできない
                    # ため、従来通り段落ベースのチャンキングを使う
                    file_chunks = self.chunk_markdown_text(text, str_path)

                new_chunks_by_file[str_path] = file_chunks
            except Exception:
                # 文字コードの問題などで読めないファイルはスキップし、
                # インデックス作成全体を止めないようにする
                continue

        chunking_end_time = time.time()
        lexical_time = chunking_end_time - start_time

        # --- 新規・変更ファイルの埋め込み計算 ---
        # ここが以前の実装で最も遅かった箇所。
        # 「ファイルごとに小さいバッチでencode()を呼ぶ」のではなく、
        # 今回処理する全ファイル・全チャンクのテキストを1つの巨大な
        # リストにまとめてから、1回（内部的には指定したbatch_sizeで
        # 自動分割されるが、呼び出し自体は1回）だけ encode() を呼ぶ。
        # モデルのフォワードパス自体はバッチが大きいほど効率的なうえ、
        # Python関数呼び出し自体のオーバーヘッドも「ファイル数分」から
        # 「1回」に削減されるため、ファイル数が多いコーパスほど
        # 高速化の効果が大きい。
        new_embeddings_by_file: Dict[str, np.ndarray] = {}

        # (ファイルパス, このファイルの開始位置, 終了位置) を記録しておき、
        # 一括で計算した埋め込み配列を後でファイルごとに切り分けられるようにする
        file_slices: List[Tuple[str, int, int]] = []
        all_search_texts: List[str] = []

        for str_path, chunks in new_chunks_by_file.items():
            if not chunks:
                continue
            start = len(all_search_texts)
            # 埋め込み計算にはファイル名ヒントを付与した検索用テキストを使う
            # （chunks.jsonに保存される "text" 自体は変更しない）
            all_search_texts.extend(
                build_search_text(str_path, c["text"]) for c in chunks
            )
            file_slices.append((str_path, start, len(all_search_texts)))

        vector_time = 0.0
        if not self.skip_vector and all_search_texts:
            print("Computing semantic embeddings (Vector)...")
            embed_start_time = time.time()

            # 埋め込み計算用の軽量モデル（CPUで動くもの）をロード。
            # 実際に埋め込みが必要な場合のみロードすることで、
            # 変更ファイルが0件の再実行時にモデルロードコストすら
            # 発生しないようにしている。
            embed_model = SentenceTransformer(
                EMBEDDING_MODEL_NAME, device="cpu"
            )
            all_embeddings = embed_model.encode(
                all_search_texts,
                batch_size=128,
                normalize_embeddings=True,
                show_progress_bar=True,
            )
            for str_path, start, end in file_slices:
                new_embeddings_by_file[str_path] = all_embeddings[start:end]

            vector_time = time.time() - embed_start_time
        elif self.skip_vector and all_search_texts:
            print("Skipping semantic embeddings (Vector) computation...")

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
            elif chunks and self.skip_vector:
                # Vectorをスキップした場合、既存の引き継ぎ分の長さに合わせて0埋めするか、
                # または新しいチャンクだけ追加する（ロード時にフォールバックするため問題ない）
                for i, chunk in enumerate(chunks):
                    final_chunks.append(chunk)

        self.chunks = final_chunks
        if final_embeddings_list and not self.skip_vector:
            final_embeddings = np.array(final_embeddings_list)
        else:
            # チャンクが1件もない場合やVectorスキップ時でも、
            # 埋め込み次元(384)だけ合わせた空配列を保存しておくことで、
            # 後続のロード処理がshape不一致で落ちないようにする。
            final_embeddings = np.zeros((0, 384))

        # ディスクへ永続化
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
        if not self.skip_vector:
            print(f"  - Vector (Embedding): {vector_time:.2f} seconds")
        print(f"  - Total Time: {total_time:.2f} seconds\n")

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
        # tokenize_text() は識別子のsnake_case/camelCase分割にも対応した
        # 共有トークナイザで、Retriever側の検索クエリのトークン化とも
        # 一致させることで、質問とチャンクの語彙のずれを減らしている。
        # また、ファイル名ヒントを付与した検索用テキストに対して
        # トークン化することで、ファイル名に関するキーワードも拾える。
        tokenized_corpus = [
            tokenize_text(build_search_text(chunk["file_path"], chunk["text"]))
            for chunk in self.chunks
        ]
        self.bm25 = BM25Okapi(tokenized_corpus)
