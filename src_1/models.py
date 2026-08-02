"""
models.py
=========

パイプラインの各ステージ（Indexer -> Retriever -> Generator -> API/CLI）の
間でやり取りされるデータ構造を pydantic の BaseModel として定義するモジュール。

仕様書で「データモデルは pydantic を使うこと」と明記されている通り、
ここで定義されたモデルはmoulinette（評価スクリプト）が読み込むJSONの
構造と完全に一致している必要がある。フィールド名・型を勝手に変更すると
評価に失敗するため、拡張する場合は既存フィールドを壊さず追加のみ行うこと。
"""

import uuid
from typing import List
from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    """
    検索によって見つかった「ソースの場所」を表す最小単位のモデル。

    テキストの中身そのものは持たず、「どのファイルの、
    何文字目から何文字目までか」という位置情報だけを持つ。
    評価システム（moulinette）はこのファイルパスと文字インデックスの
    範囲を見て、正解データとどれだけ重なっているか（recall@k）を判定する。

    例:
        {
            "file_path": "data/raw/vllm-0.10.1/docs/serving/openai~~.md",
            "first_character_index": 9867,
            "last_character_index": 10100
        }

    Attributes:
        file_path: ソースファイルのパス。検索対象コーパスのパスと
            完全一致している必要がある（末尾の差異等も含め厳密比較される）。
        first_character_index: 該当範囲の開始文字位置（0始まり）
        last_character_index: 該当範囲の終了文字位置（inclusive、
            つまりこの位置の文字も範囲に含まれる）
    """
    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """
    ユーザーからの質問データを表すモデル（まだ正解データを持たない状態）。

    例:
        {
            "question_id": "uuid-string-1234",
            "question": "How to configure OpenAI server?"
        }

    Attributes:
        question_id: 質問を一意に識別するID。
            指定がなければUUIDが自動生成される。
        question: 質問文そのもの
    """
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """
    質問に加えて、正解のソース（sources）と模範解答（answer）が
    セットになったモデル。評価用データセット（ground truth）で使われる。

    例:
        {
            "question_id": "uuid-string-1234",
            "question": "How to configure OpenAI server?",
            "sources": [
                {"file_path": "...",
                 "first_character_index": 0,
                 "last_character_index": 100}
            ],
            "answer": "To configure..."
        }

    Attributes:
        sources: この質問に対する正解のソース位置のリスト
        answer: この質問に対する模範解答
    """
    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """
    評価用データセットのJSONファイル全体構造を表現するモデル。

    rag_questions は AnsweredQuestion（正解付き）と
    UnansweredQuestion（正解なし、質問のみ）のどちらも許容する。

    例:
        {
            "rag_questions": [
                {"question_id": "...", "question": "..."},
                ...
            ]
        }

    Attributes:
        rag_questions: 質問オブジェクトのリスト
            （AnsweredQuestion または UnansweredQuestion のいずれか）
    """
    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """
    1つの質問に対して、検索エンジンが導き出した上位 k 件の検索結果を
    格納するモデル。search / search_dataset コマンドの出力の構成要素。

    例:
        {
            "question_id": "q1",
            "question": "How to configure OpenAI server?",
            "retrieved_sources": [
                {"file_path": "...",
                 "first_character_index": 0,
                 "last_character_index": 50}
            ]
        }

    Attributes:
        question_id: 質問のID
        question: 質問文
        retrieved_sources: 検索によって取得されたソースのリスト
            （関連度が高い順に並んでいることが望ましい）
    """
    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """
    MinimalSearchResults の全フィールドに加え、LLMが生成した
    自然言語の回答（answer）を保持するモデル。
    answer / answer_dataset コマンドの出力の構成要素。

    例:
        {
            "answer": "Generated answer text..."
        }

    Attributes:
        answer: LLMによって生成された回答文字列
    """
    answer: str


class StudentSearchResults(BaseModel):
    """
    検索コマンド（search_dataset）がファイルとして出力する
    最終的なJSONフォーマット。

    例:
        {
            "search_results": [...],
            "k": 10
        }

    Attributes:
        search_results: 各質問に対する MinimalSearchResults のリスト
        k: 検索時に指定した上位件数
    """
    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """
    回答生成コマンド（answer_dataset）がファイルとして出力する
    最終的なJSONフォーマット。

    例:
        {
            "search_results": [...],
            "k": 10
        }

    Attributes:
        search_results: 各質問に対する MinimalAnswer（回答つき）のリスト
        k: 検索・回答生成時に指定した上位件数
    """
    search_results: List[MinimalAnswer]
    k: int
