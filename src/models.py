import uuid
from typing import List
from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    """
    【データ構造の形・例】
    {
        "file_path": "data/raw/vllm-0.10.1/docs/serving/openai~~.md",
        "first_character_index": 9867,
        "last_character_index": 10100
    }

    【役割】
    検索してきたソース（根拠となるテキストの場所）の情報を表す最小単位のモデルです。
    評価システム（Moulinette）は、このファイルパスと文字インデックスの範囲を見て正解かどうかを判定します。
    """
    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """
    【データ構造の形・例】
    {
        "question_id": "uuid-string-1234",
        "question": "How to configure OpenAI server?"
    }

    【役割】
    ユーザーからの質問データを表すモデルです。
    """
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """
    【データ構造の形・例】
    {
        "question_id": "uuid-string-1234",
        "question": "How to configure OpenAI server?",
        "sources": [
        {"file_path": "...",
        "first_character_index": 0,
        "last_character_index": 100
        }],
        "answer": "To configure..."
    }

    【役割】
    質問に加えて、正解のソース（sources）と模範解答（answer）がセットになったモデルです。
    """
    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """
    【データ構造の形・例】
    {
        "rag_questions": [
            {"question_id": "...", "question": "..."},
            ...
        ]
    }

    【役割】
    評価用データセットのJSONファイル全体構造を表現するモデルです。
    """
    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """
    【データ構造の形・例】
    {
        "question_id": "q1",
        "question": "How to configure OpenAI server?",
        "retrieved_sources": [
            {"file_path": "...",
             "first_character_index": 0,
             "last_character_index": 50}
        ]
    }

    【役割】
    1つの質問に対して、検索エンジンが導き出した上位 k 件の検索結果を格納するモデルです。
    """
    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """
    【データ構造の形・例】
    MinimalSearchResults の全フィールドに加え、以下のフィールドを持つ：
    {
        "answer": "Generated answer text..."
    }

    【役割】
    検索結果に加えて、LLMが生成した自然言語の回答（answer）を保持するモデルです。
    """
    answer: str


class StudentSearchResults(BaseModel):
    """
    【データ構造の形・例】
    {
        "search_results": [...],
        "k": 10
    }

    【役割】
    検索コマンド（search_dataset）がファイルとして出力する最終的なJSONフォーマットです。
    """
    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """
    【データ構造の形・例】
    {
        "search_results": [...],
        "k": 10
    }

    【役割】
    回答生成コマンド（answer_dataset）がファイルとして出力する最終的なJSONフォーマットです。
    """
    search_results: List[MinimalAnswer]
    k: int
