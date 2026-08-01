import uuid
from typing import List
from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    """
    {
        "file_path": "data/raw/vllm-0.10.1/docs/serving/openai~~.md",
        "first_character_index": 9867,
        "last_character_index": 10100
    }

    検索してきたソース（根拠となるテキストの場所）の情報を表す最小単位のモデル
    評価システム（Moulinette）は、このファイルパスと文字インデックスの範囲を見て正解かどうかを判定
    """
    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """
    {
        "question_id": "uuid-string-1234",
        "question": "How to configure OpenAI server?"
    }

    ユーザーからの質問データを表すモデル
    """
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """
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

    質問に加えて、正解のソース（sources）と模範解答（answer）がセットになったモデル
    """
    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """
    {
        "rag_questions": [
            {"question_id": "...", "question": "..."},
            ...
        ]
    }

    評価用データセットのJSONファイル全体構造を表現するモデル
    """
    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """
    {
        "question_id": "q1",
        "question": "How to configure OpenAI server?",
        "retrieved_sources": [
            {"file_path": "...",
             "first_character_index": 0,
             "last_character_index": 50}
        ]
    }

    1つの質問に対して、検索エンジンが導き出した上位 k 件の検索結果を格納するモデル
    """
    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """
    MinimalSearchResults の全フィールドに加え、以下のフィールドを持つ：
    {
        "answer": "Generated answer text..."
    }

    検索結果に加えて、LLMが生成した自然言語の回答（answer）を保持するモデル
    """
    answer: str


class StudentSearchResults(BaseModel):
    """
    {
        "search_results": [...],
        "k": 10
    }

    検索コマンド（search_dataset）がファイルとして出力する最終的なJSONフォーマット
    """
    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """
    {
        "search_results": [...],
        "k": 10
    }

    回答生成コマンド（answer_dataset）がファイルとして出力する最終的なJSONフォーマット
    """
    search_results: List[MinimalAnswer]
    k: int
