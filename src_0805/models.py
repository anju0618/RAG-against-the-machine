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
                 "last_character_index": 100}
            ],
            "answer": "To configure..."
        }
    """
    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """
    評価用データセットのJSONファイル全体構造を表現するモデル。
        {
            "rag_questions": [
                {"question_id": "...", "question": "..."},
                ...
            ]
        }
    """
    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """
    1つの質問に対して、検索エンジンが導き出した上位 k 件の検索結果を
    格納するモデル
        {
            "question_id": "q1",
            "question": "How to configure OpenAI server?",
            "retrieved_sources": [
                {"file_path": "...",
                 "first_character_index": 0,
                 "last_character_index": 50}
            ]
        }
    """
    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """
    LLMが生成した自然言語の回答を保持するモデル
        {
            "answer": "Generated answer text..."
        }
    """
    answer: str


class StudentSearchResults(BaseModel):
    """
    検索コマンド（search_dataset）がファイルとして出力する
        {
            "search_results": [...],
            "k": 10
        }
    """
    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """
    回答生成コマンド（answer_dataset）がファイルとして出力する
    最終的なJSONフォーマット

    例:
        {
            "search_results": [...],
            "k": 10
        }
    """
    search_results: List[MinimalAnswer]
    k: int
