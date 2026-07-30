import uuid
from typing import List
from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    """
    keep the path of source text
    whitch file and index
    """
    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """
    keep Unanswered Question.
    """
    # unique ID of Question.
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str  # question text


class AnsweredQuestion(UnansweredQuestion):
    """
    Model keeping set of Question and Ans
    """
    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """
    Model keep data set of Questions
    use when read from JSON file
    """
    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """
    検索（Retrieval）フェーズの結果を表すモデル。
    ある質問に対して、AIに見せるためのどのソースを引っ張ってきたかを記録します。
    """
    question_id: str  # どの質問に対する検索結果か
    question: str     # 質問のテキスト本体
    retrieved_sources: List[MinimalSource]  # 検索して見つけてきたソースのリスト


class MinimalAnswer(MinimalSearchResults):
    """
    生成（Generation）フェーズの結果を表すモデル。
    検索結果（MinimalSearchResults）に加えて、最終的にAIが生成した回答を保持
    """
    answer: str  # 抽出したソースをもとにAIが生成した回答


class StudentSearchResults(BaseModel):
    """
    Moulinetteに「検索結果」として提出するための最終フォーマット。
    CLIで `search_dataset` コマンドを実行したときに出力するJSONの型
    """
    search_results: List[MinimalSearchResults]  # 全質問に対する検索結果のリスト
    k: int  # 質問1つにつき、最大で何件のソースを検索したか（例: トップ10件なら 10）


class StudentSearchResultsAndAnswer(BaseModel):
    """
    Moulinetteに「検索結果＋回答」として提出するための最終フォーマット。
    CLIで `answer_dataset` コマンドを実行したときに出力するJSONの型
    """
    search_results: List[MinimalAnswer]  # 全質問に対する検索結果＋回答のリスト
    k: int  # 検索件数上限
