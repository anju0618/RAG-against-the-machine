import uuid
from typing import List
from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    """
    検索で見つけてきたソーステキストの場所を記録するモデル。
    どのファイルの、何文字目から何文字目までを切り取ってきたかを定義
    """
    # コーパス内のファイルパス。評価時は一言一句一致している必要があります。
    # 例: "data/raw/vllm-0.10.1/docs/features/lora.md"
    file_path: str  
    first_character_index: int  # 抽出したテキストの開始位置（何文字目か）
    last_character_index: int   # 抽出したテキストの終了位置（何文字目か）


class UnansweredQuestion(BaseModel):
    """
    まだAIが回答していない、ユーザーからの「質問」を表すモデル。
    データセットから質問を読み込む際などに使います。
    """
    # 質問を一意に識別するためのID。
    # 指定されなければ自動でUUID（ランダムな一意の文字列）を生成します。
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str  # 質問のテキスト本体


class AnsweredQuestion(UnansweredQuestion):
    """
    AIが回答し終わった「質問と回答のセット」を表すモデル。
    UnansweredQuestionを継承しているため、question_idとquestionも含まれます
    """
    sources: List[MinimalSource]  # 回答の根拠として使ったソース（証拠）のリスト
    answer: str  # Qwen3-0.6BなどのAIが生成した回答のテキスト


class RagDataset(BaseModel):
    """
    複数の質問を束ねたデータセット全体を表すモデル。
    JSONファイルから質問集を丸ごと読み込むときの型として使います。
    """
    # 回答済みの質問、または未回答の質問のリスト
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
