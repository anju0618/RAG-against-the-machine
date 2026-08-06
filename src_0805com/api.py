"""
api.py
======

ボーナス「Local HTTP API」の実装。
CLIと同じ Retriever / Generator を使い回しつつ、FastAPI経由で
HTTP越しに検索・回答生成を叩けるようにする薄いラッパー。

起動例:
    uv run uvicorn src.api:app --reload

エンドポイント:
    POST /search  -> 検索のみ実行し、上位k件のソースを返す
    POST /answer  -> 検索 + LLMによる回答生成までを実行して返す
"""

from typing import Any, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.generator import Generator
from src.models import MinimalSearchResults
from src.retriever import Retriever

app = FastAPI(title="RAG Local API")

# Retriever / Generator はどちらも初期化コストが高い
# （インデックスのロード、LLMモデルのロード）ため、
# アプリ起動時に一度だけインスタンス化し、以降の全リクエストで使い回す。
retriever = Retriever()
generator = Generator()


class SearchRequest(BaseModel):
    """POST /search のリクエストボディ。"""
    query: str
    k: int = 5


class AnswerRequest(BaseModel):
    """POST /answer のリクエストボディ。"""
    query: str
    k: int = 5


@app.post("/search")
def api_search(
    req: SearchRequest,
) -> Dict[str, Any]:
    """
    指定されたクエリに対してハイブリッド検索を実行し、上位 k 件のソースを返す。

    Args:
        req: query（検索文字列）と k（取得件数）を含むリクエストボディ

    Returns:
        {"query": ..., "k": ..., "sources": [...]} 形式の辞書

    Raises:
        HTTPException: 検索処理中に例外が発生した場合は500エラーとして返す
            （不正なクエリでサーバー全体を落とさないためのガード）
    """
    try:
        sources = retriever.search(req.query, req.k)
        return {"query": req.query, "k": req.k, "sources": sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/answer")
def api_answer(
    req: AnswerRequest,
) -> Dict[str, Any]:
    """
    指定されたクエリに対して検索を行い、LLMによる回答生成までを実行して返す。

    Args:
        req: query（検索文字列）と k（取得件数）を含むリクエストボディ

    Returns:
        {"query": ..., "k": ..., "sources": [...], "answer": ...} 形式の辞書

    Raises:
        HTTPException: 検索・生成処理中に例外が発生した場合は500エラーとして返す
    """
    try:
        sources = retriever.search(req.query, req.k)
        search_result = MinimalSearchResults(
            question_id="api-query",
            question=req.query,
            retrieved_sources=sources
        )
        # generator は起動時に一度だけロードした共有インスタンス
        # （デフォルト k=5）だが、ここでは req.k を明示的に渡すことで、
        # リクエストごとに異なる k が指定されてもプロンプトに詰め込む
        # ソース数がその k と一致するようにしている。
        # （k を渡し忘れると、常に起動時のデフォルト件数だけが
        #  コンテキストに使われてしまうバグになる）
        answers = generator.generate_answers([search_result], k=req.k)
        return {
            "query": req.query,
            "k": req.k,
            "sources": sources,
            "answer": answers[0]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
