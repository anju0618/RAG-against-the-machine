from typing import Any, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.generator import Generator
from src.models import MinimalSearchResults
from src.retriever import Retriever

app = FastAPI(title="RAG Local API")

retriever = Retriever()
generator = Generator()


class SearchRequest(BaseModel):
    query: str
    k: int = 5


class AnswerRequest(BaseModel):
    query: str
    k: int = 5


@app.post("/search")
def api_search(
    req: SearchRequest,
) -> Dict[str, Any]:
    """
    指定されたクエリに対してハイブリッド検索を実行し、上位 k 件のソースを返す
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
    指定されたクエリに対して検索を行い、LLMによる回答生成までを実行して返す
    """
    try:
        sources = retriever.search(req.query, req.k)
        search_result = MinimalSearchResults(
            question_id="api-query",
            question=req.query,
            retrieved_sources=sources
        )
        
        # 【修正】 req.k を generate_answers に明示的に渡すことで、
        # Generatorインスタンス初期化時のデフォルト k=5 を上書きし、
        # APIリクエストで指定された件数分のソースをプロンプトに埋め込むように修正。
        answers = generator.generate_answers([search_result], k=req.k)
        
        return {
            "query": req.query,
            "k": req.k,
            "sources": sources,
            "answer": answers[0]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
