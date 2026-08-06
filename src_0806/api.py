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
    """POST /search のリクエストボディ"""
    query: str
    k: int = 5


class AnswerRequest(BaseModel):
    """POST /answer のリクエストボディ"""
    query: str
    k: int = 5


@app.post("/search")
def api_search(
    req: SearchRequest,
) -> Dict[str, Any]:

    try:
        sources = retriever.search(req.query, req.k)
        return {"query": req.query, "k": req.k, "sources": sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/answer")
def api_answer(
    req: AnswerRequest,
) -> Dict[str, Any]:

    try:
        sources = retriever.search(req.query, req.k)
        search_result = MinimalSearchResults(
            question_id="api-query",
            question=req.query,
            retrieved_sources=sources
        )
        answers = generator.generate_answers([search_result], k=req.k)
        return {
            "query": req.query,
            "k": req.k,
            "sources": sources,
            "answer": answers[0]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
