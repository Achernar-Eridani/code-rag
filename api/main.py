# -*- coding: utf-8 -*-
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import pathlib, sys

ROOT = pathlib.Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from retriever.hybrid_search import HybridSearcher  # noqa

app = FastAPI(title="Code-RAG MVP", version="0.2.0")

_searcher: Optional[HybridSearcher] = None
def get_searcher() -> HybridSearcher:
    global _searcher
    if _searcher is None:
        _searcher = HybridSearcher()
    return _searcher

@app.get("/ping")
def ping():
    return {"ok": True}

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = 10
    symbol_boost: float = 2.0

class SearchResult(BaseModel):
    id: str
    score: float
    name: str = ""
    kind: str = ""
    path: str = ""
    start_line: int = 0
    end_line: int = 0
    text_preview: str = ""

class SearchResponse(BaseModel):
    query: str
    total: int
    results: List[SearchResult]

@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    try:
        hs = get_searcher()
        rs = hs.search(req.query, top_k=req.top_k, symbol_boost=req.symbol_boost)
        out: List[SearchResult] = []
        for r in rs:
            m = r["metadata"]
            out.append(
                SearchResult(
                    id=r["id"],
                    score=float(r["score"]),
                    name=str(m.get("name", "")),
                    kind=str(m.get("kind", "")),
                    path=str(m.get("path", "")),
                    start_line=int(m.get("start_line", 0) or 0),
                    end_line=int(m.get("end_line", 0) or 0),
                    text_preview=str(r.get("text_preview", "")),
                )
            )
        return SearchResponse(query=req.query, total=len(out), results=out)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"search error: {e}")

@app.get("/search/{symbol}")
def search_symbol(symbol: str, top_k: int = 10):
    try:
        hs = get_searcher()
        rows = hs.search_by_symbol(symbol, top_k=top_k)
        return {"symbol": symbol, "total": len(rows), "results": rows}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"search_by_symbol error: {e}")
