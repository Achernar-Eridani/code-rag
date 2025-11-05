# -*- coding: utf-8 -*-
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pathlib, sys, time, textwrap, os
from dotenv import load_dotenv
import re, unicodedata

load_dotenv()
_SURR_RE = re.compile(r'[\ud800-\udfff]')  # 代理区
_CTRL_RE = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F]')  # 可见之外的控制符（保留 \t\n\r）

ROOT = pathlib.Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from retriever.hybrid_search import HybridSearcher
from ai.llm import LLMClient

app = FastAPI(title="Code-RAG MVP", version="0.4.0")

_searcher: Optional[HybridSearcher] = None
def get_searcher() -> HybridSearcher:
    global _searcher
    if _searcher is None:
        _searcher = HybridSearcher()
    return _searcher

def sanitize(text: str) -> str:
    if not isinstance(text, str):
        return text
    # 去掉代理区与控制符
    text = _SURR_RE.sub('', text)
    text = _CTRL_RE.sub('', text)
    # 也可按需再做 NFC 归一化
    try:
        text = unicodedata.normalize('NFC', text)
    except Exception:
        pass
    return text


@app.get("/ping")
def ping():
    provider = (os.getenv("RAG_LLM_PROVIDER") or "openai").lower()
    model = os.getenv("RAG_LLM_MODEL") or os.getenv("QWEN_API_MODEL") or os.getenv("QWEN_GGUF_PATH") or "unset"
    return {"ok": True, "provider": provider, "model": model}

# ===== Day3: /search =====
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
    hs = get_searcher()
    rs = hs.search(req.query, top_k=req.top_k, symbol_boost=req.symbol_boost, include_documents=False)
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

@app.get("/search/{symbol}")
def search_symbol(symbol: str, top_k: int = 10):
    hs = get_searcher()
    rows = hs.search_by_symbol(symbol, top_k=top_k)
    return {"symbol": symbol, "total": len(rows), "results": rows}

# ===== Day4: /explain =====
class ExplainRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = 6
    symbol_boost: float = 2.5
    max_ctx_chars: int = 6000
    max_chunk_chars: int = 1200
    temperature: float = 0.2
    max_tokens: int = 700
    model: Optional[str] = None
    provider: Optional[str] = None  # 可覆盖环境变量（openai|qwen_api|local）

class Evidence(BaseModel):
    id: str
    path: str
    name: str
    kind: str
    start_line: int
    end_line: int
    score: float

class ExplainResponse(BaseModel):
    query: str
    answer: str
    evidences: List[Evidence]
    timings_ms: Dict[str, float]
    model: str
    provider: str
    usage: Optional[Dict[str, Any]] = None

SYSTEM_PROMPT = """You are a professional code assistant. Answer ONLY based on the provided context.
- If uncertain, say "Not sure" and state the missing info.
- Be concise; use 1–3 bullet points when helpful.
- Cite evidence like [#1], [#2].
- IMPORTANT: Respond in **English** only.
"""


def build_context_blocks(results: List[dict], max_ctx_chars: int, max_chunk_chars: int) -> str:
    ctx = []
    used = 0
    for i, r in enumerate(results, 1):
        m = r["metadata"]
        full = sanitize(r.get("text_full") or "")
        snippet = sanitize(full[:max_chunk_chars])
        block = sanitize(textwrap.dedent(f"""
        [#{i}] {m.get('kind','')} {m.get('name','')}  @ {m.get('path','')}  L{m.get('start_line',0)}-{m.get('end_line',0)}
        ---
        {snippet}
        """).strip())
        if used + len(block) > max_ctx_chars:
            break
        ctx.append(block)
        used += len(block)
    return "\n\n".join(ctx)

def build_fallback_answer(query: str, results: List[dict], max_items: int = 6) -> str:
    """LLM 不可用时的降级输出：结构化证据摘要"""
    lines = [f"（降级：LLM 不可用）基于检索证据对「{query}」的摘要："]
    for i, r in enumerate(results[:max_items], 1):
        m = r["metadata"]
        lines.append(
            f"- [#{i}] {m.get('kind','')} {m.get('name','')}  @ {m.get('path','')}  "
            f"L{m.get('start_line',0)}-{m.get('end_line',0)}（score={r.get('score',0):.2f}）"
        )
    lines.append("提示：要获取自然语言解释，请配置 RAG_LLM_PROVIDER 与对应的 API/本地模型。")
    return "\n".join(lines)

@app.post("/explain", response_model=ExplainResponse)
def explain(req: ExplainRequest):
    # 1) 检索
    t0 = time.perf_counter()
    hs = get_searcher()
    results = hs.search(
        req.query, top_k=req.top_k, symbol_boost=req.symbol_boost, include_documents=True
    )
    t1 = time.perf_counter()

    if not results:
        return ExplainResponse(
            query=req.query, answer="未检索到相关代码片段。",
            evidences=[], timings_ms={"retrieval": round((t1-t0)*1000,2), "generation": 0},
            model=req.model or (os.getenv("RAG_LLM_MODEL") or "unset"),
            provider=req.provider or (os.getenv("RAG_LLM_PROVIDER") or "openai"),
            usage=None
        )

    # 2) 上下文
    ctx_text = build_context_blocks(results, req.max_ctx_chars, req.max_chunk_chars)
    user_prompt = sanitize(textwrap.dedent(f"""
    问题：
    {req.query}

    可用证据（按编号引用）：
    {ctx_text}

    要求：
    - 优先结合证据中的函数名/注释/实现细节
    - 对"如何实现/如何使用"类问题，给出简要步骤或伪代码
    - 必要时用 [#编号] 引用证据
    """).strip())

    # 3) LLM 调用（失败→降级）
    provider = (req.provider or os.getenv("RAG_LLM_PROVIDER") or "openai").lower()
    model = req.model or os.getenv("RAG_LLM_MODEL") or os.getenv("QWEN_API_MODEL") or os.getenv("QWEN_GGUF_PATH") or "unset"

    t2 = time.perf_counter()
    try:
        llm = LLMClient(provider=provider, model=model)
        text, meta = llm.complete(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        )
        t3 = time.perf_counter()
        
        # 证据列表
        evs: List[Evidence] = []
        for r in results:
            m = r["metadata"]
            evs.append(Evidence(
                id=r["id"], path=str(m.get("path","")), name=str(m.get("name","")),
                kind=str(m.get("kind","")), start_line=int(m.get("start_line",0) or 0),
                end_line=int(m.get("end_line",0) or 0), score=float(r.get("score",0.0))
            ))
        
        return ExplainResponse(
            query=req.query, 
            answer=text, 
            evidences=evs,
            timings_ms={"retrieval": round((t1-t0)*1000,2), "generation": round((t3-t2)*1000,2)},
            model=(meta.get("usage",{}) or {}).get("model") or str(model),
            provider=provider,
            usage=meta.get("usage")  # ← 关键：透传 usage
        )
        
    except HTTPException:
        # 优雅降级：返回证据摘要（仍然 200）
        t3 = time.perf_counter()
        fallback = build_fallback_answer(req.query, results)
        evs: List[Evidence] = []
        for r in results:
            m = r["metadata"]
            evs.append(Evidence(
                id=r["id"], path=str(m.get("path","")), name=str(m.get("name","")),
                kind=str(m.get("kind","")), start_line=int(m.get("start_line",0) or 0),
                end_line=int(m.get("end_line",0) or 0), score=float(r.get("score",0.0))
            ))
        return ExplainResponse(
            query=req.query, answer=fallback, evidences=evs,
            timings_ms={"retrieval": round((t1-t0)*1000,2), "generation": round((t3-t2)*1000,2)},
            model=str(model), provider=provider,
            usage=None  # 降级时没有 usage
        )
