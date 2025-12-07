# -*- coding: utf-8 -*-
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pathlib, sys, time, textwrap, os, math, json
from dotenv import load_dotenv
import re, unicodedata
from fastapi.responses import StreamingResponse

from rq.job import Job

# 项目内模块
ROOT = pathlib.Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from retriever.hybrid_search import HybridSearcher
from api.cache import (
    get_cached_search_response,
    set_cached_search_response,
    get_redis,
)
from indexer.tasks import enqueue_rebuild_embeddings
from ai.agent import run_code_agent

# =============================================================================
#  New: Imports for Context & LLM Configuration (Backend Step 3)
# =============================================================================
from api.deps import RequestContext, get_request_context
from ai.llm import get_client_for_request, resolve_llm_config

load_dotenv()

# ------- 文本清洗 & token 估算 -------
_SURR_RE = re.compile(r"[\ud800-\udfff]")  # 代理区
_CTRL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")  # 控制符

def sanitize(text: str) -> str:
    if not isinstance(text, str):
        return text
    text = _SURR_RE.sub("", text)
    text = _CTRL_RE.sub("", text)
    try:
        text = unicodedata.normalize("NFC", text)
    except Exception:
        pass
    return text

def _estimate_tokens(s: str) -> int:
    """粗略估算：~4 字符 ≈ 1 token"""
    if not isinstance(s, str):
        return 0
    return max(1, math.ceil(len(s) / 4))

# ------- app & 搜索器 单例 -------
app = FastAPI(title="Code-RAG MVP", version="0.6.0")

# =============================================================================
#  New: CORS Middleware (Optional but recommended for VS Code Webview dev)
# =============================================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境建议收紧
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_searcher: Optional[HybridSearcher] = None

def get_searcher() -> HybridSearcher:
    global _searcher
    if _searcher is None:
        _searcher = HybridSearcher()
    return _searcher


# =============================================================================
#  Modified: /ping (Using RequestContext)
# =============================================================================

class PingResponse(BaseModel):
    ok: bool
    provider: str
    model: str

@app.get("/ping", response_model=PingResponse)
async def ping(ctx: RequestContext = Depends(get_request_context)):
    """
    用于 VS Code 状态栏：显示当前这次请求会用哪个 provider / model。
    支持通过 x-llm-provider / x-api-key 动态切换。
    """
    cfg = resolve_llm_config(ctx)
    
    # 如果是 local 模式，尝试探测真实 ID (可选优化)
    real_model_id = cfg.model
    if cfg.provider == "local" and cfg.base_url:
        try:
            # 简单探测一下本地服务是否活着
            import requests
            r = requests.get(f"{cfg.base_url}/models", timeout=1)
            if r.ok:
                data = r.json()
                if isinstance(data, dict) and isinstance(data.get("data"), list) and data["data"]:
                    real_model_id = data["data"][0].get("id") or cfg.model
        except Exception:
            pass  # 探测失败不影响返回配置值

    return PingResponse(
        ok=True,
        provider=cfg.provider,
        model=real_model_id,
    )


# ===== /search (No changes needed for Context, using Payload Cache) =====

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = 10
    symbol_boost: float = 2.0
    path_prefix: Optional[str] = None
    kind: Optional[str] = None
    min_score: Optional[float] = None

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
    # 1) 缓存检查
    payload = req.model_dump()
    cached = get_cached_search_response(SearchResponse, payload)
    if cached is not None:
        return cached

    # 2) 检索
    hs = get_searcher()
    rs = hs.search(
        req.query,
        top_k=req.top_k,
        symbol_boost=req.symbol_boost,
        include_documents=False,
    )

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

    # 3) 过滤
    if req.path_prefix:
        pref = req.path_prefix.replace("\\", "/")
        out = [x for x in out if x.path.replace("\\", "/").startswith(pref)]
    if req.kind:
        out = [x for x in out if (x.kind or "").lower() == req.kind.lower()]
    if req.min_score is not None:
        out = [x for x in out if x.score >= float(req.min_score)]

    resp = SearchResponse(query=req.query, total=len(out), results=out)

    # 4) 写入缓存
    set_cached_search_response(payload, resp, ttl_seconds=3600)
    return resp

@app.get("/search/{symbol}")
def search_symbol(symbol: str, top_k: int = 10):
    hs = get_searcher()
    rows = hs.search_by_symbol(symbol, top_k=top_k)
    return {"symbol": symbol, "total": len(rows), "results": rows}


# ===== /explain Utils =====

class ExplainRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = 6
    symbol_boost: float = 2.5
    max_ctx_chars: int = 6000
    max_chunk_chars: int = 1200
    temperature: float = 0.2
    max_tokens: int = 700
    # model/provider 字段保留用于兼容 API Body 传参，但现在 Header 优先级更高

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
        block = sanitize(
            textwrap.dedent(
                f"""
                [#{i}] {m.get('kind','')} {m.get('name','')}  @ {m.get('path','')}  L{m.get('start_line',0)}-{m.get('end_line',0)}
                ---
                {snippet}
                """
            ).strip()
        )
        if used + len(block) > max_ctx_chars:
            break
        ctx.append(block)
        used += len(block)
    return "\n\n".join(ctx)

def build_fallback_answer(query: str, results: List[dict], max_items: int = 6) -> str:
    lines = [f"（降级：LLM 不可用）基于检索证据对「{query}」的摘要："]
    for i, r in enumerate(results[:max_items], 1):
        m = r["metadata"]
        lines.append(
            f"- [#{i}] {m.get('kind','')} {m.get('name','')}  @ {m.get('path','')}  "
            f"L{m.get('start_line',0)}-{m.get('end_line',0)}(score={r.get('score',0):.2f})"
        )
    lines.append("Note: To obtain natural language interpretation, please configure RAG_LLM_PROVIDER.")
    return "\n".join(lines)


# =============================================================================
#  Modified: /explain (Inject Client from Context)
# =============================================================================

@app.post("/explain", response_model=ExplainResponse)
async def explain(
    req: ExplainRequest,
    ctx: RequestContext = Depends(get_request_context),
):
    # 1) 检索
    t0 = time.perf_counter()
    hs = get_searcher()
    results = hs.search(
        req.query, top_k=req.top_k, symbol_boost=req.symbol_boost, include_documents=True
    )
    t1 = time.perf_counter()

    if not results:
        cfg = resolve_llm_config(ctx)
        return ExplainResponse(
            query=req.query,
            answer="未检索到相关代码片段。",
            evidences=[],
            timings_ms={"retrieval": round((t1 - t0) * 1000, 2), "generation": 0},
            model=cfg.model,
            provider=cfg.provider,
            usage=None,
        )

    # 2) 构建上下文
    ctx_text = build_context_blocks(results, req.max_ctx_chars, req.max_chunk_chars)
    user_prompt = sanitize(
        textwrap.dedent(
            f"""
            问题：
            {req.query}

            可用证据（按编号引用）：
            {ctx_text}

            要求：
            - 优先结合证据中的函数名/注释/实现细节
            - 对"如何实现/如何使用"类问题，给出简要步骤或伪代码
            - 必要时用 [#编号] 引用证据
            """
        ).strip()
    )

    # 3) LLM 调用 (使用 get_client_for_request 获取配置好的 Client)
    client, cfg = get_client_for_request(ctx)
    
    t2 = time.perf_counter()
    try:
        # 统一使用 OpenAI SDK 风格调用（ai.llm 保证了 local 模式也返回兼容的 client）
        resp = client.chat.completions.create(
            model=cfg.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        )
        t3 = time.perf_counter()
        
        text = (resp.choices[0].message.content or "").strip()
        
        # 处理 Usage
        u = getattr(resp, "usage", None)
        usage = {
            "prompt_tokens": getattr(u, "prompt_tokens", 0),
            "completion_tokens": getattr(u, "completion_tokens", 0),
            "total_tokens": getattr(u, "total_tokens", 0),
            "model": cfg.model,
            "provider": cfg.provider,
        }
        # 如果是 Local 模式且没返回 token 数，做个估算
        if usage["total_tokens"] == 0:
             pt = _estimate_tokens(ctx_text)
             ct = _estimate_tokens(text)
             usage = {**usage, "prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt+ct, "estimated": True}

        # 构造 Evidence 列表
        evs = [
            Evidence(
                id=r["id"],
                path=str(r["metadata"].get("path", "")),
                name=str(r["metadata"].get("name", "")),
                kind=str(r["metadata"].get("kind", "")),
                start_line=int(r["metadata"].get("start_line", 0)),
                end_line=int(r["metadata"].get("end_line", 0)),
                score=float(r.get("score", 0.0)),
            ) for r in results
        ]

        return ExplainResponse(
            query=req.query,
            answer=text,
            evidences=evs,
            timings_ms={
                "retrieval": round((t1 - t0) * 1000, 2),
                "generation": round((t3 - t2) * 1000, 2),
            },
            model=cfg.model,
            provider=cfg.provider,
            usage=usage,
        )

    except Exception:
        # 降级处理
        t3 = time.perf_counter()
        fallback = build_fallback_answer(req.query, results)
        evs = [
            Evidence(
                id=r["id"],
                path=str(r["metadata"].get("path", "")),
                name=str(r["metadata"].get("name", "")),
                kind=str(r["metadata"].get("kind", "")),
                start_line=int(r["metadata"].get("start_line", 0)),
                end_line=int(r["metadata"].get("end_line", 0)),
                score=float(r.get("score", 0.0)),
            ) for r in results
        ]
        return ExplainResponse(
            query=req.query,
            answer=fallback,
            evidences=evs,
            timings_ms={
                "retrieval": round((t1 - t0) * 1000, 2),
                "generation": round((t3 - t2) * 1000, 2),
            },
            model=cfg.model,
            provider=cfg.provider,
            usage=None,
        )


# =============================================================================
#  Modified: /explain_stream (Inject Client from Context)
# =============================================================================

@app.post("/explain_stream")
async def explain_stream(
    req: ExplainRequest,
    ctx: RequestContext = Depends(get_request_context),
):
    # 1. 检索
    t0 = time.perf_counter()
    hs = get_searcher()
    results = hs.search(
        req.query, top_k=req.top_k, symbol_boost=req.symbol_boost, include_documents=True,
    )
    t1 = time.perf_counter()

    if not results:
        def no_result():
            payload = {"type": "done", "error": "no_results", "message": "未检索到相关代码片段。"}
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        return StreamingResponse(no_result(), media_type="text/event-stream")

    # 2. 构建上下文
    ctx_text = build_context_blocks(results, req.max_ctx_chars, req.max_chunk_chars)
    user_prompt = sanitize(
        textwrap.dedent(
            f"""
            问题：
            {req.query}
            可用证据（按编号引用）：
            {ctx_text}
            要求：
            - 优先结合证据中的函数名/注释/实现细节
            - 对"如何实现/如何使用"类问题，给出简要步骤或伪代码
            - 必要时用 [#编号] 引用证据
            """
        ).strip()
    )

    # 获取配置
    client, cfg = get_client_for_request(ctx)

    # 3. 生成器
    def event_stream():
        # A) 发送 Meta
        head = {
            "type": "meta",
            "query": req.query,
            "retrieval_ms": round((t1 - t0) * 1000, 2),
            "total_hits": len(results),
            "provider": cfg.provider,
            "model": cfg.model,
        }
        yield f"data: {json.dumps(head, ensure_ascii=False)}\n\n"

        t2 = time.perf_counter()
        full_text = ""
        try:
            # B) 调用 LLM (阻塞式获取全部，然后模拟流式吐出)
            # 注：如果你想做真流式，可以在 create 里加 stream=True，然后迭代 response
            resp = client.chat.completions.create(
                model=cfg.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=req.temperature,
                max_tokens=req.max_tokens,
            )
            full_text = (resp.choices[0].message.content or "").strip()
            t3 = time.perf_counter()

            # C) 模拟流式推送
            chunk_size = 20
            for i in range(0, len(full_text), chunk_size):
                chunk = full_text[i : i + chunk_size]
                if chunk:
                    payload = {"type": "chunk", "text": chunk}
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

            # D) Usage
            u = getattr(resp, "usage", None)
            usage = {
                "prompt_tokens": getattr(u, "prompt_tokens", 0),
                "completion_tokens": getattr(u, "completion_tokens", 0),
                "total_tokens": getattr(u, "total_tokens", 0),
                "model": cfg.model,
                "provider": cfg.provider,
            }
            if usage["total_tokens"] == 0:
                pt = _estimate_tokens(ctx_text)
                ct = _estimate_tokens(full_text)
                usage = {**usage, "prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt+ct, "estimated": True}

            done_payload = {
                "type": "done",
                "timings_ms": {
                    "retrieval": round((t1 - t0) * 1000, 2),
                    "generation": round((t3 - t2) * 1000, 2),
                },
                "usage": usage,
            }
            yield f"data: {json.dumps(done_payload, ensure_ascii=False)}\n\n"

        except Exception as e:
            t3 = time.perf_counter()
            fallback = build_fallback_answer(req.query, results)
            err_payload = {
                "type": "error", 
                "message": str(e),
                "timings_ms": {"retrieval": round((t1 - t0) * 1000, 2), "generation": round((t3 - t2) * 1000, 2)}
            }
            yield f"data: {json.dumps(err_payload, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'chunk', 'text': fallback}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'usage': None}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ===== /index/rebuild & /index/status (Unchanged) =====

class IndexRebuildRequest(BaseModel):
    chunks: str = "data/chunks_day2.jsonl"
    db: str = "data/chroma_db"
    collection: str = "code_chunks"
    batch_size: int = 100
    fresh: bool = True

class IndexRebuildResponse(BaseModel):
    job_id: str

class IndexStatus(BaseModel):
    job_id: str
    status: str
    error: Optional[str] = None
    enqueued_at: Optional[str] = None
    started_at: Optional[str] = None
    ended_at: Optional[str] = None

@app.post("/index/rebuild", response_model=IndexRebuildResponse)
def index_rebuild(req: IndexRebuildRequest):
    job_id = enqueue_rebuild_embeddings(
        chunks=req.chunks, db=req.db, collection=req.collection, 
        batch_size=req.batch_size, fresh=req.fresh
    )
    return IndexRebuildResponse(job_id=job_id)

@app.get("/index/status/{job_id}", response_model=IndexStatus)
def index_status(job_id: str):
    r = get_redis()
    try:
        job = Job.fetch(job_id, connection=r)
    except Exception:
        raise HTTPException(status_code=404, detail="Job not found")

    def ts(dt): return dt.isoformat() if dt else None
    return IndexStatus(
        job_id=job.id, status=job.get_status(), error=str(job.exc_info) if job.is_failed else None,
        enqueued_at=ts(job.enqueued_at), started_at=ts(job.started_at), ended_at=ts(job.ended_at),
    )


# ===== Agent Section =====

def agent_search_adapter(query: str, top_k: int = 6) -> List[Dict[str, Any]]:
    searcher = get_searcher()
    results = searcher.search(query=query, top_k=top_k, include_documents=True)
    simplified: List[Dict[str, Any]] = []
    for r in results:
        meta = r.get("metadata", {})
        simplified.append({
            "path": meta.get("path"),
            "symbol": meta.get("name"),
            "kind": meta.get("kind"),
            "start_line": meta.get("start_line"),
            "end_line": meta.get("end_line"),
            "score": r.get("score"),
            "code": (r.get("text_full") or r.get("text_preview") or "")
        })
    return simplified

class AgentExplainRequest(BaseModel):
    query: str
    max_tokens: int = 512

class AgentExplainResponse(BaseModel):
    query: str
    answer: str
    used_tool: Optional[str] = None
    tool_input: Optional[Dict[str, Any]] = None
    tool_results: Optional[List[Dict[str, Any]]] = None

@app.post("/agent/explain", response_model=AgentExplainResponse)
def agent_explain(
    req: AgentExplainRequest,
    ctx: RequestContext = Depends(get_request_context) # Added dependency
) -> AgentExplainResponse:
    # TODO: Pass 'ctx' or 'cfg' to run_code_agent to support dynamic provider switching for agents.
    # Currently it uses default env vars inside run_code_agent.
    try:
        answer, debug = run_code_agent(
            user_query=req.query,
            search_func=agent_search_adapter,
            max_tokens=req.max_tokens,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {e}")

    tool_results = debug.get("tool_results") or None
    if tool_results:
        tool_results = tool_results[:3]

    return AgentExplainResponse(
        query=req.query,
        answer=answer,
        used_tool=debug.get("used_tool"),
        tool_input=debug.get("tool_input"),
        tool_results=tool_results,
    )