# -*- coding: utf-8 -*-
"""
RQ 后台任务：
1) 兼容旧接口：/index/rebuild 读取本地 chunks.jsonl -> ingest
2) 新增工业化流水线：/index/upload_and_build 接收 workspace zip -> 解压 -> chunk -> ingest（按 workspace 隔离 collection）
"""

from __future__ import annotations

import os
import re
import shutil
import zipfile
import pathlib
from typing import Dict, Any

from rq import Queue
from redis import Redis

from api.cache import get_redis
from indexer import embed_ingest, chunker


# ------------------------
# Redis / RQ
# ------------------------

def _get_redis_conn() -> Redis:
    """和 API 侧共用同一个 Redis 连接配置（REDIS_URL）。"""
    return get_redis()

def _get_queue() -> Queue:
    """所有后台任务都放在名为 'coderag' 的队列里。"""
    return Queue("coderag", connection=_get_redis_conn())


# ------------------------
# Workspace helpers
# ------------------------

_WS_ID_SANITIZE_RE = re.compile(r"[^a-zA-Z0-9_-]")

def normalize_workspace_id(raw: str) -> str:
    ws = (raw or "").strip()
    if not ws:
        return "default"
    ws = _WS_ID_SANITIZE_RE.sub("_", ws)[:64]
    return ws or "default"

def collection_name_for_workspace(wsid: str) -> str:
    wsid = normalize_workspace_id(wsid)
    return "code_chunks" if wsid == "default" else f"code_chunks__{wsid}"


# ------------------------
# Safe zip extract
# ------------------------

def _safe_extract(zipf: zipfile.ZipFile, dest_dir: pathlib.Path) -> None:
    """Prevent Zip Slip path traversal."""
    dest_dir = dest_dir.resolve()
    for info in zipf.infolist():
        rel = info.filename
        target = (dest_dir / rel).resolve()
        if not str(target).startswith(str(dest_dir) + os.sep):
            # skip suspicious paths
            continue
        zipf.extract(info, dest_dir)


# ------------------------
# Old: ingest existing JSONL
# ------------------------

def rebuild_embeddings_job(
    chunks: str = "data/chunks_day2.jsonl",
    db: str = "data/chroma_db",
    collection: str = "code_chunks",
    batch_size: int = 100,
    fresh: bool = True,
) -> Dict[str, Any]:
    """
    兼容旧的 worker 任务：
    - 读取 JSONL 直接 ingest 到指定 collection
    - 不再 flushdb（避免把 RQ job 元数据也清掉）
    """
    summary = embed_ingest.run_ingest(
        chunks=chunks,
        db=db,
        collection=collection,
        batch_size=batch_size,
        fresh=fresh,
    )
    return {"status": "ok", "summary": summary}

def enqueue_rebuild_embeddings(
    chunks: str = "data/chunks_day2.jsonl",
    db: str = "data/chroma_db",
    collection: str = "code_chunks",
    batch_size: int = 100,
    fresh: bool = True,
) -> str:
    """给 FastAPI 的入口：入队旧的 rebuild 任务。"""
    q = _get_queue()
    job = q.enqueue(
        rebuild_embeddings_job,
        chunks,
        db,
        collection,
        batch_size,
        fresh,
        job_timeout=60 * 60,
    )
    return job.id


# ------------------------
# New: Zip -> Chunk -> Ingest (workspace isolated)
# ------------------------

def build_index_from_zip_job(
    workspace_id: str,
    zip_path: str,
    fresh: bool = True,
    db: str = "data/chroma_db",
    batch_size: int = 100,
) -> Dict[str, Any]:
    """
    Worker 里执行的任务：
    1) 解压 zip 到 data/workspaces/{workspace_id}/src
    2) chunk_repo -> data/workspaces/{workspace_id}/chunks.jsonl
    3) run_ingest -> collection: code_chunks__{workspace_id}
    """
    wsid = normalize_workspace_id(workspace_id)
    base_dir = pathlib.Path("data/workspaces") / wsid
    src_dir = base_dir / "src"
    jsonl_path = base_dir / "chunks.jsonl"
    col_name = collection_name_for_workspace(wsid)

    base_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1) unzip
        if src_dir.exists():
            shutil.rmtree(src_dir)
        src_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as z:
            _safe_extract(z, src_dir)

        # 2) chunk
        total_chunks = chunker.chunk_repo(src_dir, jsonl_path)
        if total_chunks == 0:
            return {
                "status": "warning",
                "workspace_id": wsid,
                "collection": col_name,
                "message": "No supported source files found in workspace zip.",
            }

        # 3) ingest
        summary = embed_ingest.run_ingest(
            chunks=str(jsonl_path),
            db=db,
            collection=col_name,
            batch_size=batch_size,
            fresh=bool(fresh),
        )

        return {
            "status": "ok",
            "workspace_id": wsid,
            "collection": col_name,
            "chunks_count": total_chunks,
            "summary": summary,
        }

    finally:
        # Remove uploaded zip to save disk
        try:
            if zip_path and os.path.exists(zip_path):
                os.remove(zip_path)
        except Exception:
            pass


def enqueue_indexing_job(workspace_id: str, zip_path: str, fresh: bool = True) -> str:
    """给 FastAPI 的入口：入队 workspace 索引构建任务。"""
    q = _get_queue()
    job = q.enqueue(
        build_index_from_zip_job,
        workspace_id=workspace_id,
        zip_path=zip_path,
        fresh=bool(fresh),
        job_timeout=60 * 20,  # 20 min
    )
    return job.id
