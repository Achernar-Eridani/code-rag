# -*- coding: utf-8 -*-
"""
RQ 后台任务：重建 Chroma 索引 + 清理 Redis 缓存。
"""

from typing import Dict, Any, Optional

from rq import Queue
from redis import Redis

from api.cache import get_redis   # 复用你在 api/cache.py 里的 Redis 连接
from indexer import embed_ingest


def _get_redis_conn() -> Redis:
    """
    和 API 侧共用同一个 Redis 连接配置（REDIS_URL）。
    """
    return get_redis()


def _get_queue() -> Queue:
    """
    所有后台任务都放在名为 "coderag" 的队列里。
    对应 docker-compose 里的： rq worker coderag
    """
    return Queue("coderag", connection=_get_redis_conn())


def rebuild_embeddings_job(
    chunks: str = "data/chunks_day2.jsonl",
    db: str = "data/chroma_db",
    collection: str = "code_chunks",
    batch_size: int = 100,
    fresh: bool = True,
) -> Dict[str, Any]:
    """
    真正在 worker 容器里执行的任务：
    1. 调用 embed_ingest.run_ingest 写入 / 重建 Chroma 索引
    2. 重建成功后，清空 Redis 里缓存的 /search 结果
    """
    # 1) 重建索引
    summary = embed_ingest.run_ingest(
        chunks=chunks,
        db=db,
        collection=collection,
        batch_size=batch_size,
        fresh=fresh,
    )

    # 2) 简单粗暴：刷新 Redis 当前 DB（假设这个 Redis 专门给本项目用）
    r = _get_redis_conn()
    r.flushdb()

    return {
        "status": "ok",
        "summary": summary,
    }


def enqueue_rebuild_embeddings(
    chunks: str = "data/chunks_day2.jsonl",
    db: str = "data/chroma_db",
    collection: str = "code_chunks",
    batch_size: int = 100,
    fresh: bool = True,
) -> str:
    """
    提供给 FastAPI 使用的入口：
    - 入队一个重建任务
    - 返回 job_id，方便 /index/status 轮询查看进度
    """
    q = _get_queue()
    job = q.enqueue(
        rebuild_embeddings_job,
        chunks,
        db,
        collection,
        batch_size,
        fresh,
        job_timeout=60 * 60,  # 最多跑 1 小时，足够你一个 repo 重建
    )
    return job.id

