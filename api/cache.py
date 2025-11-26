# api/cache.py
import os
import json
import hashlib
from typing import Any, Dict, Optional

import redis

_redis_client: Optional[redis.Redis] = None


def get_redis() -> redis.Redis:
    """
    单例 Redis 连接，URL 默认指向 docker-compose 里的 redis 服务。
    """
    global _redis_client
    if _redis_client is None:
        url = os.getenv("REDIS_URL", "redis://redis:6379/0")
        _redis_client = redis.Redis.from_url(url)
    return _redis_client


def _search_key(payload: Dict[str, Any]) -> str:
    """
    将 /search 的请求体序列化为稳定字符串，再做一层 hash。
    这样即便 query 比较长，Redis key 也不会太夸张。
    """
    base = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha256(base.encode("utf-8")).hexdigest()
    return f"search:{digest}"


def get_cached_search_response(model_cls, payload: Dict[str, Any]):
    """
    如果命中缓存，就直接反序列化成 Pydantic 模型返回。
    """
    r = get_redis()
    key = _search_key(payload)
    raw = r.get(key)
    if not raw:
        return None
    data = json.loads(raw)
    return model_cls(**data)


def set_cached_search_response(payload: Dict[str, Any], response_obj, ttl_seconds: int = 3600):
    """
    把 SearchResponse 整体序列化进 Redis，默认 TTL = 1 小时。
    """
    r = get_redis()
    key = _search_key(payload)
    r.setex(key, ttl_seconds, response_obj.model_dump_json())
