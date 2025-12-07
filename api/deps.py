# api/deps.py
import os
from typing import Optional, Dict

from fastapi import Header
from pydantic import BaseModel

PROVIDER_ENV_KEY_MAP: Dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "qwen_api": "QWEN_API_KEY",
    # 其它云厂商可以继续在这里加
}

DEFAULT_PROVIDER = os.getenv("RAG_LLM_PROVIDER", "local").lower()


class RequestContext(BaseModel):
    """
    请求用哪个LLM：
    - provider: local / openai / qwen_api / auto
    - api_key: 针对 cloud provider 的 key（如果有）
    """
    provider: str = "auto"
    api_key: Optional[str] = None


async def get_request_context(
    x_llm_provider: Optional[str] = Header(
        None,
        alias="x-llm-provider",
        description="Override LLM provider per request",
    ),
    x_api_key: Optional[str] = Header(
        None,
        alias="x-api-key",
        description="LLM API key for this request",
    ),
) -> RequestContext:
    # 优先使用 header，其次 fallback 环境变量。
    # 1. provider：header 优先，其次环境变量，最后 local
    provider = (x_llm_provider or DEFAULT_PROVIDER or "local").lower()

    # 2. api_key：header 优先，其次按 provider 找对应环境变量
    api_key = x_api_key
    if api_key is None:
        env_name = PROVIDER_ENV_KEY_MAP.get(provider)
        if env_name:
            api_key = os.getenv(env_name)

    return RequestContext(provider=provider, api_key=api_key)
