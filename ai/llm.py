# -*- coding: utf-8 -*-
"""
LLM 适配层（可插拔）：
- provider ∈ {"openai", "qwen_api", "local"}
- 统一接口：complete(system_prompt, user_prompt, temperature, max_tokens) -> (text, meta)
- 若所需依赖/环境缺失，抛出带提示的异常（由上层决定是否降级）

更新说明（Day 7 & Refactor）：
- 引入 RequestContext 支持 per-request 的 provider/key 切换。
- 提供 get_client_for_request 工厂函数，统一返回 OpenAI 客户端（Local 模式推荐走 HTTP）。
"""

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

from fastapi import HTTPException

# 新增导入：避免循环引用，确保 api/deps.py 存在
try:
    from api.deps import RequestContext
except ImportError:
    # 兜底：如果 api.deps 还没创建，定义一个临时的 Mock
    class RequestContext:
        provider: str = "auto"
        api_key: Optional[str] = None

# 可选依赖的占位
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    import requests
except Exception:
    requests = None


# =============================================================================
#  New: Request Context Resolution Logic 
# =============================================================================

@dataclass
class LLMResolvedConfig:
    provider: str
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None  # 给 local llama.cpp HTTP server 用


def _resolve_provider(ctx: Optional[RequestContext]) -> str:
    """优先使用 ctx.provider，其次环境变量 RAG_LLM_PROVIDER，默认 local"""
    if ctx and ctx.provider and ctx.provider != "auto":
        return ctx.provider.lower()
    return os.getenv("RAG_LLM_PROVIDER", "local").lower()


def _resolve_api_key(provider: str, ctx: Optional[RequestContext]) -> Optional[str]:
    """优先使用 ctx.api_key，其次对应 provider 的环境变量"""
    if ctx and ctx.api_key:
        return ctx.api_key

    if provider == "openai":
        return os.getenv("OPENAI_API_KEY")
    if provider == "qwen_api":
        return os.getenv("QWEN_API_KEY")
    # local provider 一般不需要 key
    return None


def _resolve_model(provider: str) -> str:
    """根据 provider 选择默认模型名 (复用现有环境变量逻辑)"""
    if provider == "openai":
        return os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if provider == "qwen_api":
        return os.getenv("QWEN_API_MODEL", "qwen-max")
    # local
    return os.getenv("LOCAL_LLM_MODEL", "qwen2.5-coder-7b-instruct-q4_k_m")


def resolve_llm_config(ctx: Optional[RequestContext]) -> LLMResolvedConfig:
    """
    对外暴露：根据 RequestContext + env 得到“本次请求应该用什么配置”。
    """
    provider = _resolve_provider(ctx)
    api_key = _resolve_api_key(provider, ctx)
    model = _resolve_model(provider)

    base_url = None
    if provider == "local":
        # local http 模式下，使用 OpenAI 兼容接口地址
        base_url = (os.getenv("LOCAL_LLM_BASE") or "http://127.0.0.1:8081/v1").rstrip("/")

    return LLMResolvedConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
    )


def get_client_for_request(ctx: Optional[RequestContext] = None) -> Tuple[Any, LLMResolvedConfig]:
    """
    新入口：基于 RequestContext + env 生成一个 OpenAI client。
    返回 (client, cfg)。
    注意：此函数假定 'local' 模式也使用 OpenAI 兼容的 HTTP 接口 (llama.cpp server)。
    """
    if OpenAI is None:
        raise HTTPException(status_code=500, detail="缺少 openai 依赖，无法创建客户端")

    cfg = resolve_llm_config(ctx)

    if cfg.provider in ("openai", "qwen_api"):
        client = OpenAI(api_key=cfg.api_key)
    elif cfg.provider == "local":
        # llama.cpp OpenAI 兼容接口
        client = OpenAI(
            api_key=cfg.api_key or "EMPTY",
            base_url=cfg.base_url,
        )
    else:
        # 兜底
        client = OpenAI(api_key=cfg.api_key)

    return client, cfg


# =============================================================================
#  Legacy: LLMClient Class (保留原有逻辑，用于未迁移的代码或 Python Mode)
# =============================================================================

class LLMClient:
    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None):
        """
        provider 选择优先级：
        1) 显式参数 > 2) 环境变量 RAG_LLM_PROVIDER > 3) 默认 "local"
        """
        self.provider = (provider or os.getenv("RAG_LLM_PROVIDER") or "local").lower()

        # 统一的“模型名”字符串
        self.model = model or os.getenv("RAG_LLM_MODEL")

        if self.provider == "openai":
            if OpenAI is None:
                raise HTTPException(status_code=400, detail="缺少 openai 依赖：pip install openai>=1.30.0")
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise HTTPException(status_code=400, detail="未检测到 OPENAI_API_KEY，请 export 或写入 .env")
            self.client = OpenAI()
            self.model = self.model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            self.timeout = float(os.getenv("LLM_TIMEOUT_SEC", "60"))

        elif self.provider == "qwen_api":
            try:
                import dashscope  # type: ignore
            except Exception:
                raise HTTPException(status_code=400, detail="缺少 dashscope 依赖：pip install dashscope>=1.14.0")
            if not os.getenv("DASHSCOPE_API_KEY"):
                raise HTTPException(status_code=400, detail="未检测到 DASHSCOPE_API_KEY")
            self.dashscope = dashscope
            self.model = self.model or os.getenv("QWEN_API_MODEL", "qwen2.5-coder-7b-instruct")
            self.timeout = float(os.getenv("LLM_TIMEOUT_SEC", "60"))

        elif self.provider == "local":
            # --- 本地 LLM（默认：llama.cpp server - OpenAI 兼容） ---
            mode = (os.getenv("LOCAL_LLM_MODE") or "http").lower()
            self.timeout = float(os.getenv("LOCAL_LLM_TIMEOUT_SEC", "120")) if self.provider == "local" else 60.0

            if mode == "http":
                if requests is None:
                    raise HTTPException(status_code=400, detail="缺少 requests 依赖：pip install requests")
                self.local_http_base = (os.getenv("LOCAL_LLM_BASE") or "http://127.0.0.1:8081/v1").rstrip("/")
                self.model = self.model or os.getenv("LOCAL_LLM_MODEL", "qwen2.5-coder-7b-instruct-q4_k_m")
                self.local_mode = "http"

            elif mode == "python":
                try:
                    from llama_cpp import Llama  # type: ignore
                except Exception:
                    raise HTTPException(status_code=400, detail="缺少 llama-cpp-python：pip install llama-cpp-python")
                model_path = os.getenv("QWEN_GGUF_PATH")
                if not model_path or not os.path.exists(model_path):
                    raise HTTPException(status_code=400, detail="未找到本地模型：请设置 QWEN_GGUF_PATH 指向 .gguf 文件")
                n_ctx = int(os.getenv("LLAMA_CTX", "8192"))
                n_gpu_layers = int(os.getenv("LLAMA_N_GPU_LAYERS", "0"))
                n_threads = int(os.getenv("LLAMA_THREADS", "4"))
                
                self.Llama = Llama
                self.local_cfg = dict(model_path=model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, n_threads=n_threads)
                self.model = self.model or os.path.basename(model_path)
                self.local_mode = "python"
            else:
                raise HTTPException(status_code=400, detail=f"LOCAL_LLM_MODE 不支持：{mode}")
        else:
            raise HTTPException(status_code=400, detail=f"不支持的 provider: {self.provider}")

    def _to_usage_meta(self, *, model: str, provider: str,
                       prompt_tokens: Optional[int] = None,
                       completion_tokens: Optional[int] = None,
                       total_tokens: Optional[int] = None) -> Dict[str, Any]:
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "model": model,
            "provider": provider,
        }

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 700,
    ) -> Tuple[str, Dict[str, Any]]:

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        if self.provider == "openai":
            resp = self.client.chat.completions.create(
                model=self.model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=messages,
            )
            text = (resp.choices[0].message.content or "").strip()
            u = getattr(resp, "usage", None)
            usage = self._to_usage_meta(
                model=self.model,
                provider="openai",
                prompt_tokens=getattr(u, "prompt_tokens", None),
                completion_tokens=getattr(u, "completion_tokens", None),
                total_tokens=getattr(u, "total_tokens", None),
            )
            return text, {"usage": usage}

        if self.provider == "qwen_api":
            raise HTTPException(status_code=400, detail="Qwen API 路径暂未启用：请先使用 provider=local 或 openai")

        if self.provider == "local":
            if self.local_mode == "http":
                try:
                    r = requests.post(
                        f"{self.local_http_base}/chat/completions",
                        json={
                            "model": self.model,
                            "messages": messages,
                            "max_tokens": max_tokens,
                            "temperature": temperature,
                        },
                        timeout=self.timeout,
                    )
                    r.raise_for_status()
                    data = r.json()
                    text = (data.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()
                    usage = self._to_usage_meta(model=self.model, provider="local")
                    return text, {"usage": usage}
                except Exception as e:
                    # 自动降级逻辑
                    if os.getenv("FALLBACK_TO_OPENAI", "false").lower() == "true" and OpenAI is not None and os.getenv("OPENAI_API_KEY"):
                        client = OpenAI()
                        resp = client.chat.completions.create(
                            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                            messages=messages,
                            max_tokens=max_tokens,
                            temperature=temperature,
                        )
                        text = (resp.choices[0].message.content or "").strip()
                        u = getattr(resp, "usage", None)
                        usage = self._to_usage_meta(
                            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                            provider="openai",
                            prompt_tokens=getattr(u, "prompt_tokens", None),
                            completion_tokens=getattr(u, "completion_tokens", None),
                            total_tokens=getattr(u, "total_tokens", None),
                        )
                        return text, {"usage": usage}
                    raise HTTPException(status_code=500, detail=f"local(http) 调用失败：{e}")

            elif self.local_mode == "python":
                Llama = getattr(self, "Llama", None)
                if Llama is None:
                    raise HTTPException(status_code=400, detail="local(python) 模式未正确初始化")
                llm = Llama(**self.local_cfg)
                try:
                    resp = llm.create_chat_completion(
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    text = (resp["choices"][0]["message"]["content"] or "").strip()
                except Exception:
                    prompt = f"[SYSTEM]\n{messages[0]['content']}\n\n[USER]\n{messages[1]['content']}\n\n[ASSISTANT]\n"
                    out = llm(prompt=prompt, max_tokens=max_tokens, temperature=temperature)
                    text = (out["choices"][0]["text"] or "").strip()

                usage = self._to_usage_meta(model=self.model, provider="local")
                return text, {"usage": usage}

        raise HTTPException(status_code=400, detail=f"不支持的 provider: {self.provider}")