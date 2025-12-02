# -*- coding: utf-8 -*-
"""
LLM 适配层（可插拔）：
- provider ∈ {"openai", "qwen_api", "local"}
- 统一接口：complete(system_prompt, user_prompt, temperature, max_tokens) -> (text, meta)
- 若所需依赖/环境缺失，抛出带提示的异常（由上层决定是否降级）

更新说明（Day 7）：
- local 模式默认改为通过 llama.cpp server（OpenAI 兼容的 /v1/chat/completions）HTTP 调用，
  避免 Windows 下编译 llama-cpp-python 的复杂度；支持可选兜底：LOCAL_LLM_MODE=python + QWEN_GGUF_PATH。
- 默认 provider 改为 local（RAG_LLM_PROVIDER 未设时）。
"""

import os
from typing import Optional, Dict, Any, Tuple

from fastapi import HTTPException

# 可选依赖的占位（仅在真正使用对应 provider 时才要求）
try:
    from openai import OpenAI
except Exception:  # 未安装也不报错，等真正使用时再提示
    OpenAI = None

# 对于 local(HTTP) 模式需要 requests
try:
    import requests
except Exception:
    requests = None


class LLMClient:
    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None):
        """
        provider 选择优先级：
        1) 显式参数 > 2) 环境变量 RAG_LLM_PROVIDER > 3) 默认 "local"
        """
        self.provider = (provider or os.getenv("RAG_LLM_PROVIDER") or "local").lower()

        # 统一的“模型名”字符串：
        # - openai: OPENAI_MODEL（默认 gpt-4o-mini）
        # - local: LOCAL_LLM_MODEL（默认 qwen2.5-coder-7b-instruct-q4_k_m）
        # - qwen_api: QWEN_API_MODEL（这里仍占位）
        self.model = model or os.getenv("RAG_LLM_MODEL")

        if self.provider == "openai":
            # --- OpenAI provider ---
            if OpenAI is None:
                raise HTTPException(status_code=400, detail="缺少 openai 依赖：pip install openai>=1.30.0")
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise HTTPException(status_code=400, detail="未检测到 OPENAI_API_KEY，请 export 或写入 .env")
            self.client = OpenAI()  # openai>=1.x 的默认读取环境变量
            self.model = self.model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            # 统一超时时间（秒），也给 local 分支使用
            self.timeout = float(os.getenv("LLM_TIMEOUT_SEC", "60"))

        elif self.provider == "qwen_api":
            # --- 阿里 DashScope（占位，按需启用） ---
            try:
                import dashscope  # type: ignore
            except Exception:
                raise HTTPException(status_code=400, detail="缺少 dashscope 依赖：pip install dashscope>=1.14.0")
            if not os.getenv("DASHSCOPE_API_KEY"):
                raise HTTPException(status_code=400, detail="未检测到 DASHSCOPE_API_KEY")
            self.dashscope = dashscope
            self.model = self.model or os.getenv("QWEN_API_MODEL", "qwen2.5-coder-7b-instruct")
            if self.provider == "local":
                self.timeout = float(os.getenv("LOCAL_LLM_TIMEOUT_SEC", "120"))  # 本地更长
            else:
                self.timeout = float(os.getenv("LLM_TIMEOUT_SEC", "60"))
                        # 这里先不实现，后续如需可补

        elif self.provider == "local":
            # --- 本地 LLM（默认：llama.cpp server - OpenAI 兼容） ---
            # 这里提供两种模式：
            # 1) HTTP（推荐，默认）：通过 llama.cpp 的 server 进程暴露 /v1/chat/completions 接口
            #    需设置 LOCAL_LLM_BASE（默认 http://127.0.0.1:8081/v1）
            #    模型名来自 LOCAL_LLM_MODEL（默认 qwen2.5-coder-7b-instruct-q4_k_m），只是一个标签字符串
            # 2) python 兜底：LOCAL_LLM_MODE=python 且设置 QWEN_GGUF_PATH 指向 .gguf 文件，
            #    使用 llama-cpp-python 直接加载（Windows 上可能涉及编译，不推荐 MVP 用）
            mode = (os.getenv("LOCAL_LLM_MODE") or "http").lower()
            if self.provider == "local":
                self.timeout = float(os.getenv("LOCAL_LLM_TIMEOUT_SEC", "120"))  # 本地更长
            else:
                self.timeout = float(os.getenv("LLM_TIMEOUT_SEC", "60"))

            if mode == "http":
                if requests is None:
                    raise HTTPException(status_code=400, detail="缺少 requests 依赖：pip install requests")
                self.local_http_base = (os.getenv("LOCAL_LLM_BASE") or "http://127.0.0.1:8081/v1").rstrip("/")
                self.model = self.model or os.getenv("LOCAL_LLM_MODEL", "qwen2.5-coder-7b-instruct-q4_k_m")
                self.local_mode = "http"

            elif mode == "python":
                # 可选兜底：直接用 llama-cpp-python 默认使用http
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
                # 保存类引用与配置，惰性实例化
                self.Llama = Llama
                self.local_cfg = dict(model_path=model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, n_threads=n_threads)
                # 将“模型名标签”也对齐，方便上层展示
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
        """
        统一把 usage 结构返回给上层。为兼容你现有 api/main.py 的读取方式，
        保留把 model/provider 放在 usage 里的习惯。 即使本地 llama.cpp 不提供 token 数 返回none usage
        """
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "model": model,
            "provider": provider,
        }

# 唯一对外接口
    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 700,
    ) -> Tuple[str, Dict[str, Any]]:

        # 统一消息格式（OpenAI / llama.cpp server 兼容）
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        if self.provider == "openai":
            # ---- OpenAI 路径 ----
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
            # ---- DashScope 占位（后续如需可实现）----
            raise HTTPException(status_code=400, detail="Qwen API 路径暂未启用：请先使用 provider=local 或 openai")

        if self.provider == "local":
            if self.local_mode == "http":
                # ---- llama.cpp server（OpenAI 兼容 HTTP）----
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
                    # llama.cpp server 一般不返回 usage；保持 None，但为兼容，你的上层仍期望 usage 有 model/provider
                    usage = self._to_usage_meta(model=self.model, provider="local")
                    return text, {"usage": usage}
                except Exception as e:
                    # 可选自动降级到 OpenAI：设置 FALLBACK_TO_OPENAI=true 并提供 OPENAI_API_KEY
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
                    # 否则抛出，让上层感知错误（日志：/tmp/llama_server.log）
                    raise HTTPException(status_code=500, detail=f"local(http) 调用失败：{e}")

            # ---- llama-cpp-python 兜底（LOCAL_LLM_MODE=python）----
            Llama = getattr(self, "Llama", None)
            if Llama is None:
                raise HTTPException(status_code=400, detail="local(python) 模式未正确初始化（缺少 llama-cpp-python 或配置）")

            # 惰性实例化（每次创建会慢；如需复用，可将 llm 提升为实例属性并自行管理并发）
            llm = Llama(**self.local_cfg)
            try:
                # 新版接口：chat completion 风格
                resp = llm.create_chat_completion(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                text = (resp["choices"][0]["message"]["content"] or "").strip()
            except Exception:
                # 退化成 prompt 拼接（旧版 API 兜底）
                prompt = f"[SYSTEM]\n{messages[0]['content']}\n\n[USER]\n{messages[1]['content']}\n\n[ASSISTANT]\n"
                out = llm(prompt=prompt, max_tokens=max_tokens, temperature=temperature)
                text = (out["choices"][0]["text"] or "").strip()

            usage = self._to_usage_meta(model=self.model, provider="local")
            return text, {"usage": usage}

        # 兜底（理论到不了这里）
        raise HTTPException(status_code=400, detail=f"不支持的 provider: {self.provider}")
