# -*- coding: utf-8 -*-
"""
LLM 适配层（可插拔）：
- provider ∈ {"openai", "qwen_api", "local"}
- 统一接口：complete(system_prompt, user_prompt, temperature, max_tokens) -> (text, meta)
- 若所需依赖/环境缺失，抛出带提示的异常（由上层决定是否降级）
"""
import os
from typing import Optional, Dict, Any, Tuple

from fastapi import HTTPException

# 可选依赖的占位
try:
    from openai import OpenAI
except Exception:  # 未安装也不报错，等真正使用时再提示
    OpenAI = None

class LLMClient:
    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None):
        self.provider = (provider or os.getenv("RAG_LLM_PROVIDER") or "openai").lower()
        self.model = model or os.getenv("RAG_LLM_MODEL")

        if self.provider == "openai":
            if OpenAI is None:
                raise HTTPException(status_code=400, detail="缺少 openai 依赖：pip install openai>=1.30.0")
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise HTTPException(status_code=400, detail="未检测到 OPENAI_API_KEY，请 export 该变量或写入 .env")
            self.client = OpenAI()
            self.model = self.model or "gpt-4o-mini"

        elif self.provider == "qwen_api":
            # 走阿里 DashScope（可选，今天先打壳）
            try:
                import dashscope  # type: ignore
            except Exception:
                raise HTTPException(status_code=400, detail="缺少 dashscope 依赖：pip install dashscope>=1.14.0")
            if not os.getenv("DASHSCOPE_API_KEY"):
                raise HTTPException(status_code=400, detail="未检测到 DASHSCOPE_API_KEY")
            self.dashscope = dashscope
            self.model = self.model or os.getenv("QWEN_API_MODEL", "qwen2.5-coder-7b-instruct")

        elif self.provider == "local":
            # llama.cpp （GGUF，本地推理）
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
            # 简化初始化；Day7 再做更细参数调优
            self.Llama = Llama  # 保存类引用
            self.local_cfg = dict(model_path=model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, n_threads=n_threads)
        else:
            raise HTTPException(status_code=400, detail=f"不支持的 provider: {self.provider}")

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 700,
    ) -> Tuple[str, Dict[str, Any]]:
        if self.provider == "openai":
            resp = self.client.chat.completions.create(
                model=self.model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            text = resp.choices[0].message.content or ""
            usage = getattr(resp, "usage", None)
            usage_dict = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
                "model": self.model,
                "provider": "openai",
            }
            return text, {"usage": usage_dict}

        if self.provider == "qwen_api":
            # 轻量示范（真实参数以 dashscope SDK 为准；今天先不强绑定）
            # 这里抛出“暂未实现”的温和提示，Day7 再补齐
            raise HTTPException(status_code=400, detail="Qwen API 路径暂未启用：请先使用 provider=openai 或 local")

        if self.provider == "local":
            # 惰性加载（避免 import 成本 & 提示更准确）
            Llama = self.Llama
            llm = Llama(**self.local_cfg)
            # 使用 chat completion 风格（较新的 llama-cpp 支持）
            try:
                resp = llm.create_chat_completion(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                text = resp["choices"][0]["message"]["content"]
            except Exception:
                # 退化成 prompt 拼接（旧版 API 兜底）
                prompt = f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{user_prompt}\n\n[ASSISTANT]\n"
                out = llm(prompt=prompt, max_tokens=max_tokens, temperature=temperature)
                text = out["choices"][0]["text"]
            return text, {"usage": {"model": os.path.basename(self.local_cfg["model_path"]), "provider": "local"}}

        raise HTTPException(status_code=400, detail=f"不支持的 provider: {self.provider}")
