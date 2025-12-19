# ai/agent.py
from __future__ import annotations

import json
import os
from typing import Callable, Dict, Any, List, Tuple

from openai import OpenAI

from ai.tools import AGENT_TOOLS

# search_func: (query, top_k) -> List[Dict]
SearchFunc = Callable[[str, int], List[Dict[str, Any]]]

# 更加严格的 System Prompt，防止幻觉
AGENT_SYSTEM_PROMPT = """You are a senior code architect expert. 
Your goal is to answer user questions based STRICTLY on the provided code evidence.

Guideline:
1. Search Strategy: When searching for implementation details, use precise keywords like "function name" or "class name" rather than abstract concepts. 
   - BAD: "adapter selection"
   - GOOD: "getAdapter", "dispatchRequest", "adapter"
2. Evidence Handling: 
   - Prioritize runtime code (src/, lib/) over tests (test/, spec/).
   - If the search results contain only tests or .d.ts files, state clearly that you cannot find the runtime implementation.
   - Do NOT hallucinate code or file paths. If it's not in the tool results, it doesn't exist.
3. Citation: You must cite the evidence using [#id] format at the end of relevant sentences.
"""

# 由于使用openai function calling, 目前只能用在openai api
def _get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set, cannot run Agent with OpenAI.")
    return OpenAI(api_key=api_key)


def run_code_agent(
    user_query: str,
    search_func: SearchFunc,
    max_tokens: int = 512,
    default_top_k: int = 6,
) -> Tuple[str, Dict[str, Any]]:
    """
    最小可用 Code Agent：
    1. 先让 LLM 决定要不要调用工具（search_code）
    2. 如果调用工具，则执行 search_func 获取代码片段
    3. (New) 证据守门人：过滤掉 test/d.ts 等噪音，如果没有有效证据直接返回 Not Sure
    4. 再让 LLM 基于代码片段 + 问题给出最终回答

    返回：
        answer: 最终回答（字符串）
        debug:  调试信息，包含 used_tool / tool_input / tool_results
    """
    client = _get_openai_client()
    model = os.getenv("RAG_LLM_MODEL", "gpt-4o-mini")

    # 1) 初始对话
    messages: List[Dict[str, Any]] = [
        {
            "role": "system",
            "content": AGENT_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_query,
        },
    ]

    # 2) 第一轮：让模型决定是否使用工具
    first = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=AGENT_TOOLS,
        tool_choice="auto",
        temperature=0.2,
    )
    msg = first.choices[0].message
    tool_calls = msg.tool_calls or []

    debug: Dict[str, Any] = {
        "used_tool": None,
        "tool_input": None,
        "tool_results": None,
    }

    # 如果模型觉得不需要工具，直接返回
    if not tool_calls:
        return msg.content or "", debug

    # 我们目前只处理第一个 tool_call
    tool_call = tool_calls[0]
    fn_name = tool_call.function.name
    try:
        fn_args = json.loads(tool_call.function.arguments or "{}")
    except json.JSONDecodeError:
        fn_args = {}

    # 3) 执行工具逻辑
    if fn_name != "search_code":
        return msg.content or "", debug

    search_query = fn_args.get("query") or user_query
    top_k = int(fn_args.get("top_k") or default_top_k)

    debug["used_tool"] = "search_code"
    debug["tool_input"] = {"query": search_query, "top_k": top_k}

    # 获取原始检索结果
    raw_results = search_func(search_query, top_k=top_k)
    debug["tool_results"] = raw_results # 调试信息保留原始全量结果

    # --- 核心改进：证据守门人 (Evidence Gatekeeper) ---
    valid_evidence = []
    skipped_count = 0
    
    for res in raw_results:
        path = str(res.get("path", "")).lower()
        # 严格过滤：跳过测试文件和类型定义
        if ("/test/" in path or "/tests/" in path or "/spec/" in path or 
            "/__tests__/" in path or path.endswith(".d.ts")):
            skipped_count += 1
            continue
        valid_evidence.append(res)
    
    # 情况 A: 搜到了结果，但全被过滤掉了 (全是测试代码)
    if not valid_evidence and raw_results:
        return (
            f"I searched for '{search_query}' but only found test files or type definitions (filtered {skipped_count} results). "
            "I cannot provide a definitive answer based on runtime source code.",
            debug
        )
    
    # 情况 B: 根本没搜到
    if not raw_results:
        # 让 LLM 自己处理 "没找到" 的情况，或者直接这里拦截
        pass 

    # 4) 第二轮：只把 valid_evidence 喂给 LLM
    # 这样 LLM 根本看不到测试代码，就不会被误导
    
    messages.append(
        {
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": fn_name,
                        "arguments": tool_call.function.arguments,
                    },
                }
            ],
        }
    )

    messages.append(
        {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": fn_name,
            # 关键：喂给模型的是过滤后的干净数据
            "content": json.dumps(valid_evidence, ensure_ascii=False),
        }
    )

    second = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2, # 低温，减少胡编
        max_tokens=max_tokens,
    )
    final_msg = second.choices[0].message
    return final_msg.content or "", debug