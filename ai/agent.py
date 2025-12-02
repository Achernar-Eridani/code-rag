# ai/agent.py
from __future__ import annotations

import json
import os
from typing import Callable, Dict, Any, List, Tuple

from openai import OpenAI

from ai.tools import AGENT_TOOLS

# search_func: (query, top_k) -> List[Dict]
SearchFunc = Callable[[str, int], List[Dict[str, Any]]]

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
    3. 再让 LLM 基于代码片段 + 问题给出最终回答

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
            "content": (
                "You are a code review and explanation assistant for a codebase "
                "indexed by an AST-aware code search engine. "
                "When the user asks about specific code behavior, implementation, "
                "or how something works, you should usually call tools to search code. "
                "When the question is general (e.g. greetings), you can answer directly."
            ),
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

    # 3) 执行工具逻辑（目前只支持 search_code）
    if fn_name != "search_code":
        # 未知工具，降级为直接回答
        return msg.content or "", debug

    search_query = fn_args.get("query") or user_query
    top_k = int(fn_args.get("top_k") or default_top_k)

    debug["used_tool"] = "search_code"
    debug["tool_input"] = {"query": search_query, "top_k": top_k}

    code_results = search_func(search_query, top_k=top_k)
    debug["tool_results"] = code_results

    # 把工具调用的决定写回 messages（OpenAI 要求）
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

    # 把工具的输出作为 tool role 加入对话
    messages.append(
        {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": fn_name,
            # 注意：这里把检索结果序列化成 JSON 字符串给模型看
            "content": json.dumps(code_results, ensure_ascii=False),
        }
    )

    # 4) 第二轮：基于代码结果给出最终回答
    second = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
        max_tokens=max_tokens,
    )
    final_msg = second.choices[0].message
    return final_msg.content or "", debug
