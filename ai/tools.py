# ai/tools.py
"""
定义 Code Agent 可用的工具（OpenAI tools / function calling 格式）
目前只做一个 search_code，后续可以加 lint_code 等。
"""

AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": (
                "Semantic search for relevant code snippets. "
                "Use this when you need to understand implementation details, "
                "find function definitions, or explain how code works."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query, e.g. 'how does redis cache work in this project'"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "How many code snippets to retrieve (default 6)",
                        "minimum": 1,
                        "maximum": 20
                    },
                },
                "required": ["query"],
            },
        },
    }
]
