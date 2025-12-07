# Code-RAG MVP
MVP finished, SSE, Redis, RQ finished
Working on Agent
docker compose up -d --build
curl http://127.0.0.1:8000/ping


第 0 步：健康检查 `/ping`

```bash
curl -s http://127.0.0.1:8000/ping | python -m json.tool
```

预期：


{
  "ok": true,
  "provider": "openai",
  "model": "gpt-4o-mini"
}


---

第 1 步：重建索引 `/index/rebuild` + `/index/status`

**提交重建任务：**

```bash
curl -s -X POST "http://127.0.0.1:8000/index/rebuild" -H "Content-Type: application/json" -d '{"chunks":"data/chunks_day2.jsonl","db":"data/chroma_db","collection":"code_chunks","batch_size":100,"fresh":true}' | python -m json.tool
```

会返回：

```json
{
  "job_id": "xxxx-xxxx-..."
}
```

**然后查状态：**

```bash
curl -s "http://127.0.0.1:8000/index/status/fd709260-c52e-4c29-89be-ead884361e4e" | python -m json.tool
```

看到：

```json
"status": "finished"
```


---

第 2 步：基础 RAG `/search`


```bash
curl -s -X POST "http://127.0.0.1:8000/search" -H "Content-Type: application/json" -d '{"query":"explain this function baseIntersection","top_k":3}' | python -m json.tool
```

或者

```bash
curl -s -X POST "http://127.0.0.1:8000/search" -H "Content-Type: application/json" -d '{"query":"function","top_k":3}' | python -m json.tool


返回类似

{
  "query": "function",
  "total": 3,
  "results": [
    {
      "id": "...",
      "score": 0.98,
      "name": "xxx",
      "kind": "function",
      "path": "src/xxx.ts",
      "start_line": 10,
      "end_line": 40,
      "text_preview": "..."
    },
    ...
  ]
}



---

### 第 3 步：Explain `/explain`



curl -s -X POST "http://127.0.0.1:8000/explain" -H "Content-Type: application/json" -d '{"query":"explain this function baseIntersection","max_tokens":200}' | python -m json.tool


---

第 4 步：Agent `/agent/explain`


```bash
curl -s -X POST "http://127.0.0.1:8000/agent/explain" -H "Content-Type: application/json" -d '{"query":"please explain a function from this codebase"}' | python -m json.tool
```

返回：

* `answer`: 一段文字
* `used_tool`: 有可能是 `"search_code"` 或 `null`
* `tool_results`: 如果调了工具，就会有几条 chunk 简略信息



