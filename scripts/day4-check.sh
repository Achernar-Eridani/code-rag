#!/usr/bin/env bash
set -euo pipefail

echo "=== Day 4: 一键自检（/explain 可降级） ==="

if [ ! -d "data/chroma_db" ]; then
  echo "✗ 未发现 data/chroma_db，请先完成 Day3 入库："
  echo "  python indexer/embed_ingest.py --chunks data/chunks_day2.jsonl --fresh"
  exit 1
fi

# 起服务
echo -e "\n→ 启动 API（uvicorn）..."
python -m uvicorn api.main:app --host 127.0.0.1 --port 8002 &> /tmp/day4_api.log &
API_PID=$!
sleep 4

echo -e "\n→ /ping"
curl -s "http://127.0.0.1:8002/ping" | python -m json.tool

echo -e "\n→ /search"
curl -s -X POST "http://127.0.0.1:8002/search" \
  -H "Content-Type: application/json" \
  -d '{"query":"chunk function","top_k":3}' | python -m json.tool

echo -e "\n→ /explain（无论是否有 Key/模型，都会返回 200；有则走 LLM，无则降级）"
curl -s -X POST "http://127.0.0.1:8002/explain" \
  -H "Content-Type: application/json" \
  -d '{"query":"Explain lodash `chunk` implementation in plain words","top_k":4,"max_tokens":300}' \
  | python -m json.tool

kill $API_PID >/dev/null 2>&1 || true
echo -e "\n✓ Day 4: 自检完成"
