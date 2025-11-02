#!/usr/bin/env bash
set -euo pipefail

echo "=== Day 3: 一键自检开始（Chroma 1.x 栈） ==="

if [ -d "data/chroma_db" ]; then
  echo "✓ Chroma DB 存在：data/chroma_db"
else
  echo "✗ 未发现 data/chroma_db，请先执行入库脚本："
  echo "  python indexer/embed_ingest.py --chunks data/chunks_day2.jsonl --fresh"
  exit 1
fi

echo -e "\n→ 启动 API（uvicorn）..."
python -m uvicorn api.main:app --host 127.0.0.1 --port 8001 &> /tmp/day3_api.log &
API_PID=$!
sleep 4

echo -e "\n→ /ping"
curl -s "http://127.0.0.1:8001/ping" | python -m json.tool

echo -e "\n→ /search"
curl -s -X POST "http://127.0.0.1:8001/search" \
  -H "Content-Type: application/json" \
  -d '{"query":"asciiToArray","top_k":3}' | python -m json.tool

echo -e "\n→ /search/{symbol}"
curl -s "http://127.0.0.1:8001/search/Parser?top_k=2" | python -m json.tool

kill $API_PID >/dev/null 2>&1 || true
echo -e "\n✓ Day 3: 自检完成"
