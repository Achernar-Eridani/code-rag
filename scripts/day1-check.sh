#!/usr/bin/env bash
set -e
echo "=== Day 1 Completion Check ==="
# 准备样例
mkdir -p sandbox
cat > sandbox/demo.ts <<'TS'
export function add(a, b) { return a + b; }
class Calc { sum(x, y) { return x + y; } }
const helper = (x) => x * 2;
TS
# 启动FastAPI（后台）
python -m uvicorn api.main:app --port 8001 &> /tmp/mvp_uvicorn.log &
UV_PID=$!
sleep 2
# 健康检查
curl -s http://127.0.0.1:8001/ping | grep -q '"ok":true' && echo "✓ FastAPI /ping OK" || (echo "✗ FastAPI failed" && exit 1)
kill $UV_PID 2>/dev/null || true
# AST 导出
python indexer/ast_dump.py --repo ./sandbox --out ./data/ast_min.jsonl
test -s ./data/ast_min.jsonl && echo "✓ AST JSONL generated" || (echo "✗ AST output missing" && exit 1)
# Git tag 检查（若还未打）
git tag | grep -q '^mvp-day1$' && echo "✓ tag mvp-day1 exists" || echo "ℹ tip: git tag mvp-day1"
echo "All checks passed."
