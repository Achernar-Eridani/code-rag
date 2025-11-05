#!/usr/bin/env bash
set -euo pipefail

# 1) Ping
echo "== Ping =="
curl -s http://127.0.0.1:8000/ping | python -m json.tool

# 2) 短评测
echo
echo "== Eval /search + /explain (qa_min.jsonl) =="
python eval/run_eval.py --base http://127.0.0.1:8000 --data eval/qa_min.jsonl --outdir eval/results --top_k 5 --max_tokens 400 --max_ctx 6000 | tee eval/results/console.txt

echo
echo "Report written to eval/results/report.json"
