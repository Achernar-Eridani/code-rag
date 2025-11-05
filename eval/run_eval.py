# -*- coding: utf-8 -*-
"""
Day6 Mini Eval: /search + /explain
- 读 jsonl 数据集
- 计算 /search 的 Hit@K、MRR@K
- 计算 /explain 的 must_any 命中率
- 统计 P50/P95 延迟与平均 token 用量
"""
import json, time, statistics, argparse, os
import requests

def load_jsonl(p):
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                yield json.loads(line)

def pct(arr, p):
    if not arr: return 0.0
    arr_sorted = sorted(arr)
    k = (len(arr_sorted)-1) * p
    f = int(k); c = min(f+1, len(arr_sorted)-1)
    if f == c: return float(arr_sorted[f])
    return float(arr_sorted[f] + (arr_sorted[c]-arr_sorted[f])*(k-f))

def mrr_at_k(ranked_items, gold_set, key_fn=lambda x: x):
    """gold_set: set of strings to match; ranked_items: list; key_fn-> string"""
    for idx, it in enumerate(ranked_items, start=1):
        if key_fn(it) in gold_set:
            return 1.0/idx
    return 0.0

def run(args):
    base = args.base.rstrip("/")
    top_k = args.top_k

    qas = list(load_jsonl(args.data))
    # metrics
    srch_lat, expl_lat = [], []
    srch_hit, srch_mrr = [], []
    expl_hit = []
    tok_in, tok_out, tok_total = [], [], []

    for ex in qas:
        qid = ex["id"]
        query = ex["query"]
        expect = ex.get("expect", {})
        hits = set(expect.get("hit", []))
        must_any = expect.get("must_any", [])

        # ---- /search ----
        t0 = time.perf_counter()
        try:
            r = requests.post(f"{base}/search", json={"query": query, "top_k": top_k, "symbol_boost": 2.0}, timeout=60)
            r.raise_for_status()
            sres = r.json()
        except Exception as e:
            print(f"[{qid}] /search error: {e}")
            sres = {"results": []}
        t1 = time.perf_counter()
        srch_lat.append((t1-t0)*1000)

        results = sres.get("results", [])
        # format "path:name"
        ranked_keys = [f'{it.get("path","")}:{it.get("name","")}' for it in results]
        # Hit@K（命中任意 gold）
        hit = any(k in hits for k in ranked_keys)
        srch_hit.append(1.0 if hit else 0.0)
        # MRR@K
        srch_mrr.append(mrr_at_k(ranked_keys, set(hits), key_fn=lambda x: x))

        # ---- /explain ----
        t2 = time.perf_counter()
        try:
            r = requests.post(f"{base}/explain", json={
                "query": query, "top_k": top_k, "max_tokens": args.max_tokens, "max_ctx_chars": args.max_ctx
            }, timeout=120)
            r.raise_for_status()
            eres = r.json()
        except Exception as e:
            print(f"[{qid}] /explain error: {e}")
            eres = {"answer": "", "usage": {}}
        t3 = time.perf_counter()
        expl_lat.append((t3-t2)*1000)

        ans = (eres.get("answer") or "").lower()
        ok = any(k.lower() in ans for k in must_any) if must_any else bool(ans.strip())
        expl_hit.append(1.0 if ok else 0.0)

        usage = eres.get("usage") or {}
        if "prompt_tokens" in usage: tok_in.append(usage["prompt_tokens"])
        if "completion_tokens" in usage: tok_out.append(usage["completion_tokens"])
        if "total_tokens" in usage: tok_total.append(usage["total_tokens"])

    report = {
        "n": len(qas),
        "search": {
            "top_k": top_k,
            "hit@k": sum(srch_hit)/len(srch_hit) if srch_hit else 0.0,
            "mrr@k": sum(srch_mrr)/len(srch_mrr) if srch_mrr else 0.0,
            "latency_ms": {
                "p50": pct(srch_lat, 0.50),
                "p95": pct(srch_lat, 0.95)
            }
        },
        "explain": {
            "must_any_acc": sum(expl_hit)/len(expl_hit) if expl_hit else 0.0,
            "latency_ms": {
                "p50": pct(expl_lat, 0.50),
                "p95": pct(expl_lat, 0.95)
            },
            "avg_tokens": {
                "prompt": sum(tok_in)/len(tok_in) if tok_in else 0.0,
                "completion": sum(tok_out)/len(tok_out) if tok_out else 0.0,
                "total": sum(tok_total)/len(tok_total) if tok_total else 0.0
            }
        }
    }
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--data", default="eval/qa_min.jsonl")
    ap.add_argument("--outdir", default="eval/results")
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--max_tokens", type=int, default=400)
    ap.add_argument("--max_ctx", type=int, default=6000)
    args = ap.parse_args()
    run(args)
