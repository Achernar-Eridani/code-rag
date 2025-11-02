# -*- coding: utf-8 -*-
"""
Day 3：将 Day2 生成的 chunks（data/chunks_day2.jsonl）写入 Chroma（由集合的 ONNX 嵌入函数自动嵌入）。
- 采用 Chroma 1.x API：PersistentClient + collection.add(documents=..., metadatas=..., ids=...)
- 集合挂载 ONNXMiniLM_L6_V2（本地 CPU 可跑）
"""
import argparse
import json
import pathlib
from typing import Dict, Any, Iterable, List

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions  # 1.x：内置多种 embedding function


def iter_jsonl(path: pathlib.Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def sanitize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in meta.items():
        if v is None:
            out[k] = ""
        elif isinstance(v, (bool, int, float, str)):
            out[k] = v
        else:
            out[k] = str(v)
    return out


def ensure_collection(client: chromadb.ClientAPI, name: str, fresh: bool):
    if fresh:
        try:
            client.delete_collection(name)
        except Exception:
            pass
    try:
        col = client.get_collection(name)
    except Exception:
        # 关键：在创建集合时挂载 ONNX 嵌入函数（默认 all-MiniLM-L6-v2）
        onnx_ef = embedding_functions.ONNXMiniLM_L6_V2()
        col = client.create_collection(
            name=name,
            embedding_function=onnx_ef,
            metadata={"desc": "Code chunks (AST-aware) from Day2"},
        )
    return col


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="data/chunks_day2.jsonl")
    ap.add_argument("--db", default="data/chroma_db")
    ap.add_argument("--collection", default="code_chunks")
    ap.add_argument("--batch-size", type=int, default=100)
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    p = pathlib.Path(args.chunks)
    assert p.exists(), f"未找到 chunks 文件：{p}"

    # 1) 初始化持久化客户端（1.x 仍支持 PersistentClient）
    client = chromadb.PersistentClient(
        path=args.db, settings=Settings(anonymized_telemetry=False)
    )
    col = ensure_collection(client, args.collection, args.fresh)
    print(f"[Day3] 使用集合：{args.collection}（fresh={args.fresh}）")

    # 2) 分批写入（documents + metadatas + ids；由集合的 embedding_function 自动嵌入）
    ids: List[str] = []
    docs: List[str] = []
    metas: List[Dict[str, Any]] = []
    total = 0
    bs = int(args.batch_size)

    def flush():
        if not ids:
            return
        col.add(ids=ids, documents=docs, metadatas=metas)
        ids.clear(); docs.clear(); metas.clear()

    for obj in iter_jsonl(p):
        ids.append(obj["id"])
        docs.append(obj["text"])
        metas.append(sanitize_metadata(obj.get("meta", {})))
        total += 1
        if len(ids) >= bs:
            flush()
    flush()

    # 3) 验证
    cnt = col.count()
    print(f"✓ 入库完成：输入 {total} 条；集合计数 {cnt}")
    sample = col.get(limit=1, include=["documents", "metadatas"])
    if sample["ids"]:
        print("[Sample]", sample["ids"][0], sample["metadatas"][0].get("name"),
              (sample["documents"][0] or "")[:80].replace("\n", " "))


if __name__ == "__main__":
    main()
