# -*- coding: utf-8 -*-
"""
Day 3：将 Day2 生成的 chunks（data/chunks_day2.jsonl）写入 Chroma（由集合的 ONNX 嵌入函数自动嵌入）。
- 采用 Chroma 1.x API：PersistentClient + collection.add(documents=..., metadatas=..., ids=...)
- 集合挂载 ONNXMiniLM_L6_V2（本地 CPU 可跑）
"""

import argparse
import json
import pathlib
from typing import Dict, Any, Iterable, List, Optional

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


def run_ingest(
    chunks: str = "data/chunks_day2.jsonl",
    db: str = "data/chroma_db",
    collection: str = "code_chunks",
    batch_size: int = 100,
    fresh: bool = False,
) -> Dict[str, Any]:
    """
    供代码 / RQ 调用的入口：
    - chunks: chunks jsonl 路径
    - db: Chroma 持久化目录
    - collection: 集合名
    - batch_size: 每批写入数量
    - fresh: 是否先删除集合再重建

    返回一个 summary dict，方便在日志 / RQ job result 里查看。
    """
    chunks_path = pathlib.Path(chunks)
    assert chunks_path.exists(), f"未找到 chunks 文件：{chunks_path}"

    client = chromadb.PersistentClient(
        path=db,
        settings=Settings(anonymized_telemetry=False),
    )
    col = ensure_collection(client, collection, fresh)
    print(f"[Day3] 使用集合：{collection}（fresh={fresh}）")

    ids: List[str] = []
    docs: List[str] = []
    metas: List[Dict[str, Any]] = []
    total = 0
    bs = int(batch_size)

    def flush():
        if not ids:
            return
        col.add(ids=ids, documents=docs, metadatas=metas)
        ids.clear()
        docs.clear()
        metas.clear()

    for obj in iter_jsonl(chunks_path):
        ids.append(obj["id"])
        docs.append(obj["text"])
        metas.append(sanitize_metadata(obj.get("meta", {})))
        total += 1
        if len(ids) >= bs:
            flush()
    flush()

    cnt = col.count()
    print(f"✓ 入库完成：输入 {total} 条；集合计数 {cnt}")

    sample = col.get(limit=1, include=["documents", "metadatas"])
    sample_id: Optional[str] = None
    sample_name: Optional[str] = None
    sample_preview: Optional[str] = None
    if sample["ids"]:
        sample_id = sample["ids"][0]
        sample_name = (sample["metadatas"][0] or {}).get("name")
        sample_preview = (sample["documents"][0] or "")[:80].replace("\n", " ")
        print("[Sample]", sample_id, sample_name, sample_preview)

    # 把关键指标返回给调用方（RQ job / 调试日志）
    return {
        "chunks": str(chunks_path),
        "db": str(pathlib.Path(db)),
        "collection": collection,
        "fresh": fresh,
        "batch_size": bs,
        "input_count": total,
        "collection_count": cnt,
        "sample_id": sample_id,
        "sample_name": sample_name,
        "sample_preview": sample_preview,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="data/chunks_day2.jsonl")
    ap.add_argument("--db", default="data/chroma_db")
    ap.add_argument("--collection", default="code_chunks")
    ap.add_argument("--batch-size", type=int, default=100)
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    summary = run_ingest(
        chunks=args.chunks,
        db=args.db,
        collection=args.collection,
        batch_size=args.batch_size,
        fresh=args.fresh,
    )
    # CLI 下已经在 run_ingest 内部打印了关键信息，这里可以不额外输出
    # 如果你想看 summary 也可以再 print 一次：
    # print(summary)


if __name__ == "__main__":
    main()
