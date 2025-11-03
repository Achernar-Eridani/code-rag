# -*- coding: utf-8 -*-
"""
Day 3（新版栈）：Hybrid 检索 v1（符号优先 + 向量召回 + ast_path 去重）
- 不再手动生成查询向量，直接用 collection.query(query_texts=[...])
"""
"""
Hybrid 检索 v1
- 支持 include_documents: True 时返回 text_full（给 /explain 用）
- 默认 False，仅返回 text_preview（给 /search 用）
"""

import re
from typing import Any, Dict, List

import chromadb
from chromadb.config import Settings


class HybridSearcher:
    def __init__(
        self,
        db_path: str = "./data/chroma_db",
        collection_name: str = "code_chunks",
    ):
        self.client = chromadb.PersistentClient(
            path=db_path, settings=Settings(anonymized_telemetry=False)
        )
        self.col = self.client.get_collection(collection_name)

    @staticmethod
    def _extract_symbols(q: str) -> List[str]:
        backtick = re.findall(r"`([^`]+)`", q)
        ident = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", q)
        seen, out = set(), []
        for s in backtick + ident:
            s2 = s.strip()
            if s2 and s2 not in seen:
                out.append(s2); seen.add(s2)
        return out

    @staticmethod
    def _dedup_by_ast_path(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        best = {}
        for r in items:
            m = r["metadata"] or {}
            key = m.get("ast_path") or f"{m.get('path')}:{m.get('name')}"
            if key not in best or r["score"] > best[key]["score"]:
                best[key] = r
        return list(best.values())

    def search(
        self,
        query: str,
        top_k: int = 10,
        symbol_boost: float = 2.0,
        include_documents: bool = False,
    ) -> List[Dict[str, Any]]:
        assert isinstance(query, str) and query.strip(), "query 不能为空"

        # 1) 向量召回（由集合的 embedding function 自动对 query 嵌入）
        n_cand = min(top_k * 2, 100)
        hits = self.col.query(
            query_texts=[query],
            n_results=n_cand,
            include=["metadatas", "documents", "distances"],
        )

        ids = hits.get("ids", [[]])[0]
        metas = hits.get("metadatas", [[]])[0]
        docs = hits.get("documents", [[]])[0]
        dists = hits.get("distances", [[]])[0]

        symbols = [s.lower() for s in self._extract_symbols(query)]

        results: List[Dict[str, Any]] = []
        for i in range(len(ids)):
            meta = metas[i] or {}
            doc = docs[i] or ""
            dist = float(dists[i]) if dists else 0.0

            base_score = 1.0 / (1.0 + dist)

            # 2) 轻量符号加权：name 精确=1.0 / 包含=0.5；doc 出现=0.1
            sym_score = 0.0
            name = (meta.get("name") or "").lower()
            dlow = doc.lower()
            for s in symbols:
                if name == s:
                    sym_score += 1.0
                elif s in name:
                    sym_score += 0.5
                elif s in dlow:
                    sym_score += 0.1

            final = base_score + symbol_boost * sym_score

            item = {
                "id": ids[i],
                "score": final,
                "base_score": base_score,
                "symbol_score": sym_score,
                "metadata": meta,
                "text_preview": doc[:200],
                "distance": dist,
            }
            if include_documents:
                item["text_full"] = doc
            results.append(item)

        # 3) 按 ast_path 去重 + 排序 + 截断
        results = self._dedup_by_ast_path(results)
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def search_by_symbol(self, symbol: str, top_k: int = 10) -> List[Dict[str, Any]]:
        sym = symbol.strip()
        if not sym:
            return []
        rows = self.col.get(where={"name": sym}, limit=top_k, include=["metadatas", "documents"])
        out = []
        for i in range(len(rows["ids"])):
            out.append(
                {
                    "id": rows["ids"][i],
                    "metadata": rows["metadatas"][i],
                    "text_preview": (rows["documents"][i] or "")[:200],
                }
            )
        return out


if __name__ == "__main__":
    hs = HybridSearcher()
    for q in ["asciiToArray", "Parser", "get_language", "build ast path", "chunk function"]:
        print("=" * 60, "\nQuery:", q)
        rs = hs.search(q, top_k=3, include_documents=False)
        for j, r in enumerate(rs, 1):
            m = r["metadata"]
            print(f"[{j}] score={r['score']:.3f} (base={r['base_score']:.3f}, sym={r['symbol_score']:.3f})")
            print(f"    {m.get('kind')} {m.get('name')}  @ {m.get('path')}  L{m.get('start_line')}-{m.get('end_line')}")
