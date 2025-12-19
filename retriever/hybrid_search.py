# -*- coding: utf-8 -*-
"""
混合检索器 (HybridSearcher) 
- 封装 ChromaDB 的 query 调用
- 实现 Hybrid 策略：Semantic Score + Symbol Boost
- 支持动态 Collection 切换 (Workspace Isolation)
"""

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Optional
import os

class HybridSearcher:
    def __init__(self, db_path: str = "data/chroma_db"):
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False),
        )
        # 默认 Embedding (ONNX)
        self.ef = embedding_functions.ONNXMiniLM_L6_V2()
        
        # 缓存 collection 对象，避免每次 get_collection
        self._cols: Dict[str, chromadb.Collection] = {}

    def _get_col(self, name: str) -> Optional[chromadb.Collection]:
        if name in self._cols:
            return self._cols[name]
        try:
            # 尝试获取，如果不存在(没建索引)则返回 None，不要报错炸掉
            col = self.client.get_collection(name, embedding_function=self.ef)
            self._cols[name] = col
            return col
        except Exception as e:
            # print once for debug (or use logger)
            # print(f"[HybridSearcher] get_collection failed: {name} ({type(e).__name__}: {e})")
            return None


    def search(
        self,
        query: str,
        top_k: int = 10,
        symbol_boost: float = 2.0,
        include_documents: bool = False,
        collection_name: str = "code_chunks"
    ) -> List[Dict[str, Any]]:
        """
        混合检索入口
        """
        col = self._get_col(collection_name)
        if not col:
            # 集合不存在，直接返回空，不报错
            return []

        # 1. 向量检索
        # query_texts 会自动被 embedding_function 向量化
        results = col.query(
            query_texts=[query],
            n_results=top_k * 2,  # 多取一点给重排序用
            include=["metadatas", "documents", "distances"] if include_documents else ["metadatas", "distances"]
        )

        if not results["ids"] or not results["ids"][0]:
            return []

        # 2. 结果展开 & 初始打分
        # Chroma distance 是 欧氏距离? 余弦距离? 默认 L2 (squared Euclidean) -> distance越小越好
        # 转换 score = 1 / (1 + distance) 或者 1 - distance (如果 cosine)
        # 这里假设 distance 是 L2，范围 [0, +inf)
        ids = results["ids"][0]
        dists = results["distances"][0]
        metas = results["metadatas"][0]
        docs = results["documents"][0] if include_documents else [None] * len(ids)

        candidates = []
        for i, cid in enumerate(ids):
            dist = dists[i]
            meta = metas[i] or {}
            doc_text = docs[i]
            
            # 基础分
            base_score = 1.0 / (1.0 + dist)
            
            candidates.append({
                "id": cid,
                "score": base_score,
                "metadata": meta,
                "text_full": doc_text,
                "text_preview": (doc_text or "")[:80].replace("\n", " ")
            })

        # 3. 符号/路径 增强 (Re-rank)
        # 简单策略：如果 query 包含 meta['name']，加分
        q_lower = query.lower()
        
        final_list = []
        for cand in candidates:
            m = cand["metadata"]
            name = str(m.get("name", "")).lower()
            ast_path = str(m.get("ast_path", "")).lower()
            
            boost = 1.0
            # A. 精确匹配函数名
            if name and name in q_lower:
                 boost *= symbol_boost
            
            # B. 路径关键匹配 (可选)
            # if "controller" in q_lower and "controller" in ast_path:
            #     boost *= 1.2
            
            cand["score"] *= boost
            final_list.append(cand)

        # 4. 排序 & 截断
        final_list.sort(key=lambda x: x["score"], reverse=True)
        return final_list[:top_k]

    # 最小：先尝试精确，再尝试大小写不敏感（用 query 退化）
    def search_by_symbol(self, symbol: str, top_k: int = 10, collection_name: str = "code_chunks"):
        col = self._get_col(collection_name)
        if not col:
            return []

        # 1) try exact where
        try:
            results = col.get(where={"name": symbol}, limit=top_k, include=["metadatas", "documents"])
        except Exception:
            results = {"ids": [], "metadatas": [], "documents": []}

        out = []
        if results.get("ids"):
            for i, cid in enumerate(results["ids"]):
                out.append({"id": cid, "metadata": results["metadatas"][i], "text_full": results["documents"][i], "score": 1.0})
            return out

        # 2) fallback: vector query by symbol text
        return self.search(symbol, top_k=top_k, include_documents=True, collection_name=collection_name)
