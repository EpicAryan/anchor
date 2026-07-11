from __future__ import annotations

from pathlib import Path

import chromadb


class VectorStore:
    def __init__(self, persist_dir: Path):
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(
            "anchor", metadata={"hnsw:space": "cosine"})

    def add(self, vector_ids: list[str], embeddings: list[list[float]],
            texts: list[str], metadatas: list[dict]) -> None:
        if not vector_ids:
            return
        self._collection.upsert(
            ids=vector_ids, embeddings=embeddings,
            documents=texts, metadatas=metadatas)

    def delete(self, vector_ids: list[str]) -> None:
        if vector_ids:
            self._collection.delete(ids=vector_ids)

    def query(self, embedding: list[float], top_k: int = 5,
              source_type: str | None = None) -> list[dict]:
        n = min(top_k, self._collection.count())
        if n == 0:
            return []
        where = {"source_type": source_type} if source_type else None
        res = self._collection.query(
            query_embeddings=[embedding], n_results=n, where=where)
        return [
            {"vector_id": vid, "text": doc, "metadata": meta, "distance": dist}
            for vid, doc, meta, dist in zip(
                res["ids"][0], res["documents"][0],
                res["metadatas"][0], res["distances"][0])
        ]
