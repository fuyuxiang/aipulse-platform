from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from runtime.storage.vector_store_base import VectorStore


class LocalVectorStore(VectorStore):
    def __init__(self, path: Path):
        self.path = path
        self.documents: dict[str, dict[str, Any]] = {}
        self.load()

    def add_documents(self, documents: list[dict[str, Any]]) -> None:
        for doc in documents:
            doc_id = str(doc["id"])
            vector = list(doc.get("vector") or [])
            if not vector:
                raise ValueError("document vector is required")
            self.documents[doc_id] = {**doc, "id": doc_id, "vector": vector}
        self.persist()

    def delete_documents(self, ids: list[str]) -> None:
        for doc_id in ids:
            self.documents.pop(str(doc_id), None)
        self.persist()

    def update_document(self, document: dict[str, Any]) -> None:
        self.add_documents([document])

    def similarity_search(self, query_vector: list[float], top_k: int = 5, metadata_filter: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        scored = []
        for doc in self.documents.values():
            if not self._matches(doc.get("metadata", {}), metadata_filter or {}):
                continue
            scored.append((self._cosine(query_vector, doc["vector"]), doc))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [{**doc, "score": score} for score, doc in scored[:top_k]]

    def keyword_search(self, query: str, top_k: int = 5, metadata_filter: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        words = {word.lower() for word in query.split()}
        scored = []
        for doc in self.documents.values():
            if not self._matches(doc.get("metadata", {}), metadata_filter or {}):
                continue
            text_words = {word.lower() for word in str(doc.get("text", "")).split()}
            score = len(words & text_words) / max(1, len(words))
            if score > 0:
                scored.append((score, doc))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [{**doc, "score": score} for score, doc in scored[:top_k]]

    def hybrid_search(self, query: str, query_vector: list[float], top_k: int = 5, metadata_filter: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for item in self.similarity_search(query_vector, top_k=top_k * 2, metadata_filter=metadata_filter):
            merged[item["id"]] = {**item, "score": item["score"] * 0.6}
        for item in self.keyword_search(query, top_k=top_k * 2, metadata_filter=metadata_filter):
            existing = merged.get(item["id"], item)
            existing["score"] = existing.get("score", 0) + item["score"] * 0.4
            merged[item["id"]] = existing
        return sorted(merged.values(), key=lambda item: item["score"], reverse=True)[:top_k]

    def rebuild_index(self, documents: list[dict[str, Any]]) -> None:
        self.documents = {}
        self.add_documents(documents)

    def persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(list(self.documents.values()), ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self) -> None:
        if self.path.exists():
            self.documents = {str(item["id"]): item for item in json.loads(self.path.read_text(encoding="utf-8"))}

    def metadata_filter(self, metadata_filter: dict[str, Any]) -> list[dict[str, Any]]:
        return [doc for doc in self.documents.values() if self._matches(doc.get("metadata", {}), metadata_filter)]

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        dot = sum(a * b for a, b in zip(left, right))
        norm = (math.sqrt(sum(a * a for a in left)) or 1.0) * (math.sqrt(sum(b * b for b in right)) or 1.0)
        return dot / norm

    @staticmethod
    def _matches(metadata: dict[str, Any], expected: dict[str, Any]) -> bool:
        return all(metadata.get(key) == value for key, value in expected.items())

