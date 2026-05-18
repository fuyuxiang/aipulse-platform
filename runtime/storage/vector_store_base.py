from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class VectorStore(ABC):
    @abstractmethod
    def add_documents(self, documents: list[dict[str, Any]]) -> None:
        """Add embedded documents."""

    @abstractmethod
    def delete_documents(self, ids: list[str]) -> None:
        """Delete documents by id."""

    @abstractmethod
    def similarity_search(self, query_vector: list[float], top_k: int = 5, metadata_filter: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Return nearest vectors."""

