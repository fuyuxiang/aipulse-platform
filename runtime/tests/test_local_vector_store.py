from __future__ import annotations

from runtime.storage.local_vector_store import LocalVectorStore


def test_local_vector_store_hybrid_and_delete(tmp_path) -> None:
    store = LocalVectorStore(tmp_path / "vectors.json")
    store.add_documents(
        [
            {"id": "a", "text": "alpha beta", "vector": [1.0, 0.0], "metadata": {"tenant_id": "t1"}},
            {"id": "b", "text": "gamma", "vector": [0.0, 1.0], "metadata": {"tenant_id": "t1"}},
        ]
    )
    assert store.similarity_search([1.0, 0.0], metadata_filter={"tenant_id": "t1"})[0]["id"] == "a"
    assert store.keyword_search("alpha")[0]["id"] == "a"
    assert store.hybrid_search("alpha", [1.0, 0.0])[0]["id"] == "a"
    store.delete_documents(["a"])
    assert all(item["id"] != "a" for item in store.similarity_search([1.0, 0.0], top_k=10))

