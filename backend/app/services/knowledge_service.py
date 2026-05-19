from __future__ import annotations

import base64
import csv
import re
import sys
import zipfile
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.constants import ErrorCode
from app.core.errors import AppError
from app.services.model_services import deterministic_embedding
from app.services.resource_service import ResourceService

project_root = settings.project_root
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from runtime.storage.local_object_store import LocalObjectStore  # noqa: E402
from runtime.storage.local_vector_store import LocalVectorStore  # noqa: E402


class KnowledgeService:
    def __init__(self, db: Session):
        self.resources = ResourceService(db)
        self.object_store = LocalObjectStore(settings.resolve_path(settings.object_store_dir))
        self.vector_root = settings.resolve_path(settings.vector_store_dir)

    def upload_document(self, tenant_id: str, user_id: str, kb_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        kb = self.resources.get("knowledge_bases", tenant_id, kb_id)
        filename = self._safe_filename(str(payload.get("filename") or payload.get("name") or "document.txt"))
        data = self._payload_bytes(payload)
        if not data:
            raise AppError(ErrorCode.VALIDATION_ERROR, "document content or base64 data is required for local upload", 422)
        content_type = str(payload.get("content_type") or self._content_type(filename))
        stored = self.object_store.put_bytes(tenant_id, filename, data)
        preview = self._extract_text(data, filename, content_type)[:500]
        document = self.resources.create(
            "knowledge_documents",
            tenant_id,
            user_id,
            {
                "name": str(payload.get("name") or filename),
                "code": str(payload.get("code") or Path(filename).stem),
                "status": "uploaded",
                "parent_id": kb.id,
                "knowledge_base_id": kb.id,
                "config": {"embedding_model_id": (kb.config or {}).get("embedding_model_id"), "rerank_model_id": (kb.config or {}).get("rerank_model_id")},
                "spec": {
                    "filename": filename,
                    "content_type": content_type,
                    "uri": stored["uri"],
                    "sha256": stored["sha256"],
                    "size": stored["size"],
                },
                "input_payload": {"content_preview": preview, "size": stored["size"]},
            },
        )
        if payload.get("auto_index", True):
            self.parse_document(tenant_id, user_id, document.id)
            self.reindex_document(tenant_id, user_id, document.id)
            document = self.resources.get("knowledge_documents", tenant_id, document.id)
        return ResourceService.to_dict(document)

    def delete_document(self, tenant_id: str, user_id: str, document_id: str) -> dict[str, Any]:
        document = self.resources.get("knowledge_documents", tenant_id, document_id)
        store = self._store(tenant_id, document.knowledge_base_id)
        store.delete_documents([doc_id for doc_id, doc in store.documents.items() if (doc.get("metadata") or {}).get("document_id") == document_id])
        deleted = self.resources.delete("knowledge_documents", tenant_id, user_id, document_id)
        return {"status": deleted["status"], "id": document_id, "deleted_vectors": True}

    def parse_document(self, tenant_id: str, user_id: str, document_id: str) -> dict[str, Any]:
        document = self.resources.get("knowledge_documents", tenant_id, document_id)
        text = self._document_text(document)
        chunks = self._chunk_text(text, int((document.config or {}).get("chunk_size") or 800))
        existing_chunks, _ = self.resources.list("knowledge_chunks", tenant_id, 1, 1000, {"parent_id": document.id})
        for chunk in existing_chunks:
            self.resources.delete("knowledge_chunks", tenant_id, user_id, chunk.id)
        job = self.resources.create(
            "knowledge_parse_jobs",
            tenant_id,
            user_id,
            {
                "name": f"parse {document.name}",
                "status": "completed",
                "parent_id": document.id,
                "knowledge_base_id": document.knowledge_base_id,
                "output_payload": {"chunk_count": len(chunks)},
            },
        )
        created_chunks = []
        for index, chunk in enumerate(chunks):
            row = self.resources.create(
                "knowledge_chunks",
                tenant_id,
                user_id,
                {
                    "name": f"{document.name} chunk {index + 1}",
                    "status": "parsed",
                    "parent_id": document.id,
                    "knowledge_base_id": document.knowledge_base_id,
                    "spec": {"index": index, "text": chunk, "token_count": len(chunk.split()), "metadata": {"filename": (document.spec or {}).get("filename", "")}},
                },
            )
            created_chunks.append(row.id)
        self.resources.update("knowledge_documents", tenant_id, user_id, document.id, {"status": "parsed"})
        return {"job_id": job.id, "document_id": document.id, "chunk_ids": created_chunks, "chunk_count": len(created_chunks), "status": "completed"}

    def reindex_document(self, tenant_id: str, user_id: str, document_id: str) -> dict[str, Any]:
        document = self.resources.get("knowledge_documents", tenant_id, document_id)
        kb = self.resources.get("knowledge_bases", tenant_id, document.knowledge_base_id)
        chunks, total = self.resources.list("knowledge_chunks", tenant_id, 1, 1000, {"parent_id": document.id})
        if total == 0:
            self.parse_document(tenant_id, user_id, document.id)
            chunks, total = self.resources.list("knowledge_chunks", tenant_id, 1, 1000, {"parent_id": document.id})
        dimensions = int((kb.config or {}).get("embedding_dimensions") or 128)
        embedding_model_id = str((kb.config or {}).get("embedding_model_id") or "")
        vector_documents: list[dict[str, Any]] = []
        for chunk in chunks:
            text = str((chunk.spec or {}).get("text") or "")
            vector = deterministic_embedding(text, dimensions)
            self.resources.require_embedding_dimensions(kb, vector)
            vector_id = f"{document.id}:{chunk.id}"
            metadata = {"tenant_id": tenant_id, "knowledge_base_id": kb.id, "document_id": document.id, "chunk_id": chunk.id}
            vector_documents.append({"id": vector_id, "text": text, "vector": vector, "metadata": metadata})
            self.resources.create(
                "knowledge_embeddings",
                tenant_id,
                user_id,
                {
                    "name": f"embedding {chunk.name}",
                    "status": "indexed",
                    "parent_id": chunk.id,
                    "knowledge_base_id": kb.id,
                    "model_id": embedding_model_id,
                    "spec": {"vector_id": vector_id, "dimensions": dimensions, "metadata": metadata},
                },
            )
        store = self._store(tenant_id, kb.id)
        store.delete_documents([doc_id for doc_id, doc in store.documents.items() if (doc.get("metadata") or {}).get("document_id") == document.id])
        store.add_documents(vector_documents)
        index = self.resources.create(
            "knowledge_indexes",
            tenant_id,
            user_id,
            {
                "name": f"index {kb.name}",
                "status": "active",
                "parent_id": kb.id,
                "knowledge_base_id": kb.id,
                "config": {"embedding_model_id": embedding_model_id, "embedding_dimensions": dimensions},
                "spec": {"document_id": document.id, "chunk_count": total, "vector_count": len(vector_documents), "store_path": str(self._store_path(tenant_id, kb.id))},
            },
        )
        self.resources.update("knowledge_documents", tenant_id, user_id, document.id, {"status": "indexed"})
        return {"index_id": index.id, "document_id": document.id, "chunk_count": total, "vector_count": len(vector_documents), "status": "completed"}

    def rebuild_index(self, tenant_id: str, user_id: str, kb_id: str) -> dict[str, Any]:
        kb = self.resources.get("knowledge_bases", tenant_id, kb_id)
        documents, total = self.resources.list("knowledge_documents", tenant_id, 1, 1000, {"knowledge_base_id": kb.id})
        store = self._store(tenant_id, kb.id)
        store.rebuild_index([])
        indexed = [self.reindex_document(tenant_id, user_id, document.id) for document in documents]
        job = self.resources.create(
            "knowledge_rebuild_jobs",
            tenant_id,
            user_id,
            {"name": f"rebuild {kb.name}", "status": "completed", "parent_id": kb.id, "knowledge_base_id": kb.id, "output_payload": {"documents": total, "indexed": indexed}},
        )
        return {"job_id": job.id, "knowledge_base_id": kb.id, "documents": total, "status": "completed", "indexed": indexed}

    def retrieve(self, tenant_id: str, user_id: str, kb_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        kb = self.resources.get("knowledge_bases", tenant_id, kb_id)
        query = str(payload.get("query") or payload.get("text") or "")
        if not query.strip():
            raise AppError(ErrorCode.VALIDATION_ERROR, "query is required", 422)
        top_k = int(payload.get("top_k") or 5)
        score_threshold = float(payload.get("score_threshold") or 0)
        dimensions = int((kb.config or {}).get("embedding_dimensions") or 128)
        metadata_filter = dict(payload.get("metadata_filter") or {})
        metadata_filter.update({"tenant_id": tenant_id, "knowledge_base_id": kb.id})
        query_vector = deterministic_embedding(query, dimensions)
        mode = str(payload.get("mode") or "hybrid")
        store = self._store(tenant_id, kb.id)
        if payload.get("auto_index", True):
            self._ensure_indexed(tenant_id, user_id, kb.id)
            store = self._store(tenant_id, kb.id)
        if mode == "vector":
            matches = store.similarity_search(query_vector, top_k=top_k, metadata_filter=metadata_filter)
        elif mode == "keyword":
            matches = store.keyword_search(query, top_k=top_k, metadata_filter=metadata_filter)
        else:
            matches = store.hybrid_search(query, query_vector, top_k=top_k, metadata_filter=metadata_filter)
        filtered = [item for item in matches if float(item.get("score", 0)) >= score_threshold]
        rerank_model_id = str(payload.get("rerank_model_id") or (kb.config or {}).get("rerank_model_id") or "")
        if payload.get("rerank") and rerank_model_id:
            filtered = self._rerank(query, filtered, top_k)
        log = self.resources.create(
            "knowledge_retrieval_logs",
            tenant_id,
            user_id,
            {
                "name": "knowledge retrieval",
                "status": "success",
                "knowledge_base_id": kb.id,
                "model_id": str((kb.config or {}).get("embedding_model_id") or ""),
                "input_payload": payload,
                "output_payload": {"count": len(filtered), "matches": self._summarize_matches(filtered)},
                "spec": {"embedding_model_id": (kb.config or {}).get("embedding_model_id"), "rerank_model_id": rerank_model_id, "mode": mode},
            },
        )
        return {"log_id": log.id, "knowledge_base_id": kb.id, "query": query, "matches": filtered, "total": len(filtered)}

    def build_context(self, tenant_id: str, user_id: str, kb_ids: list[str], query: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = payload or {}
        top_k = int(payload.get("top_k") or 4)
        per_kb = max(1, int(payload.get("per_kb") or top_k))
        sources: list[dict[str, Any]] = []
        retrieval_logs: list[str] = []
        for kb_id in [str(item) for item in kb_ids if str(item)]:
            result = self.retrieve(
                tenant_id,
                user_id,
                kb_id,
                {
                    "query": query,
                    "top_k": per_kb,
                    "mode": payload.get("mode", "hybrid"),
                    "score_threshold": payload.get("score_threshold", 0),
                    "rerank": payload.get("rerank", True),
                    "auto_index": payload.get("auto_index", True),
                },
            )
            retrieval_logs.append(str(result["log_id"]))
            for match in result.get("matches", []):
                metadata = dict(match.get("metadata") or {})
                sources.append(
                    {
                        "knowledge_base_id": kb_id,
                        "chunk_id": metadata.get("chunk_id") or match.get("id", ""),
                        "document_id": metadata.get("document_id", ""),
                        "title": metadata.get("filename") or metadata.get("document_name") or "",
                        "content": str(match.get("text") or match.get("content") or ""),
                        "score": float(match.get("rerank_score") or match.get("score") or 0),
                        "metadata": metadata,
                    }
                )
        sources.sort(key=lambda item: float(item.get("score") or 0), reverse=True)
        sources = sources[:top_k]
        lines = []
        for index, source in enumerate(sources, 1):
            title = source.get("title") or source.get("document_id") or source.get("chunk_id")
            lines.append(f"[K{index}] {title}\n{source.get('content', '')}")
        return {"sources": sources, "retrieval_log_ids": retrieval_logs, "context_text": "\n\n".join(lines), "total": len(sources)}

    def rerank(self, tenant_id: str, user_id: str, kb_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        kb = self.resources.get("knowledge_bases", tenant_id, kb_id)
        query = str(payload.get("query") or "")
        documents = payload.get("documents") or []
        ranked = self._rerank(query, [{"id": str(index), "text": str(document), "score": 0.0, "metadata": {}} for index, document in enumerate(documents)], int(payload.get("top_n") or len(documents) or 1))
        log = self.resources.create(
            "knowledge_retrieval_logs",
            tenant_id,
            user_id,
            {"name": "knowledge rerank", "status": "success", "knowledge_base_id": kb.id, "input_payload": payload, "output_payload": {"ranked": ranked}},
        )
        return {"log_id": log.id, "ranked": ranked}

    def stats(self, tenant_id: str, kb_id: str) -> dict[str, Any]:
        kb = self.resources.get("knowledge_bases", tenant_id, kb_id)
        _, documents = self.resources.list("knowledge_documents", tenant_id, 1, 1, {"knowledge_base_id": kb.id})
        _, chunks = self.resources.list("knowledge_chunks", tenant_id, 1, 1, {"knowledge_base_id": kb.id})
        _, retrievals = self.resources.list("knowledge_retrieval_logs", tenant_id, 1, 1, {"knowledge_base_id": kb.id})
        return {"knowledge_base_id": kb.id, "documents": documents, "chunks": chunks, "retrievals": retrievals, "vector_store": str(self._store_path(tenant_id, kb.id))}

    def _store(self, tenant_id: str, kb_id: str) -> LocalVectorStore:
        return LocalVectorStore(self._store_path(tenant_id, kb_id))

    def _store_path(self, tenant_id: str, kb_id: str) -> Path:
        return self.vector_root / tenant_id / f"{kb_id}.json"

    def _ensure_indexed(self, tenant_id: str, user_id: str, kb_id: str) -> None:
        documents, _ = self.resources.list("knowledge_documents", tenant_id, 1, 1000, {"knowledge_base_id": kb_id})
        for document in documents:
            if document.status != "indexed":
                self.reindex_document(tenant_id, user_id, document.id)

    @staticmethod
    def _safe_filename(filename: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", filename).strip("._") or "document.txt"

    @staticmethod
    def _payload_bytes(payload: dict[str, Any]) -> bytes:
        encoded = payload.get("content_base64") or payload.get("file_base64") or (payload.get("spec") or {}).get("content_base64")
        if encoded:
            return base64.b64decode(str(encoded), validate=True)
        raw = payload.get("content") or payload.get("text") or (payload.get("spec") or {}).get("content")
        if isinstance(raw, bytes):
            return raw
        return str(raw or "").encode("utf-8")

    @staticmethod
    def _chunk_text(text: str, chunk_size: int) -> list[str]:
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
        chunks: list[str] = []
        current = ""
        for paragraph in paragraphs or [text]:
            if len(current) + len(paragraph) + 1 <= chunk_size:
                current = f"{current}\n{paragraph}".strip()
            else:
                if current:
                    chunks.append(current)
                while len(paragraph) > chunk_size:
                    chunks.append(paragraph[:chunk_size])
                    paragraph = paragraph[chunk_size:]
                current = paragraph
        if current:
            chunks.append(current)
        return chunks

    @staticmethod
    def _document_text(document: Any) -> str:
        uri = (document.spec or {}).get("uri")
        if uri and Path(str(uri)).exists():
            path = Path(str(uri))
            return KnowledgeService._extract_text(path.read_bytes(), str((document.spec or {}).get("filename") or path.name), str((document.spec or {}).get("content_type") or ""))
        preview = (document.input_payload or {}).get("content_preview")
        if preview:
            return str(preview)
        raise AppError(ErrorCode.NOT_FOUND, "stored document content not found", 404)

    @staticmethod
    def _extract_text(data: bytes, filename: str, content_type: str) -> str:
        suffix = Path(filename).suffix.lower()
        if suffix in {".txt", ".md", ".markdown", ".csv"} or "text/" in content_type or "csv" in content_type:
            text = data.decode("utf-8", errors="ignore")
            if suffix == ".csv":
                rows = csv.reader(StringIO(text))
                return "\n".join(" ".join(cell.strip() for cell in row if cell.strip()) for row in rows)
            return text
        if suffix in {".html", ".htm"} or "html" in content_type:
            text = data.decode("utf-8", errors="ignore")
            return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text)).strip()
        if suffix == ".docx":
            return KnowledgeService._extract_docx(data)
        if suffix == ".xlsx":
            return KnowledgeService._extract_xlsx(data)
        if suffix == ".pdf" or "pdf" in content_type:
            return KnowledgeService._extract_pdf_text(data)
        return data.decode("utf-8", errors="ignore")

    @staticmethod
    def _content_type(filename: str) -> str:
        suffix = Path(filename).suffix.lower()
        return {
            ".txt": "text/plain",
            ".md": "text/markdown",
            ".markdown": "text/markdown",
            ".html": "text/html",
            ".htm": "text/html",
            ".csv": "text/csv",
            ".pdf": "application/pdf",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        }.get(suffix, "application/octet-stream")

    @staticmethod
    def _extract_docx(data: bytes) -> str:
        with zipfile.ZipFile(BytesIO(data)) as archive:
            xml = archive.read("word/document.xml")
        root = ElementTree.fromstring(xml)
        return "\n".join(node.text or "" for node in root.iter() if node.tag.endswith("}t") and node.text)

    @staticmethod
    def _extract_xlsx(data: bytes) -> str:
        with zipfile.ZipFile(BytesIO(data)) as archive:
            shared: list[str] = []
            if "xl/sharedStrings.xml" in archive.namelist():
                root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
                shared = [" ".join(text.text or "" for text in item.iter() if text.tag.endswith("}t")) for item in root]
            lines: list[str] = []
            for name in sorted(item for item in archive.namelist() if item.startswith("xl/worksheets/sheet") and item.endswith(".xml")):
                root = ElementTree.fromstring(archive.read(name))
                cells = []
                for cell in root.iter():
                    if not cell.tag.endswith("}c"):
                        continue
                    cell_type = cell.attrib.get("t")
                    value_node = next((child for child in cell if child.tag.endswith("}v")), None)
                    if value_node is None or value_node.text is None:
                        continue
                    if cell_type == "s" and shared:
                        cells.append(shared[int(value_node.text)])
                    else:
                        cells.append(value_node.text)
                if cells:
                    lines.append(" ".join(cells))
            return "\n".join(lines)

    @staticmethod
    def _extract_pdf_text(data: bytes) -> str:
        text = data.decode("latin-1", errors="ignore")
        literal_strings = re.findall(r"\(([^()]*)\)\s*Tj", text)
        array_strings = re.findall(r"\[(.*?)\]\s*TJ", text, flags=re.DOTALL)
        for array in array_strings:
            literal_strings.extend(re.findall(r"\(([^()]*)\)", array))
        extracted = " ".join(item.replace(r"\(", "(").replace(r"\)", ")") for item in literal_strings)
        return extracted or re.sub(r"[^A-Za-z0-9\u4e00-\u9fff.,;:!?()\s-]+", " ", text)

    @staticmethod
    def _rerank(query: str, matches: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        query_terms = {part.lower() for part in query.split() if part}
        ranked = []
        for item in matches:
            text_terms = {part.lower() for part in str(item.get("text", "")).split() if part}
            lexical = len(query_terms & text_terms) / max(1, len(query_terms))
            ranked.append({**item, "rerank_score": round(0.7 * lexical + 0.3 * float(item.get("score", 0)), 6)})
        return sorted(ranked, key=lambda row: float(row.get("rerank_score", 0)), reverse=True)[:top_k]

    @staticmethod
    def _summarize_matches(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{"id": item.get("id"), "score": item.get("score"), "metadata": item.get("metadata"), "text": str(item.get("text", ""))[:300]} for item in matches]
