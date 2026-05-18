from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.domain_router import add_action_route, add_crud_routes, add_list_route

router = APIRouter(tags=["knowledge"])

add_crud_routes(router, table="knowledge_bases", prefix="/knowledge-bases", permission="knowledge")

for method, path, table, action, output in [
    ("post", "/knowledge-bases/{kb_id}/permissions", "knowledge_bases", "permissions", "knowledge_base_permissions"),
    ("post", "/knowledge-bases/{kb_id}/documents", "knowledge_bases", "upload_document", "knowledge_documents"),
    ("get", "/knowledge-bases/{kb_id}/documents", "knowledge_documents", "documents", None),
    ("get", "/knowledge-documents/{document_id}", "knowledge_documents", "document", None),
    ("delete", "/knowledge-documents/{document_id}", "knowledge_documents", "delete", None),
    ("post", "/knowledge-documents/{document_id}/parse", "knowledge_documents", "parse", "knowledge_parse_jobs"),
    ("get", "/knowledge-documents/{document_id}/parse-status", "knowledge_parse_jobs", "parse_status", None),
    ("get", "/knowledge-documents/{document_id}/chunks", "knowledge_chunks", "chunks", None),
    ("post", "/knowledge-documents/{document_id}/reindex", "knowledge_documents", "reindex", "knowledge_rebuild_jobs"),
    ("post", "/knowledge-bases/{kb_id}/rebuild-index", "knowledge_bases", "rebuild_index", "knowledge_rebuild_jobs"),
    ("post", "/knowledge-bases/{kb_id}/retrieve", "knowledge_bases", "retrieve", "knowledge_retrieval_logs"),
    ("post", "/knowledge-bases/{kb_id}/rerank", "knowledge_bases", "rerank", "knowledge_retrieval_logs"),
    ("get", "/knowledge-bases/{kb_id}/retrieval-logs", "knowledge_retrieval_logs", "retrieval_logs", None),
    ("get", "/knowledge-bases/{kb_id}/stats", "knowledge_indexes", "stats", None),
]:
    if method == "get":
        add_list_route(router, method=method, path=path, table=table, permission="knowledge")
    else:
        add_action_route(router, method=method, path=path, table=table, permission="knowledge", action=action, output_table=output)

