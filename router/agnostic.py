"""
router/agnostic.py
------------------
POST /api/v1/agnostic/query  --  Delegates to the existing processor
(same FAISS-indexed collections as hybrid.py) so queries actually work
against uploaded documents.

The "agnostic" label is preserved for UI compatibility; internally this
is the same hybrid-search path with a HybridResponse-compatible schema.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request, Depends
from pydantic import BaseModel, Field

from config import config
from processor import processor
import storage as supabase_storage
from router.auth import get_current_user, UserRecord
from router.public_links import resolve_active_public_link_sources
from router.database_connections import resolve_active_database_connections

logger = logging.getLogger(__name__)
router = APIRouter(tags=["agnostic"])



# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------

class AgnosticQueryRequest(BaseModel):
    question: str = Field(..., min_length=1)

    # Kept for schema compatibility with HF Space / old clients — ignored locally.
    source: Optional[str] = Field(None)

    include_pdf_results:  Optional[bool] = True
    # Legacy: queries THIS app's own fixed database (app management data, not
    # a user-connected source). Kept for backward compatibility only.
    include_db_results:   Optional[bool] = False
    include_chat_results: Optional[bool] = False
    include_public_links: Optional[bool] = False
    # Queries the user's own external database connection(s) set up in
    # Sources > Database — the actual "database as a knowledge source" path.
    include_external_db:  Optional[bool] = False
    llm_provider:         Optional[str]  = None
    llm_model:            Optional[str]  = None
    source_mode:          Optional[str]  = None

    # Optional collection selectors (defaults to "all")
    pdf_collection_ids:      Optional[List[str]] = None
    chat_collection_ids:     Optional[List[str]] = None
    public_link_ids:         Optional[List[str]] = None
    external_db_connection_ids: Optional[List[str]] = None


# ---------------------------------------------------------------------------
# Response schema  (HybridResponse-compatible)
# ---------------------------------------------------------------------------

class PdfSourceDetail(BaseModel):
    file_name:        str
    collection_id:    str
    page:             Optional[int]   = None
    relevance_score:  Optional[float] = None
    content_preview:  Optional[str]   = None
    file_url:         Optional[str]   = None
    page_url:         Optional[str]   = None
    search_text:      Optional[str]   = None


class AgnosticQueryResponse(BaseModel):
    answer:               str
    model_used:           str
    pdf_sources:          List[str]
    pdf_sources_detailed: List[PdfSourceDetail]
    db_results:           Dict[str, Any]
    chat_results:         List[Any]
    processing_time:      float
    search_terms:         List[str]
    target_tables:        List[str]
    source_type:          str = "Indexed Collections"
    retrieved_count:      int = 0


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

def _describe_source_type(has_public_links: bool, has_external_db: bool) -> str:
    extras = []
    if has_public_links:
        extras.append("Realtime Public Links")
    if has_external_db:
        extras.append("Realtime Database Connections")
    if not extras:
        return "Indexed Collections"
    return "Indexed Collections + " + " + ".join(extras)


@router.post("/agnostic/query", response_model=AgnosticQueryResponse)
async def agnostic_query(
    req: AgnosticQueryRequest,
    request: Request,
    user: UserRecord = Depends(get_current_user),
):
    start_time = datetime.now()
    logger.info("agnostic_query: question=%r user=%s", req.question, user.user_id)
    base_url = str(request.base_url).rstrip('/')
    is_admin = user.role == "admin"

    try:
        # Resolve collections — scoped to what this user is allowed to see.
        # Admins can reach every PDF collection; everyone else only their own.
        allowed_pdf_ids = set(await asyncio.to_thread(
            supabase_storage.list_collection_ids_for_user, user.user_id, is_admin
        ))
        if req.pdf_collection_ids:
            pdf_collection_ids = [cid for cid in req.pdf_collection_ids if cid in allowed_pdf_ids]
        elif req.include_pdf_results:
            pdf_collection_ids = list(allowed_pdf_ids)
        else:
            pdf_collection_ids = []
        logger.info("agnostic_query: %d PDF collection(s) accessible to user", len(pdf_collection_ids))

        # "Meta/help" questions (about the app itself, e.g. "apa yang bisa
        # dilakukan disini") short-circuit here — deliberately BEFORE running
        # hybrid_search/generate_hybrid_answer, since there's no document to
        # ground an LLM answer in. Answer is built from real state (this
        # user's own collections), never LLM-generated. See
        # Processor.is_meta_help_query / build_meta_help_answer.
        if processor.is_meta_help_query(req.question):
            all_rows = await asyncio.to_thread(supabase_storage.list_collections)
            titles = [
                row.get("title") or (row.get("file_names") or [""])[0] or row["collection_id"]
                for row in all_rows
                if row.get("collection_id") in allowed_pdf_ids
            ]
            help_answer = processor.build_meta_help_answer(len(allowed_pdf_ids), titles)
            elapsed = (datetime.now() - start_time).total_seconds()
            return AgnosticQueryResponse(
                answer=help_answer,
                model_used="system/meta-help",
                pdf_sources=[],
                pdf_sources_detailed=[],
                db_results={},
                chat_results=[],
                processing_time=elapsed,
                search_terms=[req.question],
                target_tables=[],
                source_type="System",
                retrieved_count=0,
            )

        # Chat is admin-only (business rule — non-admins never search it,
        # regardless of what flags/ids the client sends).
        chat_collection_ids = req.chat_collection_ids if is_admin else []
        if is_admin and not chat_collection_ids and req.include_chat_results:
            chat_collection_ids = processor.get_all_chat_collections()

        should_search_pdfs = bool(req.include_pdf_results) and bool(pdf_collection_ids)
        should_search_db   = bool(req.include_db_results)
        should_search_chat = is_admin and bool(req.include_chat_results) and bool(chat_collection_ids)
        public_link_sources: List[Dict[str, Any]] = []
        if bool(req.include_public_links):
            public_link_sources = await resolve_active_public_link_sources(
                req.public_link_ids, user_id=user.user_id, is_admin=is_admin
            )
        should_search_public_links = bool(req.include_public_links) and bool(public_link_sources)

        # Database connections are admin-only (business rule), same as chat.
        external_db_connections: List[Dict[str, Any]] = []
        if is_admin and bool(req.include_external_db):
            external_db_connections = await resolve_active_database_connections(req.external_db_connection_ids)
        should_search_external_db = is_admin and bool(req.include_external_db) and bool(external_db_connections)

        # Run hybrid search against pre-built FAISS indexes
        hybrid_results = await asyncio.to_thread(
            processor.hybrid_search,
            req.question,
            pdf_collection_ids or [],
            should_search_chat,
            should_search_pdfs,
            should_search_db,
            chat_collection_ids or [],
            should_search_public_links,
            public_link_sources,
            should_search_external_db,
            external_db_connections,
        )

        # Generate answer
        answer_result = await asyncio.to_thread(
            processor.generate_hybrid_answer,
            hybrid_results,
            req.question,
            req.llm_provider,
            req.llm_model,
        )

        if isinstance(answer_result, tuple) and len(answer_result) >= 2:
            answer, model_used = answer_result[0], answer_result[1]
        else:
            answer     = str(answer_result)
            model_used = req.llm_model or config.default_llm_model

        # Map PDF docs -> response fields
        pdf_sources: List[str] = []
        pdf_sources_detailed: List[PdfSourceDetail] = []
        for doc in hybrid_results.get("pdf_documents", []):
            meta  = getattr(doc, "metadata", {})
            fname = meta.get("source", "Unknown")
            page  = meta.get("page")
            pdf_sources.append(f"{fname} (Halaman {page})" if page else fname)

            try:
                page_num = int(page) if page is not None else None
            except (ValueError, TypeError):
                page_num = None

            collection_id = meta.get("collection_id", "")
            file_url = f"{base_url}/api/v1/files/{collection_id}/{fname}" if collection_id else None
            page_url = f"{file_url}#page={page_num}" if file_url and page_num else file_url
            content_text = doc.page_content.strip() if hasattr(doc, "page_content") else ""

            pdf_sources_detailed.append(PdfSourceDetail(
                file_name=fname,
                collection_id=collection_id,
                page=page_num,
                relevance_score=meta.get("similarity_score", 0.0),
                content_preview=(doc.page_content[:300]
                                 if hasattr(doc, "page_content") else ""),
                file_url=file_url,
                page_url=page_url,
                search_text=' '.join(content_text.split()[:15]),
            ))

        for doc in hybrid_results.get("public_link_documents", []):
            meta = getattr(doc, "metadata", {})
            fname = meta.get("source", "Unknown")
            title = meta.get("public_link_title")
            display_name = f"{title} / {fname}" if title else fname
            pdf_sources.append(display_name)
            item_url = meta.get("item_url")
            content_text = doc.page_content.strip() if hasattr(doc, "page_content") else ""
            pdf_sources_detailed.append(PdfSourceDetail(
                file_name=display_name,
                collection_id=meta.get("collection_id", "public-link"),
                relevance_score=meta.get("similarity_score", 0.0),
                content_preview=(doc.page_content[:300] if hasattr(doc, "page_content") else ""),
                file_url=item_url,
                page_url=item_url,
                search_text=' '.join(content_text.split()[:15]),
            ))

        for doc in hybrid_results.get("external_db_documents", []):
            meta = getattr(doc, "metadata", {})
            table_name = meta.get("source", "Unknown")
            label = meta.get("external_db_label", "Database")
            display_name = f"{label} / {table_name}"
            pdf_sources.append(display_name)
            pdf_sources_detailed.append(PdfSourceDetail(
                file_name=display_name,
                collection_id=meta.get("collection_id", "external-db"),
                relevance_score=meta.get("similarity_score", 0.0),
                content_preview=(doc.page_content[:300] if hasattr(doc, "page_content") else ""),
            ))

        chat_results = []
        for doc in hybrid_results.get("chat_documents", []):
            meta = getattr(doc, "metadata", {})
            chat_results.append({
                "source":          meta.get("source", "Unknown"),
                "platform":        meta.get("platform", "chat"),
                "relevance_score": meta.get("similarity_score", 0),
                "content_preview": (doc.page_content[:200]
                                    if hasattr(doc, "page_content") else ""),
            })

        elapsed = (datetime.now() - start_time).total_seconds()

        return AgnosticQueryResponse(
            answer=answer,
            model_used=model_used,
            pdf_sources=pdf_sources,
            pdf_sources_detailed=pdf_sources_detailed,
            db_results=hybrid_results.get("database_results", {}),
            chat_results=chat_results,
            processing_time=elapsed,
            search_terms=hybrid_results.get("search_terms", [req.question]),
            target_tables=hybrid_results.get("target_tables", []),
            source_type=_describe_source_type(should_search_public_links, should_search_external_db),
            retrieved_count=len(pdf_sources) + len(chat_results),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("agnostic_query failed")
        raise HTTPException(status_code=500, detail=str(e))