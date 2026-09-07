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
from router.payment import (
    log_token_usage,
    enforce_rate_limit,
    enforce_member_allocation,
    enforce_plan_limit,
    resolve_workspace_id,
    get_workspace_lock,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["agnostic"])

# MS-237 poin 1: hard cap on the client-sent conversation-history window,
# applied here regardless of what the client actually sends — a client that
# sends 200 messages must not be able to balloon the prompt/token cost. The
# window is counted in *chats* (one question plus the answer that followed
# it), so 5 chats is up to ~10 messages; MAX_MEMORY_MESSAGES is the backstop
# for a caller that pads one chat with a hundred assistant turns.
MAX_MEMORY_CHATS = 5
MAX_MEMORY_MESSAGES = MAX_MEMORY_CHATS * 4
MAX_MEMORY_CHARS = 600


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------

class MemoryTurn(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class AgnosticQueryRequest(BaseModel):
    question: str = Field(..., min_length=1)

    # Kept for schema compatibility with HF Space / old clients — ignored locally.
    source: Optional[str] = Field(None)

    # MS-237: which session this question belongs to (currently unused
    # server-side — the client sends `memory` directly instead of us looking
    # it up — kept so the contract already has it once that moves server-side).
    session_id: Optional[str] = None
    # Previous 5 messages so a follow-up like "ringkas semua di atas" has
    # something to resolve against. Re-clamped below regardless of what the
    # client sends.
    memory: Optional[List[MemoryTurn]] = None

    # MS-252: one-shot skill invocation — the uploaded Skill (see
    # router/skills.py) whose `instruction` should shape THIS answer only.
    # Resolved + authorization-checked server-side via
    # storage.get_skill_for_user; an unresolved id is ignored, never an
    # error (the client already cleared its own armed state either way).
    skill_id: Optional[str] = None

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

        # Unrecognized "/" command — safety net for requests that bypass the
        # chat-ui command menu (which already intercepts unmatched slash
        # input client-side). Short-circuits before hybrid_search/
        # generate_hybrid_answer, same reasoning as meta-help above: a
        # mistyped command has no document to ground an LLM answer in.
        elif processor.is_unknown_slash_command(req.question):
            elapsed = (datetime.now() - start_time).total_seconds()
            return AgnosticQueryResponse(
                answer=processor.build_unknown_command_answer(req.question),
                model_used="system/unknown-command",
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

        # No source selected at all — safety net for requests that bypass
        # the chat-ui toggle guard (regenerate, direct API calls). Without
        # this, hybrid_search silently returns empty results and
        # generate_hybrid_answer still calls the LLM with an empty context,
        # producing a misleading "not found in documents" answer instead of
        # directing the user to pick a source.
        elif not any([
            req.include_pdf_results, req.include_db_results,
            req.include_chat_results, req.include_public_links,
            req.include_external_db,
        ]):
            elapsed = (datetime.now() - start_time).total_seconds()
            return AgnosticQueryResponse(
                answer=(
                    "Pilih dulu minimal satu sumber (PDF, Database, Chat, atau Drive) "
                    "sebelum bertanya, biar jawabannya bisa saya dasarkan dari data kamu."
                ),
                model_used="system/no-source-selected",
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

        # Past this point every branch actually calls the LLM (unlike the
        # free system-answer branches above, which must stay reachable even
        # for a capped user), so this is where rate limiting (MS-248) has to
        # sit — checking any earlier would deny help text that costs nothing.
        #
        # Held for the rest of this request (through the LLM call and the
        # token-usage log below), keyed by workspace (not just this user):
        # enforce_plan_limit and enforce_member_allocation both check state
        # shared across the whole team, so two different members racing
        # concurrently must serialize against each other too, not just
        # against their own other requests — see get_workspace_lock.
        workspace_id = await asyncio.to_thread(resolve_workspace_id, user)
        async with get_workspace_lock(workspace_id):
            await asyncio.to_thread(enforce_rate_limit, user.user_id)
            await asyncio.to_thread(enforce_plan_limit, user)
            await asyncio.to_thread(enforce_member_allocation, user)
            return await _run_metered_query(
                req, user, is_admin, allowed_pdf_ids, pdf_collection_ids, base_url, start_time
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("agnostic_query failed")
        raise HTTPException(status_code=500, detail=str(e))


async def _run_metered_query(
    req: AgnosticQueryRequest,
    user: UserRecord,
    is_admin: bool,
    allowed_pdf_ids: set,
    pdf_collection_ids: List[str],
    base_url: str,
    start_time: datetime,
) -> "AgnosticQueryResponse":
    """The token-costing half of agnostic_query — everything from source
    resolution through the LLM call and usage logging, run while the
    caller holds that user's rate-limit lock. Split out from
    agnostic_query so that lock's scope is a plain, visible function call
    rather than a large indented block sharing the outer try/except."""
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

    # MS-237: clamp the client's conversation-history window before it
    # ever reaches the prompt — see MAX_MEMORY_* above. Cut at the start
    # of the 5th-from-last question rather than at a raw message count,
    # so every remembered question keeps the answer that belongs to it.
    memory_payload: Optional[List[Dict[str, str]]] = None
    if req.memory:
        question_at = [i for i, m in enumerate(req.memory) if m.role == "user"]
        start = (
            question_at[-MAX_MEMORY_CHATS]
            if len(question_at) > MAX_MEMORY_CHATS
            else 0
        )
        memory_payload = [
            {"role": m.role, "content": m.content[:MAX_MEMORY_CHARS]}
            for m in req.memory[start:][-MAX_MEMORY_MESSAGES:]
        ]

    # MS-252: resolve the one-shot skill (if any) to its instruction text.
    # get_skill_for_user already enforces personal/team visibility — an id
    # that doesn't exist or isn't visible to this user just yields None, so
    # a deleted/foreign skill_id silently falls back to normal chat instead
    # of erroring the whole request.
    skill_instruction: Optional[str] = None
    if req.skill_id:
        skill_row = await asyncio.to_thread(
            supabase_storage.get_skill_for_user, req.skill_id, user.user_id, is_admin
        )
        if skill_row:
            skill_instruction = skill_row["instruction"]
        else:
            logger.warning(
                "agnostic_query: skill_id=%s not visible to user=%s — ignoring",
                req.skill_id, user.user_id,
            )

    # Generate answer
    answer_result = await asyncio.to_thread(
        processor.generate_hybrid_answer,
        hybrid_results,
        req.question,
        req.llm_provider,
        req.llm_model,
        memory_payload,
        skill_instruction,
    )

    if isinstance(answer_result, tuple) and len(answer_result) >= 2:
        answer, model_used = answer_result[0], answer_result[1]
        answer_metadata = (
            answer_result[2] if len(answer_result) >= 3 and isinstance(answer_result[2], dict) else {}
        )
    else:
        answer     = str(answer_result)
        model_used = req.llm_model or config.default_llm_model
        answer_metadata = {}

    # Best-effort token metering (MS-248) — never let a logging hiccup
    # break a chat response that already succeeded. Run in a thread: this
    # does blocking DB I/O and we're holding the caller's rate-limit lock,
    # so a blocking call here would stall every other coroutine waiting on
    # that same lock (including this same user's next request), not just
    # this one.
    try:
        tokens_consumed = answer_metadata.get("total_tokens", 0)
        if tokens_consumed:
            await asyncio.to_thread(log_token_usage, user.user_id, tokens_consumed)
    except Exception:
        logger.warning("agnostic_query: failed to log token usage", exc_info=True)

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