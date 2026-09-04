# models.py
from pydantic import BaseModel, field_validator
from typing import List, Optional, Dict, Any, Union
from enum import Enum
from datetime import datetime

class SearchType(str, Enum):
    UNSTRUCTURED = "unstructured"
    STRUCTURED = "structured"
    HYBRID = "hybrid"

class ChatPlatform(str, Enum):
    WHATSAPP = "whatsapp"
    TEAMS = "teams"
    SLACK = "slack"
    TELEGRAM = "telegram"
    GENERIC = "generic"

class QueryIntent(str, Enum):
    COUNT = "count"
    SEARCH = "search"
    LIST = "list"
    AGGREGATE = "aggregate"
    SHOW_TABLES = "show_tables" 
    SHOW_SCHEMA = "show_schema"
    HELP = "help"
    UNKNOWN = "unknown"

class QueryRequest(BaseModel):
    question: str
    collection_id: Optional[str] = None  # Now optional
    include_sources: bool = True
    # search_type: SearchType = SearchType.UNSTRUCTURED
class StructuredQueryRequest(BaseModel):
    question: str
    table_name: Optional[str] = None


class UploadResponse(BaseModel):
    collection_id: str
    file_count: int
    status: str
    file_names: Optional[List[str]] = None
    title: Optional[str] = None


class UploadFromUrlRequest(BaseModel):
    url: str
    title: Optional[str] = None


class UploadFromUrlsRequest(BaseModel):
    urls: List[str]
    title: Optional[str] = None


class DriveFolderRequest(BaseModel):
    url: str
    recursive: Optional[bool] = True
    max_depth: Optional[int] = 5


class DriveFolderItem(BaseModel):
    id: str
    name: str
    url: str
    item_type: str


class DriveFolderItemsResponse(BaseModel):
    folder_id: str
    files: List[DriveFolderItem]
    folders: List[DriveFolderItem]
    count: int


class CreatePublicLinkRequest(BaseModel):
    title: Optional[str] = None
    url: str
    item_urls: Optional[List[str]] = None


class SetPublicLinkActiveRequest(BaseModel):
    link_id: str
    active: bool


class PublicLinkItem(BaseModel):
    id: str
    name: str
    url: str
    item_type: str


class PublicLinkSource(BaseModel):
    link_id: str
    workspace_id: Optional[str] = None
    title: str
    url: str
    status: str
    item_count: int
    created_at: Union[str, datetime]
    items: List[PublicLinkItem] = []


class PublicLinksResponse(BaseModel):
    links: List[PublicLinkSource]
    count: int


class CreateDatabaseConnectionRequest(BaseModel):
    label: Optional[str] = None
    url: str


class SetDatabaseConnectionActiveRequest(BaseModel):
    connection_id: str
    active: bool


class DbColumnInfo(BaseModel):
    name: str
    type: str
    nullable: bool = True
    primary_key: bool = False


class DbTableInfo(BaseModel):
    name: str
    row_count: Optional[int] = None
    columns: List[DbColumnInfo] = []


class DatabaseConnectionSource(BaseModel):
    connection_id: str
    workspace_id: Optional[str] = None
    label: str
    url: str
    status: str
    table_count: int
    created_at: Union[str, datetime]
    tables: List[DbTableInfo] = []


class DatabaseConnectionsResponse(BaseModel):
    connections: List[DatabaseConnectionSource]
    count: int


class CreateCheckoutSessionRequest(BaseModel):
    plan_id: str


class CheckoutSessionResponse(BaseModel):
    checkout_url: str
    payment_id: str


class PaymentRecord(BaseModel):
    payment_id: str
    plan_id: str
    amount: int
    currency: str
    status: str
    created_at: Union[str, datetime]


class PaymentResponse(BaseModel):
    payment: PaymentRecord


# ===================== TOKEN USAGE & ALLOCATION (MS-248) =====================
# Workspace-level subscription usage plus, on top of it, per-member token
# allocations that an admin distributes out of the workspace's token_limit
# (see router/payment.py for how these are computed/persisted).

class SubscriptionUsage(BaseModel):
    plan_name: str
    subscription_status: str  # active | expired | none
    token_limit: int
    token_used: int
    token_remaining: int
    period_start: Union[str, datetime]
    period_end: Union[str, datetime]
    next_reset_date: Optional[Union[str, datetime]] = None
    # True once the admin has cancelled — access still runs until period_end
    # (already paid for), it just won't be treated as renewable after that.
    cancel_at_period_end: bool = False
    # False for the synthetic Free-tier plan (nobody's paid — see
    # _get_latest_plan_window's fallback branch) — there's no purchase on
    # file to cancel, so the frontend should hide cancel/resume for it.
    is_paid: bool = True


class MemberTokenUsage(BaseModel):
    user_id: str
    email: str
    allocated_tokens: int
    used_tokens: int
    remaining_tokens: int
    usage_percent: float


class MyMemberUsageResponse(BaseModel):
    usage: Optional[MemberTokenUsage] = None


class MembersUsageResponse(BaseModel):
    subscription: Optional[SubscriptionUsage] = None
    members: List[MemberTokenUsage]
    unallocated_tokens: int


class UpdateMemberAllocationRequest(BaseModel):
    user_id: str
    allocated_tokens: int


class UpdateMemberAllocationResponse(BaseModel):
    member: MemberTokenUsage
    unallocated_tokens: int


# Flat, plan-independent safety-net rate limit (same for every user) — see
# router/payment.py::_get_rate_limit_status. Separate from SubscriptionUsage
# / MemberTokenUsage above, which track the per-plan/per-member allocation.
class RateLimitStatus(BaseModel):
    used_tokens: int
    cap_tokens: int
    window_hours: float
    blocked: bool
    reset_at: Optional[Union[str, datetime]] = None


# "Request more tokens" (MS-248 follow-up) — in-app only (polled, not real
# push). A member who hit their admin-assigned cap can ask for more; the
# admin sees pending ones in the Billing tab, raises the cap themselves via
# the existing allocation editor, then dismisses the request.
class CreateTokenRequestRequest(BaseModel):
    message: Optional[str] = None


class TokenRequestRecord(BaseModel):
    request_id: str
    user_id: str
    email: str
    message: Optional[str] = None
    status: str  # pending | resolved
    created_at: Union[str, datetime]


class TokenRequestResponse(BaseModel):
    request: TokenRequestRecord


class TokenRequestsResponse(BaseModel):
    requests: List[TokenRequestRecord]
    pending_count: int


class QAResponse(BaseModel):
    answer: str
    sources: List[str]  # Now includes collection IDs
    collection_id: str   # "all_collections" when searching globally
    processing_time: float
    search_type: SearchType


class CollectionInfo(BaseModel):
    collection_id: str
    document_count: int
    created_at: Union[str, datetime]
    file_names: List[str]
    title: Optional[str] = None
    status: str = "active"
    owner_id: Optional[str] = None


class SetPdfCollectionActiveRequest(BaseModel):
    collection_id: str
    active: bool

class DatabaseResult(BaseModel):
    table: str
    data: List[Dict[str, Any]]
    record_count: int
    avg_relevance_score: Optional[float] = None  # Average score dari search results


class PdfSourceInfo(BaseModel):
    """PDF source with URL for direct access"""
    file_name: str
    collection_id: str
    page: Optional[int] = None
    relevance_score: Optional[float] = None
    content_preview: Optional[str] = None
    file_url: Optional[str] = None  # URL to access the PDF
    page_url: Optional[str] = None  # URL with page parameter for direct jump
    search_text: Optional[str] = None  # Text snippet for highlighting/searching in PDF viewer


# models.py - Add these fields to HybridQueryRequest
class HybridQueryRequest(BaseModel):
    question: str
    collection_id: Optional[str] = None  # DEPRECATED: use pdf_collection_ids instead
    
    # Collection Selection (Optional - if not provided, searches all)
    pdf_collection_ids: Optional[List[str]] = None  # Specific PDF collections to search
    chat_collection_ids: Optional[List[str]] = None  # Specific chat collections to search
    public_link_ids: Optional[List[str]] = None  # Specific Public Link sources to search
    
    include_pdf_results: bool = True
    include_db_results: bool = True
    include_chat_results: bool = True  # NEW: Search in chat logs?
    include_public_links: bool = False
    source_mode: Optional[str] = None  # "pdf" | "chat" | "database" | "public_link" | "mixed" | "none"
    
    # LLM Selection (optional - defaults to config if not provided)
    llm_provider: Optional[str] = None  # "huggingface", "ollama", "gemini"
    llm_model: Optional[str] = None     # specific model name

    # Skill selection (optional) - see GapAnalysisRequest for the dedicated
    # gap-analysis flow. This lets normal chat scope itself to a skill
    # context (e.g. the "meta/help" intent) without changing default behavior.
    skill_id: Optional[str] = None

class HybridResponse(BaseModel):
    answer: str
    pdf_sources: List[str]  # Keep for backward compatibility
    pdf_sources_detailed: Optional[List[PdfSourceInfo]] = None  # NEW: Detailed PDF sources with URLs
    db_results: Dict[str, Any]
    chat_results: Optional[List[Dict[str, Any]]] = None  # NEW: Chat search results
    processing_time: float
    search_terms: List[str]
    target_tables: Optional[List[str]] = None  # Tables that were searched (smart routing)
    
    # Model info - tells user which model generated this response
    model_used: str  # e.g., "huggingface/google/flan-t5-base"
    available_models: Optional[Dict[str, List[str]]] = None  # Only returned on first request or error

    # answer: str
    # pdf_sources: List[str]
    # db_results: Dict[str, DatabaseResult]  # FIXED: Dict bukan List
    # processing_time: float
    # search_terms: List[str]

class SourceInfo(BaseModel):
    type: str
    source: str
    confidence: Optional[float] = None
    preview: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class StructuredQueryResponse(BaseModel):
    answer: str
    data: List[Dict[str, Any]]
    intent: QueryIntent
    table_used: str
    sql_query: Optional[str] = None
    processing_time: float

class JoinQueryRequest(BaseModel):
    question: str
    join_type: Optional[str] = "auto"  # auto, inner, left, cross
    limit: int = 20

class JoinQueryResponse(BaseModel):
    answer: str
    data: List[Dict[str, Any]]
    tables_used: List[str]
    join_conditions: List[Dict[str, str]]
    sql_query: Optional[str] = None
    processing_time: float

class TableRelationship(BaseModel):
    table1: str
    table2: str
    join_condition: str
    relationship_type: str 


# ===================== CHAT MODELS =====================

class ChatMessage(BaseModel):
    """Single chat message parsed from export file"""
    message_id: Optional[str] = None
    sender: str
    timestamp: datetime
    content: str
    platform: ChatPlatform = ChatPlatform.WHATSAPP
    thread_id: Optional[str] = None
    conversation_id: Optional[str] = None
    raw_line: Optional[str] = None  # Original line for debugging


class ChatCollection(BaseModel):
    """Metadata for an uploaded chat collection"""
    collection_id: str
    platform: ChatPlatform
    file_name: str
    message_count: int
    date_range: Optional[Dict[str, str]] = None  # {"start": ..., "end": ...}
    participants: List[str]
    created_at: datetime
    status: str = "active"


class SetChatCollectionActiveRequest(BaseModel):
    collection_id: str
    active: bool


# ===================== TELEGRAM MODELS =====================
# Live connection (Telethon/MTProto login), not a file upload — see
# router/telegram.py. A connection holds one encrypted user session; a
# connection can have multiple "selected chats", each of which becomes its
# own searchable ChatCollection (platform=telegram) once synced.

class TelegramDialog(BaseModel):
    dialog_id: str
    title: str
    type: str  # "user" | "group" | "channel"
    participants_count: Optional[int] = None


class TelegramDialogsResponse(BaseModel):
    dialogs: List[TelegramDialog]
    count: int


class TelegramSelectedChat(BaseModel):
    dialog_id: str
    title: str
    type: str
    chat_collection_id: Optional[str] = None
    message_count: Optional[int] = None
    status: str = "active"
    last_synced_at: Optional[Union[str, datetime]] = None


class TelegramConnectionSource(BaseModel):
    connection_id: str
    label: str
    phone_masked: str
    status: str
    created_at: Union[str, datetime]
    selected_chats: List[TelegramSelectedChat] = []


class TelegramConnectionsResponse(BaseModel):
    connections: List[TelegramConnectionSource]
    count: int


class TelegramConnectStartRequest(BaseModel):
    # api_id/api_hash identify the Telegram *application* making the request
    # (from https://my.telegram.org/apps) — entered here per-connection rather
    # than shared via server config, so different admins can each bring their
    # own app registration.
    api_id: int
    api_hash: str
    phone: str
    label: Optional[str] = None


class TelegramConnectStartResponse(BaseModel):
    flow_id: str
    phone: str


class TelegramConnectVerifyRequest(BaseModel):
    flow_id: str
    code: str
    password: Optional[str] = None  # only needed if the account has 2FA enabled


class TelegramConnectVerifyResponse(BaseModel):
    status: str  # "connected" | "password_required"
    connection: Optional[TelegramConnectionSource] = None


class TelegramSyncRequest(BaseModel):
    dialog_ids: List[str]
    message_limit: int = 2000


class TelegramSyncResult(BaseModel):
    dialog_id: str
    title: str
    chat_collection_id: str
    message_count: int
    status: str  # "success" | "error"
    error: Optional[str] = None


class TelegramSyncResponse(BaseModel):
    results: List[TelegramSyncResult]


class SetTelegramConnectionActiveRequest(BaseModel):
    connection_id: str
    active: bool


class ChatUploadResponse(BaseModel):
    """Response after uploading chat file"""
    collection_id: str
    file_name: str
    platform: str
    message_count: int
    participants: List[str]
    date_range: Optional[Dict[str, str]] = None
    status: str


class ChatSearchResult(BaseModel):
    """Single result from chat search"""
    message: ChatMessage
    relevance_score: float
    context_messages: Optional[List[ChatMessage]] = None  # Surrounding messages for context

# models.py - tambahkan enhanced response model
class EnhancedHybridResponse(BaseModel):
    answer: str
    answer_metadata: Dict[str, Any]
    pdf_sources: List[str]
    pdf_sources_detailed: Optional[List[PdfSourceInfo]] = None
    db_results: Dict[str, Any]
    chat_results: Optional[List[Dict[str, Any]]] = None
    processing_time: float
    search_analysis: Dict[str, Any]
    merged_results_preview: Optional[List[Dict[str, Any]]] = None
    conflicts: Optional[List[Dict[str, Any]]] = None
    model_used: str
    confidence_score: float  # Overall confidence 0-1


# ===================== SKILL / GAP-ANALYSIS MODELS =====================
# Generic "Reference Framework Gap Analysis" capability. skill_id selects
# behavior; ISO 27001 is just the first framework_name used with
# "compliance_gap_check" — nothing here is ISO-specific.

class GapAnalysisRequest(BaseModel):
    skill_id: str  # "compliance_gap_check" | "scenario_regulatory_impact"
    reference_collection_ids: List[str]  # array from day one (Opsi A: multi-framework per run)
    framework_name: str  # free-label, e.g. "ISO 27001" or "Ketentuan PPh Pinjaman vs Modal"
    target_collection_ids: List[str] = []  # required for compliance_gap_check — one guideline vs N files, verdict per file
    scenario_input: Optional[str] = None  # used by scenario_regulatory_impact instead of a target collection


class GapAnalysisItem(BaseModel):
    label: str  # control/klausul name (Skill 1) or option name (Skill 2)
    status: str  # "met" | "partial" | "not_met" | "unknown"
    evidence: Optional[str] = None
    source_citation: Optional[str] = None
    recommendation: Optional[str] = None
    target_collection_id: Optional[str] = None  # which target collection/file this item was checked against


class GapAnalysisRun(BaseModel):
    run_id: str
    skill_id: str
    framework_name: str
    reference_collection_ids: List[str]
    target_collection_ids: List[str] = []
    scenario_input: Optional[str] = None
    status: str = "completed"
    created_at: Union[str, datetime] = ""


class GapAnalysisResponse(BaseModel):
    run: GapAnalysisRun
    items: List[GapAnalysisItem]
    summary: Dict[str, int]  # counts per status, e.g. {"met": 60, "partial": 18, "not_met": 15, "unknown": 0}
    disclaimer: Optional[str] = None


# ===================== SKILLS (MS-251) =====================
# A skill is an uploaded instruction file: `instruction` holds the body of the
# .md, the rest is its frontmatter. Who may use it is decided by `scope` alone
# — "personal" is the uploader's own, "team" is an admin's and reaches every
# member that admin created (see storage.list_skills_for_user).

SKILL_SCOPES = ("personal", "team")


def _validate_scope(v: str) -> str:
    if v not in SKILL_SCOPES:
        raise ValueError(f"scope must be one of {SKILL_SCOPES}")
    return v


def _normalize_command(v: str) -> str:
    """Store commands in the one shape the "/" menu matches on, so a skill
    uploaded as "audit" is still reachable as "/audit"."""
    v = v.strip()
    if not v.lstrip("/"):
        raise ValueError("slash_command must not be empty")
    return "/" + v.lstrip("/")


def _require_text(v: str, field: str) -> str:
    if not v or not v.strip():
        raise ValueError(f"{field} must not be empty")
    return v.strip()


class SkillCreate(BaseModel):
    name: str
    slash_command: str
    instruction: str          # body of the uploaded .md
    description: str = ""
    scope: str = "personal"   # "team" is rejected for non-admins in router/skills.py

    @field_validator("scope")
    @classmethod
    def _scope(cls, v: str) -> str:
        return _validate_scope(v)

    @field_validator("slash_command")
    @classmethod
    def _command(cls, v: str) -> str:
        return _normalize_command(v)

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        return _require_text(v, "name")

    @field_validator("instruction")
    @classmethod
    def _instruction(cls, v: str) -> str:
        return _require_text(v, "instruction")


class SkillUpdate(BaseModel):
    """Partial update — every field optional, omitted ones are left alone."""
    name: Optional[str] = None
    slash_command: Optional[str] = None
    instruction: Optional[str] = None
    description: Optional[str] = None
    scope: Optional[str] = None

    @field_validator("scope")
    @classmethod
    def _scope(cls, v: Optional[str]) -> Optional[str]:
        return _validate_scope(v) if v is not None else v

    @field_validator("slash_command")
    @classmethod
    def _command(cls, v: Optional[str]) -> Optional[str]:
        return _normalize_command(v) if v is not None else v

    @field_validator("name")
    @classmethod
    def _name(cls, v: Optional[str]) -> Optional[str]:
        return _require_text(v, "name") if v is not None else v

    @field_validator("instruction")
    @classmethod
    def _instruction(cls, v: Optional[str]) -> Optional[str]:
        return _require_text(v, "instruction") if v is not None else v


class SkillResponse(BaseModel):
    skill_id: str
    name: str
    slash_command: str
    description: str = ""
    instruction: str
    scope: str
    owner_id: str
    created_at: Union[str, datetime] = ""
    updated_at: Union[str, datetime] = ""
