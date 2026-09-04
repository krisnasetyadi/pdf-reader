from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from typing import List, Optional
from enum import Enum
from urllib.parse import urlparse
import os


# Whitelist of allowed table names (security: prevents SQL injection)
ALLOWED_TABLES = frozenset(["user_profiles", "products", "orders"])


def parse_database_url(url: str) -> dict:
    """
    Parse DATABASE_URL connection string (untuk Neon, Supabase, dll)
    Format: postgresql://user:password@host:port/database?sslmode=require
    """
    parsed = urlparse(url)
    return {
        "host": parsed.hostname or "localhost",
        "port": parsed.port or 5432,
        "database": parsed.path.lstrip("/") if parsed.path else "postgres",
        "user": parsed.username or "postgres",
        "password": parsed.password or "",
    }


class LLMProvider(str, Enum):
    """Supported LLM providers (FREE only)"""
    HUGGINGFACE = "huggingface"  # Local, free
    GEMINI = "gemini"            # Cloud, free tier (60 req/min)


# Floating alias to Google's current-recommended flash model — survives
# Gemini version transitions (unlike a pinned name such as "gemini-2.5-flash",
# which 404s on API keys/projects created after Google's cutoff for it).
DEFAULT_GEMINI_MODEL = "gemini-flash-latest"

# Available models per provider
AVAILABLE_MODELS = {
    LLMProvider.HUGGINGFACE: [
        "google/flan-t5-base",      # Default, smaller
        "google/flan-t5-large",     # Better quality, more RAM
        "google/flan-t5-xl",        # Best quality, needs GPU
    ],
    LLMProvider.GEMINI: [
        DEFAULT_GEMINI_MODEL,       # Floating alias to current flash gen (recommended, default)
        "gemini-2.5-flash",         # Pinned — 404s on API keys/projects created after Google's cutoff
        "gemini-2.5-pro",           # Best quality, paid
        "gemini-2.0-flash",         # Previous gen flash
        "gemini-1.5-flash",         # Legacy, free tier
    ],
}


class Config(BaseSettings):
    # Default LLM Provider (used if no parameter sent)
    llm_provider: LLMProvider = Field(default=LLMProvider.GEMINI)

    # HuggingFace settings (local - fallback)
    model_name: str = Field(default="google/flan-t5-base")

    # (Ollama config removed for HuggingFace Spaces deployment)

    # Gemini settings (cloud - free tier)
    gemini_api_key: Optional[str] = Field(default=None)
    gemini_model: str = Field(default=DEFAULT_GEMINI_MODEL)

    @property
    def default_llm_provider(self) -> LLMProvider:
        return LLMProvider.GEMINI

    @property
    def default_llm_model(self) -> str:
        # Delegates to gemini_model (env-configurable via GEMINI_MODEL) so
        # changing .env is enough — this used to hardcode a literal model
        # name here, silently ignoring gemini_model/the env var entirely.
        return self.gemini_model
    
    # Common LLM settings
    embedding_model: str = Field(default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    max_new_tokens: int = Field(default=256)
    temperature: float = Field(default=0.3)

    # File Paths - PDF
    upload_folder: str = Field(default="data/uploads")
    index_folder: str = Field(default="data/indices")
    
    # File Paths - Chat Logs
    chat_upload_folder: str = Field(default="data/chat_uploads")
    chat_index_folder: str = Field(default="data/chat_indices")

    # Legacy .doc extraction shells out to the `antiword` CLI (utils.py).
    # Defaults to bare "antiword" (resolved via PATH) — correct in the Docker
    # deployment, where apt-get installs it onto the system PATH. Override
    # with a full path on a machine where antiword exists but isn't on PATH
    # for the process running this app (e.g. bundled with Git for Windows at
    # C:\Program Files\Git\mingw64\bin\antiword.exe, outside the default
    # PowerShell PATH) instead of editing that machine's PATH itself.
    antiword_path: str = Field(default="antiword")

    # Retrieval Parameters
    chunk_size: int = Field(default=500)
    chunk_overlap: int = Field(default=50)
    k_results: int = Field(default=5)
    k_per_collection: int = Field(default=5)
    total_k_results: int = Field(default=10)
    
    # Chat-specific retrieval
    chat_chunk_size: int = Field(default=300)
    chat_chunk_overlap: int = Field(default=50)

    # Public-link realtime retrieval (larger chunks: sources are often slide
    # decks whose per-page text is tiny). Override via env:
    # PUBLIC_LINK_CHUNK_SIZE / PUBLIC_LINK_CHUNK_OVERLAP
    public_link_chunk_size: int = Field(default=2000)
    public_link_chunk_overlap: int = Field(default=300)

    # External database connections (user-connected, realtime — schema-agnostic,
    # NOT the app's own DATABASE_URL). Caps keep a single query from scanning
    # an unbounded external database.
    # Large enough that small/medium reference tables (the common case —
    # lookup tables, config tables) land in a SINGLE chunk instead of being
    # split mid-table. Splitting scatters related rows (e.g. every
    # fraction_type/fraction_digit combination) across competing chunks, so
    # only a lucky subset survives top-k retrieval and the LLM sees a
    # partial, misleading slice of the table.
    external_db_chunk_size: int = Field(default=6000)
    external_db_chunk_overlap: int = Field(default=500)
    external_db_max_tables: int = Field(default=20)
    external_db_max_rows_per_table: int = Field(default=200)

    # Database Configuration
    # Option 1: Use DATABASE_URL (recommended for cloud: Neon, Supabase)
    database_url: Optional[str] = Field(default=None)
    
    # Option 2: Individual settings (fallback if DATABASE_URL not set)
    db_host: str = Field(default="localhost")
    db_port: int = Field(default=5432)
    db_name: str = Field(default="postgres")
    db_user: str = Field(default="postgres")
    db_password: str = Field(default="")
    db_tables: List[str] = Field(default=["user_profiles", "products", "orders"])
    
    # SSL mode for database (required for Neon)
    db_sslmode: str = Field(default="prefer")
    
    @property
    def db_config(self) -> dict:
        """Get database config - parse DATABASE_URL if available, else use individual settings"""
        if self.database_url:
            config = parse_database_url(self.database_url)
            config["sslmode"] = "require"  # Cloud DBs require SSL
            return config
        return {
            "host": self.db_host,
            "port": self.db_port,
            "database": self.db_name,
            "user": self.db_user,
            "password": self.db_password,
            "sslmode": self.db_sslmode,
        }
    
    @field_validator('db_tables')
    @classmethod
    def validate_tables(cls, v):
        """Ensure only whitelisted tables are configured"""
        invalid_tables = set(v) - ALLOWED_TABLES
        if invalid_tables:
            raise ValueError(f"Invalid table names: {invalid_tables}. Allowed: {ALLOWED_TABLES}")
        return v
    
    # Supabase Storage — S3-compatible credentials
    # Find at: Supabase Dashboard → Storage → S3 Access Keys
    # When set, PDFs and FAISS indexes are stored in Supabase Storage buckets
    # and survive container restarts (HF Space, Docker, etc.).
    # If not set, files are stored on local disk only.
    supabase_s3_access_key_id: Optional[str] = Field(default=None)
    supabase_s3_secret_key: Optional[str] = Field(default=None)
    supabase_s3_endpoint: Optional[str] = Field(default=None)
    supabase_s3_region: str = Field(default="ap-southeast-1")

    # CORS Configuration
    cors_origins: str = Field(default="*")
    
    @property
    def cors_origin_list(self) -> List[str]:
        """Parse CORS origins from comma-separated string"""
        if self.cors_origins == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]
    
    # Hybrid Search
    enable_hybrid_search: bool = Field(default=True)
    db_result_limit: int = Field(default=10)
    min_similarity_score: float = Field(default=0.3)
    
    # Supported chat platforms
    supported_chat_platforms: List[str] = Field(default=["whatsapp", "telegram"])

    # Telegram (Telethon / MTProto) — live connection, not a file upload.
    # Unlike Supabase/Gemini credentials, api_id/api_hash are NOT set here —
    # each connection brings its own (entered per-connection in the "Connect
    # Telegram" form, from https://my.telegram.org/apps), so different admins
    # can each use their own app registration. Only the at-rest encryption key
    # is server config, since it protects EVERY connection's secrets alike.
    # Generate one with:
    #   python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
    telegram_session_encryption_key: Optional[str] = Field(default=None)

    @property
    def telegram_enabled(self) -> bool:
        return bool(self.telegram_session_encryption_key)

    # Stripe (dummy/test-mode payment flow — MS-90). Test-mode keys only;
    # see router/payment.py.
    stripe_secret_key: Optional[str] = Field(default=None)
    stripe_webhook_secret: Optional[str] = Field(default=None)
    # Used to build the Checkout success/cancel redirect URLs.
    frontend_url: str = Field(default="http://localhost:3008")

    # Global per-user token rate limit (MS-248 follow-up) — a flat safety net
    # against runaway Gemini billing, applied to every user the same way,
    # independent of the per-member monthly allocation in router/payment.py.
    # Sliding window: a user is blocked once the sum of their token_usage in
    # the last rate_limit_window_hours reaches the cap, and unblocks
    # gradually as old usage ages out of the window (see
    # router/payment.py::_get_rate_limit_status). Both are env-overridable so
    # they can be turned down for testing (e.g. a tiny cap + a short window)
    # without a code change.
    rate_limit_token_cap: int = Field(default=300_000)
    rate_limit_window_hours: float = Field(default=24.0)
    # Testing convenience: set RATE_LIMIT_WINDOW_MINUTES in .env to override
    # rate_limit_window_hours with a small value (e.g. "2" for a 2-minute
    # window) without doing decimal-hour math. Unset in production.
    rate_limit_window_minutes: Optional[float] = Field(default=None)

    # Free-tier token allowance per 30-day rolling period (MS-248 follow-up)
    # — a workspace that's never paid still gets a real, enforced cap here
    # instead of relying on the flat rate limit alone. chat-ui/lib/pricing-plans.ts
    # advertises "20 queries/month" for Free as marketing copy; a real query
    # in this app costs roughly 2,000-5,000 Gemini tokens depending on
    # context size, so 60,000 tokens approximates that same budget without
    # this app tracking query *count* anywhere.
    free_plan_token_limit: int = Field(default=60_000)

    @property
    def effective_rate_limit_window_hours(self) -> float:
        if self.rate_limit_window_minutes is not None:
            return self.rate_limit_window_minutes / 60
        return self.rate_limit_window_hours

    class Config:
        env_file = ".env"
        extra = 'ignore'


config = Config()