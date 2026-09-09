"""
storage.py
----------
Supabase Storage wrapper using the S3-compatible API (boto3).

Credentials (set in .env / HF Space secrets):
  SUPABASE_S3_ACCESS_KEY_ID  -- S3 Access Key ID (Supabase Storage -> S3 Access Keys)
  SUPABASE_S3_SECRET_KEY     -- S3 Secret Access Key
  SUPABASE_S3_ENDPOINT       -- e.g. https://<ref>.storage.supabase.co/storage/v1/s3
  SUPABASE_S3_REGION         -- e.g. ap-southeast-1

DATABASE_URL is used for metadata tables (pdf_collections, chat_collections).
Auto-migration creates both tables on first run.

Buckets (create once in Supabase Dashboard -> Storage):
  pdf-uploads    raw PDF files
  pdf-indices    FAISS index files for PDFs
  chat-uploads   raw chat TXT files
  chat-indices   FAISS index files + metadata.json for chats
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional
from config import config
from utils import DOCUMENT_EXTRACTORS, CONTENT_TYPE_BY_EXT

logger = logging.getLogger(__name__)

_UPLOADS_BUCKET      = "pdf-uploads"
_INDICES_BUCKET      = "pdf-indices"
_CHAT_UPLOADS_BUCKET = "chat-uploads"
_CHAT_INDICES_BUCKET = "chat-indices"

_migration_done = False


def _database_url() -> Optional[str]:
    url = os.getenv("DATABASE_URL") or getattr(config, "database_url", None)
    if not url:
        return None
    if "sslmode=" not in url:
        separator = "&" if "?" in url else "?"
        url = f"{url}{separator}sslmode=require"
    return url


def _s3_settings() -> tuple[Optional[str], Optional[str], Optional[str], str]:
    access_key = os.getenv("SUPABASE_S3_ACCESS_KEY_ID") or getattr(config, "supabase_s3_access_key_id", None)
    secret_key = os.getenv("SUPABASE_S3_SECRET_KEY") or getattr(config, "supabase_s3_secret_key", None)
    endpoint = os.getenv("SUPABASE_S3_ENDPOINT") or getattr(config, "supabase_s3_endpoint", None)
    region = os.getenv("SUPABASE_S3_REGION") or getattr(config, "supabase_s3_region", "ap-southeast-1")
    return access_key, secret_key, endpoint, region


# ---------------------------------------------------------------------------
# Auto-migration
# ---------------------------------------------------------------------------

def ensure_schema():
    """Create pdf_collections + chat_collections tables if they don't exist."""
    global _migration_done
    if _migration_done:
        return
    database_url = _database_url()
    if not database_url:
        logger.warning("ensure_schema: DATABASE_URL not set — skipping auto-migration")
        _migration_done = True
        return
    logger.info("ensure_schema: connecting to DB to create tables...")
    try:
        import psycopg2
        # Embed sslmode in URL to avoid kwarg conflict with pooler DSN
        conn = psycopg2.connect(database_url, connect_timeout=10)
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS pdf_collections (
                    id            BIGSERIAL PRIMARY KEY,
                    collection_id TEXT        NOT NULL UNIQUE,
                    title         TEXT        NOT NULL DEFAULT '',
                    file_names    TEXT[]      NOT NULL DEFAULT '{}',
                    chunk_count   INTEGER     NOT NULL DEFAULT 0,
                    storage_paths TEXT[]      NOT NULL DEFAULT '{}',
                    status        TEXT        NOT NULL DEFAULT 'active'
                                  CHECK (status IN ('active', 'inactive')),
                    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                ALTER TABLE pdf_collections
                    ADD COLUMN IF NOT EXISTS title TEXT NOT NULL DEFAULT '';
                ALTER TABLE pdf_collections
                    ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'active';
                ALTER TABLE pdf_collections
                    ADD COLUMN IF NOT EXISTS owner_id TEXT;
                CREATE INDEX IF NOT EXISTS idx_pdf_collections_cid
                    ON pdf_collections (collection_id);
                CREATE INDEX IF NOT EXISTS idx_pdf_collections_owner
                    ON pdf_collections (owner_id);
                CREATE INDEX IF NOT EXISTS idx_pdf_collections_created
                    ON pdf_collections (created_at DESC);

                CREATE TABLE IF NOT EXISTS chat_collections (
                    id              BIGSERIAL PRIMARY KEY,
                    collection_id   TEXT        NOT NULL UNIQUE,
                    file_name       TEXT        NOT NULL DEFAULT '',
                    platform        TEXT        NOT NULL DEFAULT 'whatsapp',
                    message_count   INTEGER     NOT NULL DEFAULT 0,
                    participants    TEXT[]      NOT NULL DEFAULT '{}',
                    date_range      JSONB,
                    keywords        TEXT[]      NOT NULL DEFAULT '{}',
                    storage_paths   TEXT[]      NOT NULL DEFAULT '{}',
                    status          TEXT        NOT NULL DEFAULT 'active'
                                    CHECK (status IN ('active', 'inactive')),
                    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                ALTER TABLE chat_collections
                    ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'active';
                CREATE INDEX IF NOT EXISTS idx_chat_collections_cid
                    ON chat_collections (collection_id);
                CREATE INDEX IF NOT EXISTS idx_chat_collections_created
                    ON chat_collections (created_at DESC);

                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id               BIGSERIAL PRIMARY KEY,
                    session_id       TEXT        NOT NULL UNIQUE DEFAULT gen_random_uuid()::text,
                    title            TEXT        NOT NULL DEFAULT '',
                    pdf_collections  TEXT[]      NOT NULL DEFAULT '{}',
                    chat_collections TEXT[]      NOT NULL DEFAULT '{}',
                    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                ALTER TABLE chat_sessions
                    ADD COLUMN IF NOT EXISTS owner_id TEXT;
                CREATE INDEX IF NOT EXISTS idx_chat_sessions_sid
                    ON chat_sessions (session_id);
                CREATE INDEX IF NOT EXISTS idx_chat_sessions_updated
                    ON chat_sessions (updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_chat_sessions_owner
                    ON chat_sessions (owner_id);

                CREATE TABLE IF NOT EXISTS chat_messages (
                    id          BIGSERIAL   PRIMARY KEY,
                    message_id  TEXT        NOT NULL UNIQUE DEFAULT gen_random_uuid()::text,
                    session_id  TEXT        NOT NULL REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
                    role        TEXT        NOT NULL,
                    content     TEXT        NOT NULL DEFAULT '',
                    model_used  TEXT,
                    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                CREATE INDEX IF NOT EXISTS idx_chat_messages_session
                    ON chat_messages (session_id, created_at ASC);

                CREATE TABLE IF NOT EXISTS gap_analysis_runs (
                    id                       BIGSERIAL   PRIMARY KEY,
                    run_id                   TEXT        NOT NULL UNIQUE DEFAULT gen_random_uuid()::text,
                    skill_id                 TEXT        NOT NULL,
                    framework_name           TEXT        NOT NULL DEFAULT '',
                    reference_collection_ids TEXT[]      NOT NULL DEFAULT '{}',
                    target_collection_id     TEXT,
                    scenario_input           TEXT,
                    owner_id                 TEXT,
                    status                   TEXT        NOT NULL DEFAULT 'completed'
                                             CHECK (status IN ('running', 'completed', 'failed')),
                    created_at               TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                -- target_collection_id (singular) is kept only so old rows stay
                -- readable; new rows are written to target_collection_ids below,
                -- which supports comparing one guideline against several files.
                ALTER TABLE gap_analysis_runs
                    ADD COLUMN IF NOT EXISTS target_collection_ids TEXT[] NOT NULL DEFAULT '{}';
                -- Backfill pre-existing rows once: every read path now selects only
                -- the plural column, so without this, old runs would silently show
                -- "no target" the moment this migration lands. Idempotent — the
                -- WHERE clause only matches rows that haven't been backfilled yet.
                UPDATE gap_analysis_runs
                   SET target_collection_ids = ARRAY[target_collection_id]
                 WHERE target_collection_id IS NOT NULL
                   AND (target_collection_ids IS NULL OR target_collection_ids = '{}');
                CREATE INDEX IF NOT EXISTS idx_gap_analysis_runs_run_id
                    ON gap_analysis_runs (run_id);
                CREATE INDEX IF NOT EXISTS idx_gap_analysis_runs_owner
                    ON gap_analysis_runs (owner_id, created_at DESC);

                CREATE TABLE IF NOT EXISTS gap_analysis_items (
                    id              BIGSERIAL   PRIMARY KEY,
                    run_id          TEXT        NOT NULL REFERENCES gap_analysis_runs(run_id) ON DELETE CASCADE,
                    label           TEXT        NOT NULL,
                    status          TEXT        NOT NULL DEFAULT 'unknown'
                                    CHECK (status IN ('met', 'partial', 'not_met', 'unknown')),
                    evidence        TEXT,
                    source_citation TEXT,
                    recommendation  TEXT,
                    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                ALTER TABLE gap_analysis_items
                    ADD COLUMN IF NOT EXISTS target_collection_id TEXT;
                CREATE INDEX IF NOT EXISTS idx_gap_analysis_items_run
                    ON gap_analysis_items (run_id);

                CREATE TABLE IF NOT EXISTS users (
                    id            BIGSERIAL    PRIMARY KEY,
                    user_id       TEXT         NOT NULL UNIQUE DEFAULT gen_random_uuid()::text,
                    email         TEXT         NOT NULL UNIQUE,
                    password_hash TEXT         NOT NULL,
                    role          TEXT         NOT NULL DEFAULT 'user'
                                  CHECK (role IN ('user', 'admin')),
                    is_active     BOOLEAN      NOT NULL DEFAULT true,
                    created_at    TIMESTAMPTZ  NOT NULL DEFAULT now(),
                    updated_at    TIMESTAMPTZ  NOT NULL DEFAULT now()
                );
                CREATE INDEX IF NOT EXISTS idx_users_email   ON users (email);
                CREATE INDEX IF NOT EXISTS idx_users_user_id ON users (user_id);

                CREATE TABLE IF NOT EXISTS skills (
                    id            BIGSERIAL   PRIMARY KEY,
                    skill_id      TEXT        NOT NULL UNIQUE DEFAULT gen_random_uuid()::text,
                    name          TEXT        NOT NULL,
                    slash_command TEXT        NOT NULL,
                    description   TEXT        NOT NULL DEFAULT '',
                    instruction   TEXT        NOT NULL,
                    scope         TEXT        NOT NULL DEFAULT 'personal'
                                  CHECK (scope IN ('personal', 'team')),
                    owner_id      TEXT        NOT NULL,
                    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                -- owner_id leads the index: both visibility branches filter it by
                -- equality, and scope only has two values so it barely narrows
                -- anything on its own. Same shape as idx_gap_analysis_runs_owner.
                CREATE INDEX IF NOT EXISTS idx_skills_owner_scope
                    ON skills (owner_id, scope);
            """)
        conn.close()
        logger.info("Schema ensured: pdf_collections, chat_collections, chat_sessions, chat_messages, gap_analysis_runs, gap_analysis_items, skills.")
    except Exception as e:
        logger.warning("Auto-migration skipped (disk fallback): %s", e)
    _migration_done = True


# ---------------------------------------------------------------------------
# S3 client
# ---------------------------------------------------------------------------

def _s3_client():
    """Return a boto3 S3 client pointed at Supabase Storage, or None."""
    access_key, secret_key, endpoint, region = _s3_settings()
    if not (access_key and secret_key and endpoint):
        return None
    try:
        import boto3
        from botocore.config import Config
        return boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
            config=Config(signature_version="s3v4"),
        )
    except ImportError:
        logger.warning("boto3 not installed -- storage features disabled")
        return None
    except Exception as e:
        logger.warning("S3 client init failed: %s", e)
        return None


def is_enabled() -> bool:
    """True if S3 credentials are present."""
    access_key, secret_key, endpoint, _ = _s3_settings()
    return bool(access_key and secret_key and endpoint)


def list_collection_ids_from_s3() -> List[str]:
    """List collection IDs by scanning the pdf-indices bucket (top-level prefixes)."""
    s3 = _s3_client()
    if not s3:
        return []
    try:
        resp = s3.list_objects_v2(Bucket=_INDICES_BUCKET, Delimiter="/")
        prefixes = resp.get("CommonPrefixes", [])
        ids = [p["Prefix"].rstrip("/") for p in prefixes]
        logger.info("list_collection_ids_from_s3: found %d collections", len(ids))
        return ids
    except Exception as e:
        logger.warning("list_collection_ids_from_s3 failed: %s", e)
        return []


def list_collections_from_s3() -> List[Dict[str, Any]]:
    """Scan pdf-indices for collection IDs, then pdf-uploads for file names.
    Returns list of dicts: {collection_id, file_names, chunk_count, created_at}
    """
    s3 = _s3_client()
    if not s3:
        return []
    try:
        # Get collection IDs from indices bucket
        resp = s3.list_objects_v2(Bucket=_INDICES_BUCKET, Delimiter="/")
        prefixes = resp.get("CommonPrefixes", [])
        collection_ids = [p["Prefix"].rstrip("/") for p in prefixes]
    except Exception as e:
        logger.warning("list_collections_from_s3 (indices scan) failed: %s", e)
        return []

    results = []
    for cid in collection_ids:
        file_names = []
        created_at = ""
        try:
            upload_resp = s3.list_objects_v2(Bucket=_UPLOADS_BUCKET, Prefix=f"{cid}/")
            for obj in upload_resp.get("Contents", []):
                key = obj["Key"]
                fname = key.split("/", 1)[-1]
                if fname and os.path.splitext(fname)[1].lower() in DOCUMENT_EXTRACTORS:
                    file_names.append(fname)
                    if not created_at:
                        created_at = obj.get("LastModified", "")  # datetime or str
                        if hasattr(created_at, "isoformat"):
                            created_at = created_at.isoformat()
        except Exception as e:
            logger.warning("list_collections_from_s3 (uploads scan %s) failed: %s", cid, e)
        results.append({
            "collection_id": cid,
            "file_names": file_names,
            "chunk_count": len(file_names),
            "created_at": created_at,
        })

    logger.info("list_collections_from_s3: found %d collections", len(results))
    return results


def list_chat_collection_ids_from_s3() -> List[str]:
    """List chat collection IDs by scanning the chat-indices bucket (top-level prefixes)."""
    s3 = _s3_client()
    if not s3:
        return []
    try:
        resp = s3.list_objects_v2(Bucket=_CHAT_INDICES_BUCKET, Delimiter="/")
        prefixes = resp.get("CommonPrefixes", [])
        ids = [p["Prefix"].rstrip("/") for p in prefixes]
        logger.info("list_chat_collection_ids_from_s3: found %d collections", len(ids))
        return ids
    except Exception as e:
        logger.warning("list_chat_collection_ids_from_s3 failed: %s", e)
        return []


def has_database() -> bool:
    return bool(_database_url())


# ---------------------------------------------------------------------------
# psycopg2 connection
# ---------------------------------------------------------------------------

def _db_conn():
    url = _database_url()
    if not url:
        return None
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
        conn = psycopg2.connect(url, cursor_factory=RealDictCursor)
        conn.autocommit = True
        return conn
    except Exception as e:
        logger.warning("DB connection failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# PDF uploads & indices
# ---------------------------------------------------------------------------

def upload_pdf(collection_id: str, file_path: str, filename: str) -> Optional[str]:
    """Upload a raw document (PDF, DOCX, CSV, XLSX, TXT — name kept for
    backward compatibility) to the pdf-uploads bucket. Returns S3 key or None."""
    s3 = _s3_client()
    if not s3:
        return None
    key = f"{collection_id}/{filename}"
    ext = os.path.splitext(filename)[1].lower()
    content_type = CONTENT_TYPE_BY_EXT.get(ext, "application/octet-stream")
    try:
        s3.upload_file(file_path, _UPLOADS_BUCKET, key,
                       ExtraArgs={"ContentType": content_type})
        logger.info("Uploaded file: %s (%s)", key, content_type)
        return key
    except Exception as e:
        logger.warning("File upload failed (%s): %s", key, e)
        return None


def upload_index(collection_id: str, index_dir: str) -> bool:
    """Upload index.faiss + index.pkl to the pdf-indices bucket."""
    s3 = _s3_client()
    if not s3:
        return False
    success = True
    for fname in ("index.faiss", "index.pkl"):
        local = os.path.join(index_dir, fname)
        if not os.path.exists(local):
            success = False
            continue
        key = f"{collection_id}/{fname}"
        try:
            s3.upload_file(local, _INDICES_BUCKET, key)
            logger.info("Uploaded PDF index: %s", key)
        except Exception as e:
            logger.warning("PDF index upload failed (%s): %s", key, e)
            success = False
    return success


def download_index(collection_id: str, dest_dir: str) -> bool:
    """Download PDF FAISS index files from S3 into dest_dir."""
    s3 = _s3_client()
    if not s3:
        return False
    os.makedirs(dest_dir, exist_ok=True)
    success = True
    for fname in ("index.faiss", "index.pkl"):
        key = f"{collection_id}/{fname}"
        dest = os.path.join(dest_dir, fname)
        if os.path.exists(dest):
            continue
        try:
            s3.download_file(_INDICES_BUCKET, key, dest)
            logger.info("Downloaded PDF index: %s", key)
        except Exception as e:
            logger.warning("PDF index download failed (%s): %s", key, e)
            success = False
    return success


def get_pdf_signed_url(collection_id: str, filename: str, expires_in: int = 3600) -> Optional[str]:
    """Return a pre-signed URL to download a PDF."""
    s3 = _s3_client()
    if not s3:
        return None
    key = f"{collection_id}/{filename}"
    try:
        return s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": _UPLOADS_BUCKET, "Key": key},
            ExpiresIn=expires_in,
        )
    except Exception as e:
        logger.warning("Presigned URL failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Chat uploads & indices
# ---------------------------------------------------------------------------

def upload_chat_file(collection_id: str, file_path: str, filename: str) -> Optional[str]:
    """Upload a raw chat TXT file to the chat-uploads bucket."""
    s3 = _s3_client()
    if not s3:
        return None
    key = f"{collection_id}/{filename}"
    try:
        s3.upload_file(file_path, _CHAT_UPLOADS_BUCKET, key,
                       ExtraArgs={"ContentType": "text/plain"})
        logger.info("Uploaded chat file: %s", key)
        return key
    except Exception as e:
        logger.warning("Chat file upload failed (%s): %s", key, e)
        return None


def download_chat_file(collection_id: str, filename: str, dest_dir: str) -> bool:
    """Download a raw chat TXT file from the chat-uploads bucket into dest_dir.

    Needed on ephemeral filesystems (e.g. HF Spaces): the local copy vanishes
    on restart while the uploaded file lives on in the bucket.
    """
    s3 = _s3_client()
    if not s3:
        return False
    os.makedirs(dest_dir, exist_ok=True)
    key = f"{collection_id}/{filename}"
    dest = os.path.join(dest_dir, filename)
    if os.path.exists(dest):
        return True
    try:
        s3.download_file(_CHAT_UPLOADS_BUCKET, key, dest)
        logger.info("Downloaded chat file: %s", key)
        return True
    except Exception as e:
        logger.warning("Chat file download failed (%s): %s", key, e)
        return False


def upload_chat_index(collection_id: str, index_dir: str) -> bool:
    """Upload index.faiss + index.pkl + metadata.json to the chat-indices bucket."""
    s3 = _s3_client()
    if not s3:
        return False
    success = True
    for fname in ("index.faiss", "index.pkl", "metadata.json"):
        local = os.path.join(index_dir, fname)
        if not os.path.exists(local):
            if fname != "metadata.json":
                success = False
            continue
        key = f"{collection_id}/{fname}"
        ctype = "application/json" if fname.endswith(".json") else "application/octet-stream"
        try:
            s3.upload_file(local, _CHAT_INDICES_BUCKET, key,
                           ExtraArgs={"ContentType": ctype})
            logger.info("Uploaded chat index: %s", key)
        except Exception as e:
            logger.warning("Chat index upload failed (%s): %s", key, e)
            success = False
    return success


def download_chat_index(collection_id: str, dest_dir: str) -> bool:
    """Download chat FAISS index files from S3 into dest_dir."""
    s3 = _s3_client()
    if not s3:
        return False
    os.makedirs(dest_dir, exist_ok=True)
    success = True
    for fname in ("index.faiss", "index.pkl", "metadata.json"):
        key = f"{collection_id}/{fname}"
        dest = os.path.join(dest_dir, fname)
        if os.path.exists(dest):
            continue
        try:
            s3.download_file(_CHAT_INDICES_BUCKET, key, dest)
            logger.info("Downloaded chat index: %s", key)
        except Exception as e:
            if fname != "metadata.json":
                success = False
            logger.warning("Chat index download failed (%s): %s", key, e)
    return success


# ---------------------------------------------------------------------------
# pdf_collections metadata table
# ---------------------------------------------------------------------------

def register_collection(
    collection_id: str,
    file_names: List[str],
    chunk_count: int,
    title: Optional[str] = None,
    storage_paths: Optional[List[str]] = None,
    owner_id: Optional[str] = None,
) -> bool:
    ensure_schema()
    logger.info("register_collection: collection_id=%s, has_db=%s", collection_id, has_database())
    conn = _db_conn()
    if not conn:
        logger.warning("register_collection: no DB connection, skipping insert")
        return False
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO pdf_collections
                    (collection_id, title, file_names, chunk_count, storage_paths, owner_id)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (collection_id) DO UPDATE SET
                    title         = EXCLUDED.title,
                    file_names    = EXCLUDED.file_names,
                    chunk_count   = EXCLUDED.chunk_count,
                    storage_paths = EXCLUDED.storage_paths,
                    owner_id      = COALESCE(EXCLUDED.owner_id, pdf_collections.owner_id),
                    updated_at    = now()
            """, (collection_id, title or "", file_names, chunk_count, storage_paths or [], owner_id))
        conn.close()
        logger.info("Registered PDF collection: %s", collection_id)
        return True
    except Exception as e:
        logger.warning("register_collection failed: %s", e)
        return False


def delete_collection_from_db(collection_id: str) -> bool:
    ensure_schema()
    conn = _db_conn()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM pdf_collections WHERE collection_id = %s",
                            (collection_id,))
            conn.close()
        except Exception as e:
            logger.warning("delete PDF row failed: %s", e)
    s3 = _s3_client()
    if s3:
        # The DB row is already gone by now, so storage cleanup must never
        # raise — a failure here would surface as a 500 and tell the user the
        # delete failed when it actually succeeded. Indices use fixed key
        # names; uploads are listed because their file names vary.
        for bucket, keys in (
            (_INDICES_BUCKET, [f"{collection_id}/index.faiss", f"{collection_id}/index.pkl"]),
            (_UPLOADS_BUCKET, None),
        ):
            if keys is None:
                try:
                    resp = s3.list_objects_v2(Bucket=bucket, Prefix=f"{collection_id}/")
                    keys = [o["Key"] for o in resp.get("Contents", [])]
                except Exception as e:
                    logger.warning("PDF S3 list failed (%s): %s", bucket, e)
                    keys = []
            for key in keys:
                try:
                    s3.delete_object(Bucket=bucket, Key=key)
                except Exception as e:
                    logger.warning("PDF S3 delete failed (%s): %s", key, e)
    logger.info("Deleted PDF collection: %s", collection_id)
    return True


def list_collections() -> List[Dict[str, Any]]:
    ensure_schema()
    conn = _db_conn()
    if not conn:
        return []
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT collection_id, title, file_names, chunk_count, storage_paths, status, owner_id, created_at, updated_at
                FROM pdf_collections ORDER BY created_at DESC
            """)
            rows = cur.fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.warning("list_collections failed: %s", e)
        return []


def get_collection(collection_id: str) -> Optional[Dict[str, Any]]:
    ensure_schema()
    conn = _db_conn()
    if not conn:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT collection_id, title, file_names, chunk_count, storage_paths, status, owner_id, created_at, updated_at
                FROM pdf_collections WHERE collection_id = %s
            """, (collection_id,))
            row = cur.fetchone()
        conn.close()
        return dict(row) if row else None
    except Exception as e:
        logger.warning("get_collection failed for %s: %s", collection_id, e)
        return None


def list_collection_ids_for_user(user_id: str, is_admin: bool) -> List[str]:
    """Collection IDs a given user is allowed to see: all of them for admins,
    only their own (owner_id match) for everyone else."""
    ensure_schema()
    conn = _db_conn()
    if not conn:
        return []
    try:
        with conn.cursor() as cur:
            if is_admin:
                cur.execute("SELECT collection_id FROM pdf_collections")
            else:
                cur.execute(
                    "SELECT collection_id FROM pdf_collections WHERE owner_id = %s",
                    (user_id,),
                )
            rows = cur.fetchall()
        conn.close()
        return [r["collection_id"] for r in rows]
    except Exception as e:
        logger.warning("list_collection_ids_for_user failed: %s", e)
        return []


def set_collection_status(collection_id: str, active: bool) -> bool:
    ensure_schema()
    conn = _db_conn()
    if not conn:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE pdf_collections SET status = %s, updated_at = now() WHERE collection_id = %s",
                ("active" if active else "inactive", collection_id),
            )
            updated = cur.rowcount > 0
        conn.close()
        return updated
    except Exception as e:
        logger.warning("set_collection_status failed for %s: %s", collection_id, e)
        return False


# ---------------------------------------------------------------------------
# gap_analysis_runs / gap_analysis_items — generic "Skill" persistence
# (compliance_gap_check today, scenario_regulatory_impact scaffolded).
# Schema is skill-agnostic on purpose: no ISO/pajak-specific columns.
# ---------------------------------------------------------------------------

def create_gap_analysis_run(
    run_id: str,
    skill_id: str,
    reference_collection_ids: List[str],
    framework_name: str = "",
    target_collection_ids: Optional[List[str]] = None,
    scenario_input: Optional[str] = None,
    owner_id: Optional[str] = None,
    status: str = "completed",
) -> bool:
    ensure_schema()
    conn = _db_conn()
    if not conn:
        logger.warning("create_gap_analysis_run: no DB connection, skipping insert")
        return False
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO gap_analysis_runs
                    (run_id, skill_id, framework_name, reference_collection_ids,
                     target_collection_ids, scenario_input, owner_id, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (run_id) DO UPDATE SET
                    status = EXCLUDED.status
            """, (run_id, skill_id, framework_name, reference_collection_ids,
                  target_collection_ids or [], scenario_input, owner_id, status))
        conn.close()
        logger.info("Created gap_analysis_run: %s (skill=%s)", run_id, skill_id)
        return True
    except Exception as e:
        logger.warning("create_gap_analysis_run failed: %s", e)
        return False


def save_gap_analysis_items(run_id: str, items: List[Dict[str, Any]]) -> bool:
    """items: dicts with keys label, status, evidence, source_citation,
    recommendation, target_collection_id (which target this item was checked against)."""
    ensure_schema()
    if not items:
        return True
    conn = _db_conn()
    if not conn:
        logger.warning("save_gap_analysis_items: no DB connection, skipping insert")
        return False
    try:
        with conn.cursor() as cur:
            for item in items:
                cur.execute("""
                    INSERT INTO gap_analysis_items
                        (run_id, label, status, evidence, source_citation, recommendation, target_collection_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    run_id,
                    item.get("label", ""),
                    item.get("status", "unknown"),
                    item.get("evidence"),
                    item.get("source_citation"),
                    item.get("recommendation"),
                    item.get("target_collection_id"),
                ))
        conn.close()
        return True
    except Exception as e:
        logger.warning("save_gap_analysis_items failed for run %s: %s", run_id, e)
        return False


def list_gap_analysis_runs(owner_id: Optional[str] = None, is_admin: bool = False) -> List[Dict[str, Any]]:
    ensure_schema()
    conn = _db_conn()
    if not conn:
        return []
    try:
        with conn.cursor() as cur:
            if is_admin:
                cur.execute("""
                    SELECT run_id, skill_id, framework_name, reference_collection_ids,
                           target_collection_ids, scenario_input, owner_id, status, created_at
                    FROM gap_analysis_runs ORDER BY created_at DESC
                """)
            else:
                cur.execute("""
                    SELECT run_id, skill_id, framework_name, reference_collection_ids,
                           target_collection_ids, scenario_input, owner_id, status, created_at
                    FROM gap_analysis_runs WHERE owner_id = %s ORDER BY created_at DESC
                """, (owner_id,))
            rows = cur.fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.warning("list_gap_analysis_runs failed: %s", e)
        return []


def get_gap_analysis_run(run_id: str) -> Optional[Dict[str, Any]]:
    ensure_schema()
    conn = _db_conn()
    if not conn:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT run_id, skill_id, framework_name, reference_collection_ids,
                       target_collection_ids, scenario_input, owner_id, status, created_at
                FROM gap_analysis_runs WHERE run_id = %s
            """, (run_id,))
            row = cur.fetchone()
        conn.close()
        return dict(row) if row else None
    except Exception as e:
        logger.warning("get_gap_analysis_run failed for %s: %s", run_id, e)
        return None


def get_gap_analysis_items(run_id: str) -> List[Dict[str, Any]]:
    ensure_schema()
    conn = _db_conn()
    if not conn:
        return []
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT label, status, evidence, source_citation, recommendation, target_collection_id, created_at
                FROM gap_analysis_items WHERE run_id = %s ORDER BY id ASC
            """, (run_id,))
            rows = cur.fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.warning("get_gap_analysis_items failed for %s: %s", run_id, e)
        return []


def delete_gap_analysis_run(run_id: str) -> bool:
    """Deletes the run row; gap_analysis_items cascade-delete via the FK
    (ON DELETE CASCADE), so no separate items cleanup is needed here."""
    ensure_schema()
    conn = _db_conn()
    if not conn:
        logger.warning("delete_gap_analysis_run: no DB connection, skipping delete")
        return False
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM gap_analysis_runs WHERE run_id = %s", (run_id,))
        conn.close()
        return True
    except Exception as e:
        logger.warning("delete_gap_analysis_run failed for %s: %s", run_id, e)
        return False


# ---------------------------------------------------------------------------
# skills — user-uploaded skill instructions (MS-251)
# Visibility is derived, not stored: a "team" skill belongs to the admin who
# uploaded it and is visible to every account that admin created
# (users.created_by), so there is no separate assignment table to keep in sync.
# ---------------------------------------------------------------------------

def _team_admin_id(user_id: str, is_admin: bool) -> Optional[str]:
    """Which admin's team skills this user can see: their own id if they are an
    admin, otherwise users.created_by. Looked up rather than read off the JWT —
    the token (router/auth.py) carries no created_by, and widening it would log
    every active session out."""
    if is_admin:
        return user_id
    conn = _db_conn()
    if not conn:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT created_by FROM users WHERE user_id = %s", (user_id,))
            row = cur.fetchone()
        conn.close()
        return row["created_by"] if row else None
    except Exception as e:
        logger.warning("_team_admin_id failed for %s: %s", user_id, e)
        return None


def create_skill(
    skill_id: str,
    name: str,
    slash_command: str,
    instruction: str,
    owner_id: str,
    description: str = "",
    scope: str = "personal",
) -> bool:
    ensure_schema()
    conn = _db_conn()
    if not conn:
        logger.warning("create_skill: no DB connection, skipping insert")
        return False
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO skills
                    (skill_id, name, slash_command, description, instruction, scope, owner_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (skill_id, name, slash_command, description, instruction, scope, owner_id))
        conn.close()
        logger.info("Created skill: %s (scope=%s, owner=%s)", skill_id, scope, owner_id)
        return True
    except Exception as e:
        logger.warning("create_skill failed: %s", e)
        return False


def list_skills_for_user(user_id: str, is_admin: bool) -> List[Dict[str, Any]]:
    """Skills this account may use: its own personal ones, plus the team skills
    of the admin it belongs to."""
    ensure_schema()
    admin_id = _team_admin_id(user_id, is_admin)
    conn = _db_conn()
    if not conn:
        return []
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT skill_id, name, slash_command, description, instruction,
                       scope, owner_id, created_at, updated_at
                FROM skills
                WHERE (scope = 'personal' AND owner_id = %s)
                   OR (scope = 'team' AND owner_id = %s)
                ORDER BY created_at DESC
            """, (user_id, admin_id))
            rows = cur.fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.warning("list_skills_for_user failed for %s: %s", user_id, e)
        return []


def get_skill_for_user(skill_id: str, user_id: str, is_admin: bool) -> Optional[Dict[str, Any]]:
    """One skill, but only if this account may use it — same visibility rule as
    list_skills_for_user. Returns None when it does not exist OR is not visible,
    so callers cannot tell the two apart."""
    ensure_schema()
    admin_id = _team_admin_id(user_id, is_admin)
    conn = _db_conn()
    if not conn:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT skill_id, name, slash_command, description, instruction,
                       scope, owner_id, created_at, updated_at
                FROM skills
                WHERE skill_id = %s
                  AND ((scope = 'personal' AND owner_id = %s)
                    OR (scope = 'team' AND owner_id = %s))
            """, (skill_id, user_id, admin_id))
            row = cur.fetchone()
        conn.close()
        return dict(row) if row else None
    except Exception as e:
        logger.warning("get_skill_for_user failed for %s: %s", skill_id, e)
        return None


def update_skill(skill_id: str, owner_id: str, fields: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Partial update, owner-only. Returns the updated row, or None if the skill
    does not exist or belongs to someone else."""
    ensure_schema()
    allowed = ("name", "slash_command", "description", "instruction", "scope")
    sets = [(k, v) for k, v in fields.items() if k in allowed and v is not None]
    if not sets:
        return None
    conn = _db_conn()
    if not conn:
        return None
    try:
        assignments = ", ".join(f"{k} = %s" for k, _ in sets)
        params = [v for _, v in sets] + [skill_id, owner_id]
        with conn.cursor() as cur:
            cur.execute(f"""
                UPDATE skills SET {assignments}, updated_at = now()
                WHERE skill_id = %s AND owner_id = %s
                RETURNING skill_id, name, slash_command, description, instruction,
                          scope, owner_id, created_at, updated_at
            """, params)
            row = cur.fetchone()
        conn.close()
        return dict(row) if row else None
    except Exception as e:
        logger.warning("update_skill failed for %s: %s", skill_id, e)
        return None


def delete_skill(skill_id: str, owner_id: str) -> bool:
    """Owner-only hard delete, matching how collections and gap-analysis runs are
    removed. Returns False when nothing matched (missing, or not the owner)."""
    ensure_schema()
    conn = _db_conn()
    if not conn:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM skills WHERE skill_id = %s AND owner_id = %s",
                (skill_id, owner_id),
            )
            deleted = cur.rowcount
        conn.close()
        return deleted > 0
    except Exception as e:
        logger.warning("delete_skill failed for %s: %s", skill_id, e)
        return False


# ---------------------------------------------------------------------------
# chat_collections metadata table
# ---------------------------------------------------------------------------

def register_chat_collection(
    collection_id: str,
    file_name: str,
    platform: str,
    message_count: int,
    participants: List[str],
    date_range: Optional[Dict[str, Any]],
    keywords: Optional[List[str]] = None,
    storage_paths: Optional[List[str]] = None,
) -> bool:
    ensure_schema()
    conn = _db_conn()
    if not conn:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO chat_collections
                    (collection_id, file_name, platform, message_count,
                     participants, date_range, keywords, storage_paths)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (collection_id) DO UPDATE SET
                    file_name     = EXCLUDED.file_name,
                    platform      = EXCLUDED.platform,
                    message_count = EXCLUDED.message_count,
                    participants  = EXCLUDED.participants,
                    date_range    = EXCLUDED.date_range,
                    keywords      = EXCLUDED.keywords,
                    storage_paths = EXCLUDED.storage_paths,
                    updated_at    = now()
            """, (
                collection_id, file_name, platform, message_count,
                participants or [],
                json.dumps(date_range) if date_range else None,
                keywords or [],
                storage_paths or [],
            ))
        conn.close()
        logger.info("Registered chat collection: %s", collection_id)
        return True
    except Exception as e:
        logger.warning("register_chat_collection failed: %s", e)
        return False


def list_chat_collections() -> List[Dict[str, Any]]:
    ensure_schema()
    conn = _db_conn()
    if not conn:
        return []
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT collection_id, file_name, platform, message_count,
                       participants, date_range, keywords, storage_paths, status,
                       created_at, updated_at
                FROM chat_collections ORDER BY created_at DESC
            """)
            rows = cur.fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.warning("list_chat_collections failed: %s", e)
        return []


def get_chat_collection(collection_id: str) -> Optional[Dict[str, Any]]:
    ensure_schema()
    conn = _db_conn()
    if not conn:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT collection_id, file_name, platform, message_count,
                       participants, date_range, keywords, storage_paths, status,
                       created_at, updated_at
                FROM chat_collections WHERE collection_id = %s
            """, (collection_id,))
            row = cur.fetchone()
        conn.close()
        return dict(row) if row else None
    except Exception as e:
        logger.warning("get_chat_collection failed for %s: %s", collection_id, e)
        return None


def set_chat_collection_status(collection_id: str, active: bool) -> bool:
    ensure_schema()
    conn = _db_conn()
    if not conn:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE chat_collections SET status = %s, updated_at = now() WHERE collection_id = %s",
                ("active" if active else "inactive", collection_id),
            )
            updated = cur.rowcount > 0
        conn.close()
        return updated
    except Exception as e:
        logger.warning("set_chat_collection_status failed for %s: %s", collection_id, e)
        return False


def delete_chat_collection_from_db(collection_id: str) -> bool:
    ensure_schema()
    conn = _db_conn()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM chat_collections WHERE collection_id = %s",
                            (collection_id,))
            conn.close()
        except Exception as e:
            logger.warning("delete chat row failed: %s", e)
    s3 = _s3_client()
    if s3:
        for bucket in (_CHAT_UPLOADS_BUCKET, _CHAT_INDICES_BUCKET):
            try:
                resp = s3.list_objects_v2(Bucket=bucket, Prefix=f"{collection_id}/")
                for obj in resp.get("Contents", []):
                    s3.delete_object(Bucket=bucket, Key=obj["Key"])
            except Exception as e:
                logger.warning("Chat S3 delete failed (%s): %s", bucket, e)
    logger.info("Deleted chat collection: %s", collection_id)
    return True
