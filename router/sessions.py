# router/sessions.py
"""
Chat session persistence endpoints.

Schema (two tables):
  chat_sessions  - metadata (title, collections, timestamps)
  chat_messages  - one row per message (FK to chat_sessions, CASCADE delete)

Falls back to in-memory dict if DATABASE_URL is not set or DB is unreachable.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timezone
import logging
import uuid
import os

from router.auth import get_current_user, UserRecord

router = APIRouter()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class StoredMessage(BaseModel):
    id: str
    role: str          # "user" | "assistant"
    content: str
    model_used: Optional[str] = None
    created_at: str    # ISO

class UpsertSessionRequest(BaseModel):
    session_id: Optional[str] = None   # if None -> create new
    title: str
    messages: List[StoredMessage]
    pdf_collections: Optional[List[str]] = []
    chat_collections: Optional[List[str]] = []

class SessionResponse(BaseModel):
    session_id: str
    title: str
    created_at: str
    updated_at: str
    messages: List[StoredMessage]
    pdf_collections: List[str]
    chat_collections: List[str]

class SessionSummary(BaseModel):
    session_id: str
    title: str
    message_count: int
    created_at: str
    updated_at: str
    pdf_collections: List[str]
    chat_collections: List[str]

class RenameSessionRequest(BaseModel):
    title: str


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _get_conn():
    """Return a psycopg2 RealDictCursor connection or None if unavailable."""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        return None
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
        # Embed sslmode in URL to avoid kwarg conflict with Supabase pooler DSN
        url = database_url
        if "sslmode=" not in url:
            sep = "&" if "?" in url else "?"
            url = url + sep + "sslmode=require"
        conn = psycopg2.connect(url, cursor_factory=RealDictCursor, connect_timeout=10)
        conn.autocommit = True
        return conn
    except Exception as e:
        logger.warning("sessions: DB connection failed: %s", e)
        return None


_tables_ensured = False


def _ensure_tables(conn):
    """Create chat_sessions + chat_messages tables if they do not exist."""
    global _tables_ensured
    if _tables_ensured:
        return
    try:
        with conn.cursor() as cur:
            cur.execute("""
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
            """)
        _tables_ensured = True
        logger.info("sessions: Schema ensured successfully.")
    except Exception as e:
        logger.warning("sessions: ensure tables failed: %s", e)


def _ts(val) -> str:
    """Convert a datetime or string to ISO string."""
    if val is None:
        return datetime.now(timezone.utc).isoformat()
    if hasattr(val, "isoformat"):
        return val.isoformat()
    return str(val)


def _get_required_conn():
    conn = _get_conn()
    if not conn:
        raise HTTPException(status_code=503, detail="Session database unavailable")
    _ensure_tables(conn)
    return conn


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/sessions", response_model=SessionResponse)
async def upsert_session(
    body: UpsertSessionRequest,
    user: UserRecord = Depends(get_current_user),
):
    """
    Create or update a session.
    Session metadata goes into chat_sessions.
    Each message is upserted into chat_messages (by message_id).
    """
    sid = body.session_id or str(uuid.uuid4())
    conn = _get_required_conn()
    try:
        with conn.cursor() as cur:
            if body.session_id:
                cur.execute(
                    "SELECT owner_id FROM chat_sessions WHERE session_id = %s",
                    (sid,),
                )
                existing = cur.fetchone()
                if (
                    existing
                    and existing.get("owner_id")
                    and existing["owner_id"] != user.user_id
                    and user.role != "admin"
                ):
                    raise HTTPException(status_code=403, detail="Not allowed to modify this session")

            cur.execute("""
                INSERT INTO chat_sessions
                    (session_id, title, pdf_collections, chat_collections, owner_id, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, now(), now())
                ON CONFLICT (session_id) DO UPDATE
                    SET title            = EXCLUDED.title,
                        pdf_collections  = EXCLUDED.pdf_collections,
                        chat_collections = EXCLUDED.chat_collections,
                        owner_id         = COALESCE(chat_sessions.owner_id, EXCLUDED.owner_id),
                        updated_at       = now()
                RETURNING session_id, title, pdf_collections, chat_collections,
                          created_at, updated_at
            """, (
                sid,
                body.title,
                body.pdf_collections or [],
                body.chat_collections or [],
                user.user_id,
            ))
            session_row = cur.fetchone()

            for m in body.messages:
                cur.execute("""
                    INSERT INTO chat_messages
                        (message_id, session_id, role, content, model_used, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (message_id) DO UPDATE
                        SET content    = EXCLUDED.content,
                            model_used = EXCLUDED.model_used
                """, (
                    m.id,
                    sid,
                    m.role,
                    m.content,
                    m.model_used,
                    m.created_at,
                ))

            cur.execute("""
                SELECT message_id, role, content, model_used, created_at
                FROM chat_messages
                WHERE session_id = %s
                ORDER BY created_at ASC, id ASC
            """, (sid,))
            msg_rows = cur.fetchall()

        return SessionResponse(
            session_id=session_row["session_id"],
            title=session_row["title"],
            created_at=_ts(session_row["created_at"]),
            updated_at=_ts(session_row["updated_at"]),
            messages=[
                StoredMessage(
                    id=r["message_id"],
                    role=r["role"],
                    content=r["content"],
                    model_used=r["model_used"],
                    created_at=_ts(r["created_at"]),
                )
                for r in msg_rows
            ],
            pdf_collections=list(session_row["pdf_collections"] or []),
            chat_collections=list(session_row["chat_collections"] or []),
        )
    except Exception as e:
        logger.error("sessions upsert DB error: %s", e)
        raise HTTPException(status_code=500, detail="Failed to persist session")
    finally:
        try:
            conn.close()
        except Exception:
            pass


@router.get("/sessions", response_model=List[SessionSummary])
async def list_sessions(user: UserRecord = Depends(get_current_user)):
    """Return sessions owned by the current user (all sessions for admins),
    ordered by most recent, with message counts."""
    conn = _get_required_conn()
    try:
        with conn.cursor() as cur:
            if user.role == "admin":
                cur.execute("""
                    SELECT s.session_id,
                           s.title,
                           s.pdf_collections,
                           s.chat_collections,
                           s.created_at,
                           s.updated_at,
                           COUNT(m.id) AS message_count
                    FROM chat_sessions s
                    LEFT JOIN chat_messages m ON m.session_id = s.session_id
                    GROUP BY s.session_id, s.title, s.pdf_collections,
                             s.chat_collections, s.created_at, s.updated_at
                    ORDER BY s.updated_at DESC
                    LIMIT 200
                """)
            else:
                cur.execute("""
                    SELECT s.session_id,
                           s.title,
                           s.pdf_collections,
                           s.chat_collections,
                           s.created_at,
                           s.updated_at,
                           COUNT(m.id) AS message_count
                    FROM chat_sessions s
                    LEFT JOIN chat_messages m ON m.session_id = s.session_id
                    WHERE s.owner_id = %s
                    GROUP BY s.session_id, s.title, s.pdf_collections,
                             s.chat_collections, s.created_at, s.updated_at
                    ORDER BY s.updated_at DESC
                    LIMIT 200
                """, (user.user_id,))
            rows = cur.fetchall()
        return [
            SessionSummary(
                session_id=r["session_id"],
                title=r["title"],
                message_count=int(r["message_count"] or 0),
                created_at=_ts(r["created_at"]),
                updated_at=_ts(r["updated_at"]),
                pdf_collections=list(r["pdf_collections"] or []),
                chat_collections=list(r["chat_collections"] or []),
            )
            for r in rows
        ]
    except Exception as e:
        logger.error("sessions list DB error: %s", e)
        raise HTTPException(status_code=500, detail="Failed to list sessions")
    finally:
        try:
            conn.close()
        except Exception:
            pass


@router.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str, user: UserRecord = Depends(get_current_user)):
    """Return full session including all messages from chat_messages table."""
    conn = _get_required_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT session_id, title, pdf_collections, chat_collections,
                       owner_id, created_at, updated_at
                FROM chat_sessions WHERE session_id = %s
            """, (session_id,))
            session_row = cur.fetchone()
            if not session_row:
                raise HTTPException(status_code=404, detail="Session not found")
            if (
                user.role != "admin"
                and session_row.get("owner_id")
                and session_row["owner_id"] != user.user_id
            ):
                raise HTTPException(status_code=403, detail="Not allowed to access this session")

            cur.execute("""
                SELECT message_id, role, content, model_used, created_at
                FROM chat_messages
                WHERE session_id = %s
                ORDER BY created_at ASC, id ASC
            """, (session_id,))
            msg_rows = cur.fetchall()
        return SessionResponse(
            session_id=session_row["session_id"],
            title=session_row["title"],
            created_at=_ts(session_row["created_at"]),
            updated_at=_ts(session_row["updated_at"]),
            messages=[
                StoredMessage(
                    id=r["message_id"],
                    role=r["role"],
                    content=r["content"],
                    model_used=r["model_used"],
                    created_at=_ts(r["created_at"]),
                )
                for r in msg_rows
            ],
            pdf_collections=list(session_row["pdf_collections"] or []),
            chat_collections=list(session_row["chat_collections"] or []),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("sessions get DB error: %s", e)
        raise HTTPException(status_code=500, detail="Failed to load session")
    finally:
        try:
            conn.close()
        except Exception:
            pass


@router.put("/sessions/{session_id}", response_model=SessionSummary)
async def rename_session(
    session_id: str,
    body: RenameSessionRequest,
    user: UserRecord = Depends(get_current_user),
):
    """Rename a session. Title only — messages are untouched, so this never
    needs the client to resend the whole conversation just to relabel it."""
    title = body.title.strip()
    if not title:
        raise HTTPException(status_code=400, detail="Title cannot be empty")
    conn = _get_required_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT owner_id FROM chat_sessions WHERE session_id = %s",
                (session_id,),
            )
            existing = cur.fetchone()
            if not existing:
                raise HTTPException(status_code=404, detail="Session not found")
            if (
                user.role != "admin"
                and existing.get("owner_id")
                and existing["owner_id"] != user.user_id
            ):
                raise HTTPException(status_code=403, detail="Not allowed to rename this session")

            cur.execute("""
                UPDATE chat_sessions
                SET title = %s, updated_at = now()
                WHERE session_id = %s
                RETURNING session_id, title, pdf_collections, chat_collections,
                          created_at, updated_at
            """, (title, session_id))
            session_row = cur.fetchone()

            cur.execute(
                "SELECT COUNT(*) AS message_count FROM chat_messages WHERE session_id = %s",
                (session_id,),
            )
            message_count = cur.fetchone()["message_count"]

        return SessionSummary(
            session_id=session_row["session_id"],
            title=session_row["title"],
            message_count=int(message_count or 0),
            created_at=_ts(session_row["created_at"]),
            updated_at=_ts(session_row["updated_at"]),
            pdf_collections=list(session_row["pdf_collections"] or []),
            chat_collections=list(session_row["chat_collections"] or []),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("sessions rename DB error: %s", e)
        raise HTTPException(status_code=500, detail="Failed to rename session")
    finally:
        try:
            conn.close()
        except Exception:
            pass


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, user: UserRecord = Depends(get_current_user)):
    """Delete a session. Messages are cascade-deleted automatically."""
    conn = _get_required_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT owner_id FROM chat_sessions WHERE session_id = %s",
                (session_id,),
            )
            existing = cur.fetchone()
            if not existing:
                raise HTTPException(status_code=404, detail="Session not found")
            if (
                user.role != "admin"
                and existing.get("owner_id")
                and existing["owner_id"] != user.user_id
            ):
                raise HTTPException(status_code=403, detail="Not allowed to delete this session")
            cur.execute("DELETE FROM chat_sessions WHERE session_id = %s", (session_id,))
        return {"status": "deleted", "session_id": session_id}
    except Exception as e:
        logger.error("sessions delete DB error: %s", e)
        raise HTTPException(status_code=500, detail="Failed to delete session")
    finally:
        try:
            conn.close()
        except Exception:
            pass
