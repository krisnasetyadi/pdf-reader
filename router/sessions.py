# router/sessions.py
"""
Chat session persistence endpoints.

Schema (two tables):
  chat_sessions  - metadata (title, collections, timestamps)
  chat_messages  - one row per message (FK to chat_sessions, CASCADE delete)

Falls back to in-memory dict if DATABASE_URL is not set or DB is unreachable.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timezone
import logging
import uuid
import os

# MS-237: how many *chats* GET /sessions/{id} returns per page. One chat is
# one question plus the answer(s) that followed it — not one message row — so
# a page never lands mid-exchange with a question whose answer is still on the
# next page.
PAGE_SIZE_DEFAULT = 5

# MS-237: how many questions the navigation index below returns at most. It's
# a flat list the client holds in memory for the hover panel, so it can't be
# unbounded; past this the newest ones are the ones worth keeping (turn
# numbers stay absolute, so the panel still labels them correctly).
QUESTION_INDEX_LIMIT = 1000

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
    # Optional past creation — omitted, the existing title (auto-derived or
    # since renamed via PUT /sessions/{id}) is left untouched (MS-253).
    title: Optional[str] = None
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
    # MS-237: only meaningful on the paginated GET below. POST (create/
    # update) always returns the full list it was given, so has_more is
    # correctly False there too — nothing more to page in.
    has_more: bool = False
    next_cursor: Optional[str] = None
    # Total user-authored messages in the session (loaded or not) — lets the
    # client draw one navigation marker per question, including ones it
    # hasn't fetched yet.
    total_user_turns: int = 0

class SessionSummary(BaseModel):
    session_id: str
    title: str
    message_count: int
    created_at: str
    updated_at: str
    pdf_collections: List[str]
    chat_collections: List[str]

class SessionQuestion(BaseModel):
    """One entry in the navigation index — a question the user asked, with
    just enough text to recognise it in a list."""
    turn: int          # 1-based, counted from the start of the session
    message_id: str
    preview: str


class SessionQuestionsResponse(BaseModel):
    session_id: str
    total: int         # every question in the session, even beyond the cap below
    questions: List[SessionQuestion]


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


def _parse_cursor(cursor: Optional[str]):
    """"<created_at ISO>|<row id>" -> (created_at, id), or (None, None) if
    absent/malformed — callers treat that as "start from the most recent"."""
    if not cursor:
        return None, None
    try:
        ts_str, id_str = cursor.rsplit("|", 1)
        return ts_str, int(id_str)
    except Exception:
        return None, None


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
                VALUES (%s, COALESCE(%s, 'Untitled conversation'), %s, %s, %s, now(), now())
                ON CONFLICT (session_id) DO UPDATE
                    SET title            = COALESCE(%s, chat_sessions.title),
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
                body.title,
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
            total_user_turns=sum(1 for r in msg_rows if r["role"] == "user"),
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
async def list_sessions(
    q: Optional[str] = None,
    user: UserRecord = Depends(get_current_user),
):
    """Return sessions owned by the current user (all sessions for admins),
    ordered by most recent, with message counts.

    If `q` is given, also matches sessions whose title OR any message
    content contains it (case-insensitive), not just the title.
    """
    conn = _get_required_conn()
    try:
        search_clause = ""
        params: tuple = ()
        if q:
            # Escape backslashes first — ILIKE's default escape char is `\`,
            # so an unescaped literal backslash in the query would otherwise
            # consume the next character (including the wildcards we add).
            escaped = (
                q.replace(chr(92), chr(92) * 2)
                .replace("%", chr(92) + "%")
                .replace("_", chr(92) + "_")
            )
            pattern = f"%{escaped}%"
            search_clause = """
                AND (
                    s.title ILIKE %s
                    OR EXISTS (
                        SELECT 1 FROM chat_messages cm
                        WHERE cm.session_id = s.session_id AND cm.content ILIKE %s
                    )
                )
            """
            params = (pattern, pattern)

        with conn.cursor() as cur:
            if user.role == "admin":
                cur.execute(f"""
                    SELECT s.session_id,
                           s.title,
                           s.pdf_collections,
                           s.chat_collections,
                           s.created_at,
                           s.updated_at,
                           COUNT(m.id) AS message_count
                    FROM chat_sessions s
                    LEFT JOIN chat_messages m ON m.session_id = s.session_id
                    WHERE TRUE {search_clause}
                    GROUP BY s.session_id, s.title, s.pdf_collections,
                             s.chat_collections, s.created_at, s.updated_at
                    ORDER BY s.updated_at DESC
                    LIMIT 200
                """, params)
            else:
                cur.execute(f"""
                    SELECT s.session_id,
                           s.title,
                           s.pdf_collections,
                           s.chat_collections,
                           s.created_at,
                           s.updated_at,
                           COUNT(m.id) AS message_count
                    FROM chat_sessions s
                    LEFT JOIN chat_messages m ON m.session_id = s.session_id
                    WHERE s.owner_id = %s {search_clause}
                    GROUP BY s.session_id, s.title, s.pdf_collections,
                             s.chat_collections, s.created_at, s.updated_at
                    ORDER BY s.updated_at DESC
                    LIMIT 200
                """, (user.user_id,) + params)
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
async def get_session(
    session_id: str,
    limit: int = Query(PAGE_SIZE_DEFAULT, ge=1, le=100),
    before: Optional[str] = None,
    user: UserRecord = Depends(get_current_user),
):
    """Return a session's most recent `limit` chats — one chat being one
    question plus the answer(s) that followed it, so `limit=5` is 5 complete
    exchanges, not 5 message rows. MS-237: was "all messages, always"; a
    long-lived conversation no longer dumps its entire history into one
    response. Pass the previous response's `next_cursor` as `before` to page
    further back; `has_more`/`next_cursor` come back falsy once the start of
    the conversation is reached."""
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

            # Page by chat, not by message row: number every message with a
            # running count of the questions asked up to and including it
            # (`chat_no`), then keep only the last `limit` of those numbers.
            # A question and the answer(s) that followed it share one
            # chat_no, so they always travel together — a plain
            # "LIMIT 5 rows" would cut between them and strand an answer
            # without its question (or vice versa) on the page boundary.
            before_ts, before_id = _parse_cursor(before)
            chat_page_sql = """
                WITH ordered AS (
                    SELECT message_id, role, content, model_used, created_at, id,
                           SUM(CASE WHEN role = 'user' THEN 1 ELSE 0 END)
                               OVER (ORDER BY created_at ASC, id ASC) AS chat_no
                    FROM chat_messages
                    WHERE session_id = %s
                    {cursor_clause}
                )
                SELECT message_id, role, content, model_used, created_at, id
                FROM ordered
                WHERE chat_no > (SELECT COALESCE(MAX(chat_no), 0) FROM ordered) - %s
                ORDER BY created_at ASC, id ASC
            """
            if before_ts is not None and before_id is not None:
                cur.execute(
                    chat_page_sql.format(cursor_clause="AND (created_at, id) < (%s, %s)"),
                    (session_id, before_ts, before_id, limit),
                )
            else:
                cur.execute(
                    chat_page_sql.format(cursor_clause=""),
                    (session_id, limit),
                )
            msg_rows = cur.fetchall()

            # Anything still older than the page we just built means there's
            # another page behind it. Asked as its own EXISTS rather than
            # over-fetching a row, since the page size is counted in chats
            # and a "+1 row" trick can't tell a whole extra chat from the
            # tail of the current one.
            has_more = False
            if msg_rows:
                oldest = msg_rows[0]
                cur.execute("""
                    SELECT EXISTS(
                        SELECT 1 FROM chat_messages
                        WHERE session_id = %s AND (created_at, id) < (%s, %s)
                    ) AS more
                """, (session_id, oldest["created_at"], oldest["id"]))
                has_more = bool(cur.fetchone()["more"])

            # Needed so the client can draw a navigation marker for every
            # question ever asked in this session, including ones it hasn't
            # paged in yet (MS-237 poin 9: click an unloaded marker -> fetch
            # the pages between here and there, then scroll to it).
            cur.execute(
                "SELECT COUNT(*) AS n FROM chat_messages WHERE session_id = %s AND role = 'user'",
                (session_id,),
            )
            total_user_turns = int(cur.fetchone()["n"] or 0)

        next_cursor = None
        if has_more and msg_rows:
            oldest = msg_rows[0]
            next_cursor = f"{_ts(oldest['created_at'])}|{oldest['id']}"

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
            has_more=has_more,
            next_cursor=next_cursor,
            total_user_turns=total_user_turns,
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


@router.get("/sessions/{session_id}/questions", response_model=SessionQuestionsResponse)
async def get_session_questions(
    session_id: str,
    user: UserRecord = Depends(get_current_user),
):
    """Every question asked in this session, as a flat index: turn number,
    the message id to scroll to, and a short preview.

    MS-237: this backs the chat navigation panel, which lists the whole
    conversation while the thread itself is still paginated 5 chats at a
    time. Content is truncated in SQL so the response stays small no matter
    how long the individual messages are — the panel only ever shows one
    line per question."""
    conn = _get_required_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT session_id, owner_id FROM chat_sessions WHERE session_id = %s",
                (session_id,),
            )
            session_row = cur.fetchone()
            if not session_row:
                raise HTTPException(status_code=404, detail="Session not found")
            if (
                user.role != "admin"
                and session_row.get("owner_id")
                and session_row["owner_id"] != user.user_id
            ):
                raise HTTPException(status_code=403, detail="Not allowed to access this session")

            # Numbered over the whole session first, then cut to the newest
            # QUESTION_INDEX_LIMIT — so a capped response still reports each
            # question's real turn number rather than renumbering from 1.
            cur.execute("""
                WITH numbered AS (
                    SELECT message_id,
                           LEFT(content, 120) AS preview,
                           ROW_NUMBER() OVER (ORDER BY created_at ASC, id ASC) AS turn,
                           created_at, id
                    FROM chat_messages
                    WHERE session_id = %s AND role = 'user'
                )
                SELECT message_id, preview, turn FROM (
                    SELECT * FROM numbered ORDER BY created_at DESC, id DESC LIMIT %s
                ) newest
                ORDER BY turn ASC
            """, (session_id, QUESTION_INDEX_LIMIT))
            rows = cur.fetchall()

            cur.execute(
                "SELECT COUNT(*) AS n FROM chat_messages WHERE session_id = %s AND role = 'user'",
                (session_id,),
            )
            total = int(cur.fetchone()["n"] or 0)

        return SessionQuestionsResponse(
            session_id=session_id,
            total=total,
            questions=[
                SessionQuestion(
                    turn=int(r["turn"]),
                    message_id=r["message_id"],
                    preview=r["preview"] or "",
                )
                for r in rows
            ],
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("sessions questions DB error: %s", e)
        raise HTTPException(status_code=500, detail="Failed to load session questions")
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
