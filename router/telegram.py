# router/telegram.py
"""
Telegram as a live chat source (Telethon/MTProto), not a file upload.

Flow: connect (api_id + api_hash + phone -> OTP -> optional 2FA) -> pick
which chats/groups to bring in -> sync (repeatable) -> each selected chat
becomes a searchable ChatCollection via the same pipeline WhatsApp uploads
use (chat_ingest.py).

api_id/api_hash (from https://my.telegram.org/apps) are entered per-connection
in the connect form rather than via server config, so different admins can
each bring their own Telegram app registration — see
models.TelegramConnectStartRequest.

Mirrors router/database_connections.py's structure (its own small schema in
the app's own database, admin-only, connect/list/activate/delete) since a
Telegram connection is the same shape of thing: a live, credentialed,
user-managed source rather than a one-time file.

SECURITY: a saved Telethon session is equivalent to full access to the
Telegram account that logged in — read every chat, send messages as that
user. Both the session and the connection's api_hash are encrypted at rest
(telegram_crypto.py) and never returned to the client. Losing
TELEGRAM_SESSION_ENCRYPTION_KEY makes stored secrets permanently
undecryptable (by design — there is no server-side escape hatch).

KNOWN LIMITATION: the connect flow's in-progress Telethon client lives in an
in-memory dict between /connect/start and /connect/verify. It does not
survive a backend restart or a multi-worker deployment — acceptable for this
app's single-process deployment, same assumption processor.py's in-memory
vector_store_cache already makes.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status

import app_db
from config import config
from models import (
    ChatMessage,
    ChatPlatform,
    SetTelegramConnectionActiveRequest,
    TelegramConnectionsResponse,
    TelegramConnectionSource,
    TelegramConnectStartRequest,
    TelegramConnectStartResponse,
    TelegramConnectVerifyRequest,
    TelegramConnectVerifyResponse,
    TelegramDialog,
    TelegramDialogsResponse,
    TelegramSelectedChat,
    TelegramSyncRequest,
    TelegramSyncResponse,
    TelegramSyncResult,
)
from chat_ingest import ingest_chat_messages
from telegram_crypto import encrypt_secret, decrypt_secret
from router.auth import require_role, UserRecord

router = APIRouter()
logger = logging.getLogger(__name__)

_tables_ensured = False
PENDING_LOGIN_TTL_SECONDS = 10 * 60

# flow_id -> {"client": TelegramClient, "phone": str, "phone_code_hash": str,
#             "label": Optional[str], "created_at": float}
_pending_logins: Dict[str, Dict[str, Any]] = {}


def _require_telegram_enabled() -> None:
    if not config.telegram_enabled:
        raise HTTPException(
            status_code=503,
            detail="Telegram integration is not configured — set TELEGRAM_SESSION_ENCRYPTION_KEY in the backend .env",
        )


def _cleanup_pending_logins() -> None:
    now = time.time()
    expired = [fid for fid, entry in _pending_logins.items() if now - entry["created_at"] > PENDING_LOGIN_TTL_SECONDS]
    for fid in expired:
        entry = _pending_logins.pop(fid, None)
        if entry and entry.get("client"):
            try:
                entry["client"].disconnect()
            except Exception:
                pass


def _mask_phone(phone: str) -> str:
    digits = phone.strip()
    if len(digits) <= 6:
        return digits[0] + "*" * max(len(digits) - 1, 0)
    # Always mask at least one character between the kept head/tail — at
    # exactly 7 chars, len(digits) - 7 == 0 would otherwise leave nothing masked.
    return f"{digits[:4]}{'*' * max(len(digits) - 7, 1)}{digits[-3:]}"


def _dialog_type(entity: Any) -> str:
    if getattr(entity, "broadcast", False):
        return "channel"
    if getattr(entity, "megagroup", False) or type(entity).__name__ == "Chat":
        return "group"
    if type(entity).__name__ == "User":
        return "user"
    return "channel"


# ---------------------------------------------------------------------------
# App-DB storage (connection metadata + selected chats — NOT the chat content
# itself, which flows through chat_ingest.py into chat_collections like any
# other chat source).
# ---------------------------------------------------------------------------

def _telegram_user_id() -> str:
    return os.getenv("DB_CONNECTIONS_USER_ID", os.getenv("DEV_USER_ID", "db-connections-local-user"))


def _get_app_conn():
    return app_db.get_app_conn("telegram")


def _ensure_tables(conn) -> None:
    global _tables_ensured
    if _tables_ensured:
        return
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS telegram_connections (
                    id             BIGSERIAL   PRIMARY KEY,
                    connection_id  TEXT        NOT NULL UNIQUE,
                    user_id        TEXT        NOT NULL,
                    api_id         BIGINT      NOT NULL,
                    api_hash_enc   TEXT        NOT NULL,
                    phone          TEXT        NOT NULL,
                    label          TEXT        NOT NULL,
                    session_enc    TEXT        NOT NULL,
                    status         TEXT        NOT NULL DEFAULT 'active'
                                   CHECK (status IN ('active', 'inactive')),
                    last_synced_at TIMESTAMPTZ,
                    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                CREATE INDEX IF NOT EXISTS idx_telegram_connections_user
                    ON telegram_connections (user_id, created_at DESC);

                CREATE TABLE IF NOT EXISTS telegram_selected_chats (
                    id                  BIGSERIAL   PRIMARY KEY,
                    connection_id       TEXT        NOT NULL REFERENCES telegram_connections(connection_id) ON DELETE CASCADE,
                    dialog_id           TEXT        NOT NULL,
                    dialog_title        TEXT        NOT NULL,
                    dialog_type         TEXT        NOT NULL,
                    chat_collection_id  TEXT,
                    status              TEXT        NOT NULL DEFAULT 'active'
                                        CHECK (status IN ('active', 'inactive')),
                    last_synced_at      TIMESTAMPTZ,
                    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
                    UNIQUE (connection_id, dialog_id)
                );
                CREATE INDEX IF NOT EXISTS idx_telegram_selected_chats_connection
                    ON telegram_selected_chats (connection_id);
                """
            )
        _tables_ensured = True
    except Exception as exc:
        logger.error("telegram: ensure tables failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to initialize Telegram schema")


def _selected_chats_for(cur, connection_id: str) -> List[TelegramSelectedChat]:
    cur.execute(
        """
        SELECT dialog_id, dialog_title, dialog_type, chat_collection_id, status, last_synced_at
        FROM telegram_selected_chats
        WHERE connection_id = %s
        ORDER BY created_at ASC
        """,
        (connection_id,),
    )
    rows = cur.fetchall() or []
    return [
        TelegramSelectedChat(
            dialog_id=r["dialog_id"],
            title=r["dialog_title"],
            type=r["dialog_type"],
            chat_collection_id=r.get("chat_collection_id"),
            status=r["status"],
            last_synced_at=app_db.ts(r["last_synced_at"]) if r.get("last_synced_at") else None,
        )
        for r in rows
    ]


def _as_connection_source(cur, row: Dict[str, Any]) -> TelegramConnectionSource:
    return TelegramConnectionSource(
        connection_id=row["connection_id"],
        label=row["label"],
        phone_masked=_mask_phone(row["phone"]),
        status=row["status"],
        created_at=app_db.ts(row["created_at"]),
        selected_chats=_selected_chats_for(cur, row["connection_id"]),
    )


def _get_connection_row(connection_id: str) -> Optional[Dict[str, Any]]:
    conn = _get_app_conn()
    if not conn:
        raise HTTPException(status_code=503, detail="Database unavailable")
    _ensure_tables(conn)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM telegram_connections WHERE connection_id = %s",
                (connection_id,),
            )
            return cur.fetchone()
    finally:
        conn.close()


def _upsert_selected_chat(connection_id: str, dialog_id: str, title: str, dialog_type: str, chat_collection_id: str) -> None:
    conn = _get_app_conn()
    if not conn:
        return
    _ensure_tables(conn)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO telegram_selected_chats
                    (connection_id, dialog_id, dialog_title, dialog_type, chat_collection_id, last_synced_at)
                VALUES (%s, %s, %s, %s, %s, now())
                ON CONFLICT (connection_id, dialog_id) DO UPDATE SET
                    dialog_title       = EXCLUDED.dialog_title,
                    dialog_type        = EXCLUDED.dialog_type,
                    chat_collection_id = EXCLUDED.chat_collection_id,
                    last_synced_at     = now()
                """,
                (connection_id, dialog_id, title, dialog_type, chat_collection_id),
            )
    finally:
        conn.close()


def _get_selected_chat_collection_id(connection_id: str, dialog_id: str) -> Optional[str]:
    conn = _get_app_conn()
    if not conn:
        return None
    _ensure_tables(conn)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT chat_collection_id FROM telegram_selected_chats WHERE connection_id = %s AND dialog_id = %s",
                (connection_id, dialog_id),
            )
            row = cur.fetchone()
            return row["chat_collection_id"] if row else None
    finally:
        conn.close()


def _touch_connection_synced(connection_id: str) -> None:
    conn = _get_app_conn()
    if not conn:
        return
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE telegram_connections SET last_synced_at = now() WHERE connection_id = %s",
                (connection_id,),
            )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Telethon client helpers
# ---------------------------------------------------------------------------

def _new_client(api_id: int, api_hash: str, session_string: Optional[str] = None):
    from telethon import TelegramClient
    from telethon.sessions import StringSession

    return TelegramClient(StringSession(session_string), api_id, api_hash)


async def _connect_with_saved_session(row: Dict[str, Any]):
    session_string = decrypt_secret(row["session_enc"])
    api_hash = decrypt_secret(row["api_hash_enc"])
    if session_string is None or api_hash is None:
        raise HTTPException(
            status_code=500,
            detail="Could not decrypt the saved Telegram credentials — reconnect this account.",
        )
    client = _new_client(row["api_id"], api_hash, session_string)
    await client.connect()
    if not await client.is_user_authorized():
        await client.disconnect()
        raise HTTPException(status_code=401, detail="Telegram session expired — reconnect this account.")
    return client


# ---------------------------------------------------------------------------
# Routes — connect flow
# ---------------------------------------------------------------------------

@router.post("/telegram/connect/start", response_model=TelegramConnectStartResponse)
async def telegram_connect_start(
    body: TelegramConnectStartRequest,
    _: UserRecord = Depends(require_role("admin")),
):
    _require_telegram_enabled()
    _cleanup_pending_logins()

    phone = body.phone.strip()
    if not phone:
        raise HTTPException(status_code=400, detail="Phone number is required")
    if not body.api_hash.strip():
        raise HTTPException(status_code=400, detail="API hash is required")

    client = _new_client(body.api_id, body.api_hash.strip())
    try:
        await client.connect()
        sent = await client.send_code_request(phone)
    except Exception as exc:
        try:
            await client.disconnect()
        except Exception:
            pass
        logger.error("telegram: send_code_request failed: %s", exc)
        raise HTTPException(status_code=400, detail=f"Could not send login code: {exc}")

    flow_id = str(uuid.uuid4())
    _pending_logins[flow_id] = {
        "client": client,
        "api_id": body.api_id,
        "api_hash": body.api_hash.strip(),
        "phone": phone,
        "phone_code_hash": sent.phone_code_hash,
        "label": (body.label or "").strip() or None,
        "created_at": time.time(),
    }
    return TelegramConnectStartResponse(flow_id=flow_id, phone=phone)


@router.post("/telegram/connect/verify", response_model=TelegramConnectVerifyResponse)
async def telegram_connect_verify(
    body: TelegramConnectVerifyRequest,
    user: UserRecord = Depends(require_role("admin")),
):
    _require_telegram_enabled()

    entry = _pending_logins.get(body.flow_id)
    if not entry:
        raise HTTPException(status_code=400, detail="Login session expired or not found — start over.")

    client = entry["client"]
    from telethon.errors import SessionPasswordNeededError, PhoneCodeInvalidError, PhoneCodeExpiredError

    try:
        if body.password:
            await client.sign_in(password=body.password)
        else:
            await client.sign_in(
                phone=entry["phone"],
                code=body.code,
                phone_code_hash=entry["phone_code_hash"],
            )
    except SessionPasswordNeededError:
        return TelegramConnectVerifyResponse(status="password_required", connection=None)
    except (PhoneCodeInvalidError, PhoneCodeExpiredError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid or expired code: {exc}")
    except Exception as exc:
        logger.error("telegram: sign_in failed: %s", exc)
        raise HTTPException(status_code=400, detail=f"Sign-in failed: {exc}")

    session_string = client.session.save()
    await client.disconnect()
    _pending_logins.pop(body.flow_id, None)

    conn = _get_app_conn()
    if not conn:
        raise HTTPException(status_code=503, detail="Database unavailable")
    _ensure_tables(conn)
    connection_id = str(uuid.uuid4())
    label = entry["label"] or f"Telegram ({_mask_phone(entry['phone'])})"
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO telegram_connections
                    (connection_id, user_id, api_id, api_hash_enc, phone, label, session_enc, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, 'active')
                RETURNING connection_id, label, phone, status, created_at
                """,
                (
                    connection_id,
                    _telegram_user_id(),
                    entry["api_id"],
                    encrypt_secret(entry["api_hash"]),
                    entry["phone"],
                    label,
                    encrypt_secret(session_string),
                ),
            )
            row = cur.fetchone()
            source = _as_connection_source(cur, row)
    except Exception as exc:
        logger.error("telegram: failed to persist connection: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to save Telegram connection")
    finally:
        conn.close()

    return TelegramConnectVerifyResponse(status="connected", connection=source)


# ---------------------------------------------------------------------------
# Routes — manage connections
# ---------------------------------------------------------------------------

@router.get("/telegram/connections", response_model=TelegramConnectionsResponse)
async def list_telegram_connections(_: UserRecord = Depends(require_role("admin"))):
    conn = _get_app_conn()
    if not conn:
        raise HTTPException(status_code=503, detail="Database unavailable")
    _ensure_tables(conn)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM telegram_connections ORDER BY created_at DESC",
            )
            rows = cur.fetchall() or []
            connections = [_as_connection_source(cur, r) for r in rows]
        return TelegramConnectionsResponse(connections=connections, count=len(connections))
    except Exception as exc:
        logger.error("telegram: list connections failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to list Telegram connections")
    finally:
        conn.close()


@router.get("/telegram-connection/{connection_id}/dialogs", response_model=TelegramDialogsResponse)
async def list_telegram_dialogs(
    connection_id: str,
    _: UserRecord = Depends(require_role("admin")),
):
    _require_telegram_enabled()
    row = _get_connection_row(connection_id)
    if not row:
        raise HTTPException(status_code=404, detail="Telegram connection not found")

    client = await _connect_with_saved_session(row)
    try:
        dialogs = await client.get_dialogs(limit=200)
        result = [
            TelegramDialog(
                dialog_id=str(d.id),
                title=d.name or str(d.id),
                type=_dialog_type(d.entity),
                participants_count=getattr(d.entity, "participants_count", None),
            )
            for d in dialogs
        ]
    finally:
        await client.disconnect()

    return TelegramDialogsResponse(dialogs=result, count=len(result))


@router.post("/telegram-connection/{connection_id}/sync", response_model=TelegramSyncResponse)
async def sync_telegram_connection(
    connection_id: str,
    body: TelegramSyncRequest,
    _: UserRecord = Depends(require_role("admin")),
):
    _require_telegram_enabled()
    row = _get_connection_row(connection_id)
    if not row:
        raise HTTPException(status_code=404, detail="Telegram connection not found")
    if not body.dialog_ids:
        raise HTTPException(status_code=400, detail="Select at least one chat to sync")

    client = await _connect_with_saved_session(row)
    results: List[TelegramSyncResult] = []

    try:
        for dialog_id_str in body.dialog_ids:
            try:
                dialog_id = int(dialog_id_str)
                entity = await client.get_entity(dialog_id)
                title = getattr(entity, "title", None) or getattr(entity, "first_name", None) or dialog_id_str

                sender_names: Dict[int, str] = {}
                try:
                    participants = await client.get_participants(entity, limit=200)
                    for p in participants:
                        name = " ".join(filter(None, [getattr(p, "first_name", None), getattr(p, "last_name", None)]))
                        sender_names[p.id] = name or (getattr(p, "username", None) or str(p.id))
                except Exception:
                    pass  # channels/broadcasts often can't list participants — fall back to sender id below

                messages: List[ChatMessage] = []
                async for msg in client.iter_messages(entity, limit=body.message_limit):
                    text = msg.text or msg.message
                    if not text:
                        continue
                    sender_name = sender_names.get(msg.sender_id, str(msg.sender_id) if msg.sender_id else title)
                    messages.append(
                        ChatMessage(
                            message_id=f"tg_{dialog_id}_{msg.id}",
                            sender=sender_name,
                            timestamp=msg.date.replace(tzinfo=None) if msg.date else datetime.now(),
                            content=text,
                            platform=ChatPlatform.TELEGRAM,
                        )
                    )
                messages.reverse()  # iter_messages yields newest-first

                if not messages:
                    results.append(
                        TelegramSyncResult(
                            dialog_id=dialog_id_str, title=title, chat_collection_id="",
                            message_count=0, status="error", error="No text messages found in this chat",
                        )
                    )
                    continue

                existing_collection_id = _get_selected_chat_collection_id(connection_id, dialog_id_str)
                collection_id = existing_collection_id or str(uuid.uuid4())

                collection = await ingest_chat_messages(
                    collection_id,
                    messages,
                    file_name=f"telegram-{title}",
                    platform=ChatPlatform.TELEGRAM,
                )

                _upsert_selected_chat(connection_id, dialog_id_str, title, _dialog_type(entity), collection_id)

                results.append(
                    TelegramSyncResult(
                        dialog_id=dialog_id_str, title=title, chat_collection_id=collection_id,
                        message_count=collection.message_count, status="success",
                    )
                )
            except Exception as exc:
                logger.error("telegram: sync failed for dialog %s: %s", dialog_id_str, exc)
                results.append(
                    TelegramSyncResult(
                        dialog_id=dialog_id_str, title=dialog_id_str, chat_collection_id="",
                        message_count=0, status="error", error=str(exc),
                    )
                )
    finally:
        await client.disconnect()

    _touch_connection_synced(connection_id)
    return TelegramSyncResponse(results=results)


@router.post("/telegram-connection/activate")
async def set_telegram_connection_active(
    body: SetTelegramConnectionActiveRequest,
    _: UserRecord = Depends(require_role("admin")),
):
    conn = _get_app_conn()
    if not conn:
        raise HTTPException(status_code=503, detail="Database unavailable")
    _ensure_tables(conn)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE telegram_connections SET status = %s WHERE connection_id = %s RETURNING connection_id",
                ("active" if body.active else "inactive", body.connection_id),
            )
            updated = cur.fetchone()
        if not updated:
            raise HTTPException(status_code=404, detail="Telegram connection not found")
        return {"status": "success", "connection_id": body.connection_id, "active": body.active}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("telegram: activate failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to update active status")
    finally:
        conn.close()


@router.delete("/telegram-connection/{connection_id}")
async def delete_telegram_connection(
    connection_id: str,
    _: UserRecord = Depends(require_role("admin")),
):
    """Revoke the saved session. Already-synced chats stay searchable as
    regular chat collections — deleting the connection only stops future
    syncing, same as deleting a database connection doesn't unsync past
    query results."""
    conn = _get_app_conn()
    if not conn:
        raise HTTPException(status_code=503, detail="Database unavailable")
    _ensure_tables(conn)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM telegram_connections WHERE connection_id = %s RETURNING connection_id",
                (connection_id,),
            )
            deleted = cur.fetchone()
        if not deleted:
            raise HTTPException(status_code=404, detail="Telegram connection not found")
        return {"status": "success", "message": "Telegram connection deleted", "connection_id": connection_id}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("telegram: delete failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to delete Telegram connection")
    finally:
        conn.close()
