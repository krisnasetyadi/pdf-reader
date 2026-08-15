from __future__ import annotations

from fastapi import APIRouter, HTTPException, status, Depends
from urllib.parse import urlparse
from typing import List, Dict, Any, Optional
import logging
import os
import uuid

import app_db
from router.auth import require_role, UserRecord

from models import (
    CreateDatabaseConnectionRequest,
    SetDatabaseConnectionActiveRequest,
    DbColumnInfo,
    DbTableInfo,
    DatabaseConnectionSource,
    DatabaseConnectionsResponse,
)

router = APIRouter()
logger = logging.getLogger(__name__)

_tables_ensured = False
MAX_TABLES_PER_CONNECTION = 30


def _db_connections_user_id() -> str:
    return os.getenv("DB_CONNECTIONS_USER_ID", os.getenv("DEV_USER_ID", "db-connections-local-user"))


def _enforce_scope() -> bool:
    raw = os.getenv("DB_CONNECTIONS_ENFORCE_SCOPE", "false")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _validate_postgres_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("postgresql", "postgres"):
        raise HTTPException(status_code=400, detail="URL harus diawali postgresql:// atau postgres://")
    if not parsed.hostname:
        raise HTTPException(status_code=400, detail="URL koneksi tidak valid")


def _derive_label(url: str) -> str:
    parsed = urlparse(url)
    db_name = (parsed.path or "").lstrip("/") or "database"
    return f"{parsed.hostname}/{db_name}"


def _redact_url(url: str) -> str:
    """Mask the password before the URL is ever sent back to the client."""
    try:
        parsed = urlparse(url)
        if parsed.password:
            netloc = parsed.netloc.replace(f":{parsed.password}@", ":****@")
            return parsed._replace(netloc=netloc).geturl()
        return url
    except Exception:
        return url


def _get_app_conn():
    """Connection to THIS app's own database (metadata store, not a data source)."""
    return app_db.get_app_conn("database_connections")


def open_external_connection(url: str, timeout: int = 10):
    """Open a fresh connection to a user-supplied EXTERNAL Postgres database.

    This is the whole point of this module: unlike the app's own database
    (metadata/session storage only), this connects dynamically to whatever
    database the user provided, per request — never pooled, never persisted
    beyond the lifetime of a single call.
    """
    try:
        psycopg2 = __import__("psycopg2")
        extras = __import__("psycopg2.extras", fromlist=["RealDictCursor"])
        real_dict_cursor = getattr(extras, "RealDictCursor")
        conn = psycopg2.connect(url, cursor_factory=real_dict_cursor, connect_timeout=timeout)
        conn.autocommit = True
        return conn
    except Exception as exc:
        logger.warning("database_connections: external DB connection failed: %s", exc)
        raise HTTPException(status_code=400, detail=f"Could not connect: {exc}")


def _ensure_tables(conn) -> None:
    global _tables_ensured
    if _tables_ensured:
        return
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS database_connections (
                    id            BIGSERIAL PRIMARY KEY,
                    connection_id TEXT        NOT NULL UNIQUE,
                    user_id       TEXT        NOT NULL,
                    workspace_id  TEXT,
                    label         TEXT        NOT NULL,
                    url           TEXT        NOT NULL,
                    status        TEXT        NOT NULL DEFAULT 'active'
                                              CHECK (status IN ('active', 'inactive')),
                    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
                );

                CREATE INDEX IF NOT EXISTS idx_database_connections_user_created
                    ON database_connections (user_id, created_at DESC);

                CREATE OR REPLACE FUNCTION _set_database_connections_updated_at()
                RETURNS TRIGGER LANGUAGE plpgsql AS $$
                BEGIN
                    NEW.updated_at = now();
                    RETURN NEW;
                END;
                $$;

                DROP TRIGGER IF EXISTS trg_database_connections_updated_at ON database_connections;
                CREATE TRIGGER trg_database_connections_updated_at
                    BEFORE UPDATE ON database_connections
                    FOR EACH ROW EXECUTE FUNCTION _set_database_connections_updated_at();
                """
            )
        _tables_ensured = True
    except Exception as exc:
        logger.error("database_connections: ensure table failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to initialize database-connections schema")


def list_tables_with_columns(ext_conn, max_tables: int = MAX_TABLES_PER_CONNECTION) -> List[Dict[str, Any]]:
    """Schema-agnostic table + column + approximate row-count listing.

    No hardcoded table whitelist: this browses whatever schema the user's
    external database actually has (mirrors PostgreSQLAdapter's "all tables"
    mode from the research notebook).
    """
    with ext_conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.relname AS table_name, GREATEST(c.reltuples, 0)::bigint AS approx_rows
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE c.relkind = 'r' AND n.nspname NOT IN ('pg_catalog', 'information_schema')
            ORDER BY c.relname
            LIMIT %s
            """,
            (max_tables,),
        )
        table_rows = cur.fetchall() or []
        table_names = [r["table_name"] for r in table_rows]
        row_counts = {r["table_name"]: r["approx_rows"] for r in table_rows}

        if not table_names:
            return []

        cur.execute(
            """
            SELECT table_name, column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
              AND table_name = ANY(%s)
            ORDER BY table_name, ordinal_position
            """,
            (table_names,),
        )
        column_rows = cur.fetchall() or []

        cur.execute(
            """
            SELECT tc.table_name, kcu.column_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
                ON tc.constraint_name = kcu.constraint_name
               AND tc.table_schema = kcu.table_schema
            WHERE tc.constraint_type = 'PRIMARY KEY'
              AND tc.table_schema NOT IN ('pg_catalog', 'information_schema')
              AND tc.table_name = ANY(%s)
            """,
            (table_names,),
        )
        pk_pairs = {(r["table_name"], r["column_name"]) for r in (cur.fetchall() or [])}

    columns_by_table: Dict[str, List[Dict[str, Any]]] = {name: [] for name in table_names}
    for row in column_rows:
        columns_by_table.setdefault(row["table_name"], []).append({
            "name": row["column_name"],
            "type": row["data_type"],
            "nullable": row["is_nullable"] == "YES",
            "primary_key": (row["table_name"], row["column_name"]) in pk_pairs,
        })

    return [
        {
            "name": name,
            "row_count": row_counts.get(name),
            "columns": columns_by_table.get(name, []),
        }
        for name in table_names
    ]


def _fetch_connections(cur, user_id: str) -> List[Dict[str, Any]]:
    if _enforce_scope():
        cur.execute(
            """
            SELECT connection_id, workspace_id, label, url, status, created_at
            FROM database_connections
            WHERE user_id = %s
            ORDER BY created_at DESC
            """,
            (user_id,),
        )
    else:
        cur.execute(
            """
            SELECT connection_id, workspace_id, label, url, status, created_at
            FROM database_connections
            ORDER BY created_at DESC
            """,
        )
    return cur.fetchall() or []


def _as_source(row: Dict[str, Any], tables: Optional[List[Dict[str, Any]]] = None) -> DatabaseConnectionSource:
    tables = tables or []
    return DatabaseConnectionSource(
        connection_id=row["connection_id"],
        workspace_id=row.get("workspace_id"),
        label=row["label"],
        url=_redact_url(row["url"]),
        status=row["status"],
        table_count=len(tables),
        created_at=app_db.ts(row.get("created_at")),
        tables=[
            DbTableInfo(
                name=t["name"],
                row_count=t.get("row_count"),
                columns=[DbColumnInfo(**c) for c in t.get("columns", [])],
            )
            for t in tables
        ],
    )


async def resolve_active_database_connections(
    connection_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Used by the RAG pipeline: active connections' raw (unredacted) URLs."""
    conn = _get_app_conn()
    if not conn:
        raise HTTPException(status_code=503, detail="Database unavailable")

    _ensure_tables(conn)
    selected_ids = set(connection_ids or [])
    try:
        with conn.cursor() as cur:
            rows = _fetch_connections(cur, _db_connections_user_id())
    finally:
        conn.close()

    return [
        {
            "connection_id": r["connection_id"],
            "label": r["label"],
            "url": r["url"],
        }
        for r in rows
        if r.get("status") == "active" and (not selected_ids or r.get("connection_id") in selected_ids)
    ]


@router.get("/database-connections", response_model=DatabaseConnectionsResponse)
async def list_database_connections(_: UserRecord = Depends(require_role("admin"))):
    conn = _get_app_conn()
    if not conn:
        raise HTTPException(status_code=503, detail="Database unavailable")

    _ensure_tables(conn)
    try:
        with conn.cursor() as cur:
            rows = _fetch_connections(cur, _db_connections_user_id())
        connections = [_as_source(r) for r in rows]
        return DatabaseConnectionsResponse(connections=connections, count=len(connections))
    except Exception as exc:
        logger.error("database_connections: list failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to list database connections")
    finally:
        conn.close()


@router.post("/database-connections", response_model=DatabaseConnectionSource, status_code=status.HTTP_201_CREATED)
async def create_database_connection(
    body: CreateDatabaseConnectionRequest,
    _: UserRecord = Depends(require_role("admin")),
):
    _validate_postgres_url(body.url)

    # Test the connection and browse its schema before persisting anything.
    ext_conn = open_external_connection(body.url)
    try:
        tables = list_tables_with_columns(ext_conn)
    finally:
        ext_conn.close()

    app_conn = _get_app_conn()
    if not app_conn:
        raise HTTPException(status_code=503, detail="Database unavailable")

    _ensure_tables(app_conn)
    label = (body.label or "").strip() or _derive_label(body.url)
    connection_id = str(uuid.uuid4())

    try:
        with app_conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO database_connections (connection_id, user_id, workspace_id, label, url, status)
                VALUES (%s, %s, %s, %s, %s, 'active')
                RETURNING connection_id, workspace_id, label, url, status, created_at
                """,
                (connection_id, _db_connections_user_id(), None, label, body.url),
            )
            row = cur.fetchone()
        return _as_source(row, tables)
    except Exception as exc:
        logger.error("database_connections: create failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to create database connection")
    finally:
        app_conn.close()


@router.get("/database-connections/{connection_id}/tables", response_model=DatabaseConnectionSource)
async def refresh_database_connection_tables(
    connection_id: str,
    _: UserRecord = Depends(require_role("admin")),
):
    app_conn = _get_app_conn()
    if not app_conn:
        raise HTTPException(status_code=503, detail="Database unavailable")

    _ensure_tables(app_conn)
    try:
        with app_conn.cursor() as cur:
            if _enforce_scope():
                cur.execute(
                    """
                    SELECT connection_id, workspace_id, label, url, status, created_at
                    FROM database_connections WHERE connection_id = %s AND user_id = %s
                    """,
                    (connection_id, _db_connections_user_id()),
                )
            else:
                cur.execute(
                    """
                    SELECT connection_id, workspace_id, label, url, status, created_at
                    FROM database_connections WHERE connection_id = %s
                    """,
                    (connection_id,),
                )
            row = cur.fetchone()
    finally:
        app_conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Database connection not found")

    ext_conn = open_external_connection(row["url"])
    try:
        tables = list_tables_with_columns(ext_conn)
    finally:
        ext_conn.close()

    return _as_source(row, tables)


@router.post("/database-connections/activate")
async def set_database_connection_active(
    body: SetDatabaseConnectionActiveRequest,
    _: UserRecord = Depends(require_role("admin")),
):
    conn = _get_app_conn()
    if not conn:
        raise HTTPException(status_code=503, detail="Database unavailable")

    _ensure_tables(conn)
    try:
        with conn.cursor() as cur:
            if _enforce_scope():
                cur.execute(
                    """
                    UPDATE database_connections SET status = %s
                    WHERE connection_id = %s AND user_id = %s
                    RETURNING connection_id
                    """,
                    ("active" if body.active else "inactive", body.connection_id, _db_connections_user_id()),
                )
            else:
                cur.execute(
                    """
                    UPDATE database_connections SET status = %s
                    WHERE connection_id = %s
                    RETURNING connection_id
                    """,
                    ("active" if body.active else "inactive", body.connection_id),
                )
            updated = cur.fetchone()

        if not updated:
            raise HTTPException(status_code=404, detail="Database connection not found")
        return {"status": "success", "connection_id": body.connection_id, "active": body.active}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("database_connections: activate failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to update active status")
    finally:
        conn.close()


@router.delete("/database-connections/{connection_id}")
async def delete_database_connection(
    connection_id: str,
    _: UserRecord = Depends(require_role("admin")),
):
    conn = _get_app_conn()
    if not conn:
        raise HTTPException(status_code=503, detail="Database unavailable")

    _ensure_tables(conn)
    try:
        with conn.cursor() as cur:
            if _enforce_scope():
                cur.execute(
                    "DELETE FROM database_connections WHERE connection_id = %s AND user_id = %s RETURNING connection_id",
                    (connection_id, _db_connections_user_id()),
                )
            else:
                cur.execute(
                    "DELETE FROM database_connections WHERE connection_id = %s RETURNING connection_id",
                    (connection_id,),
                )
            deleted = cur.fetchone()

        if not deleted:
            raise HTTPException(status_code=404, detail="Database connection not found")
        return {"status": "success", "message": "Database connection deleted", "connection_id": connection_id}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("database_connections: delete failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to delete database connection")
    finally:
        conn.close()
