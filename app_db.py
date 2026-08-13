# app_db.py
"""
Shared helper for connecting to THIS APP's own database — the metadata
store for routers that keep a small schema of their own (database
connections, Telegram connections, ...), never a user-connected external
source. Previously duplicated near-verbatim across router/database_connections.py
and router/telegram.py; kept in one place so a fix (e.g. the sslmode
heuristic, connect timeout) can't silently apply to only one of them.
"""

import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

from config import config

logger = logging.getLogger(__name__)


def app_database_url() -> Optional[str]:
    url = os.getenv("DATABASE_URL") or getattr(config, "database_url", None)
    if not url:
        return None
    if "sslmode=" not in url:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}sslmode=require"
    return url


def get_app_conn(source: str = "app_db"):
    """Open a connection to the app's own database, or None if unconfigured/unreachable."""
    database_url = app_database_url()
    if not database_url:
        return None
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
        conn = psycopg2.connect(database_url, cursor_factory=RealDictCursor, connect_timeout=10)
        conn.autocommit = True
        return conn
    except Exception as exc:
        logger.warning("%s: app DB connection failed: %s", source, exc)
        return None


def ts(value: Any) -> str:
    if value is None:
        return datetime.now(timezone.utc).isoformat()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)
