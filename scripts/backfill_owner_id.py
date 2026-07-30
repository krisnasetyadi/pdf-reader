"""One-time migration: assign owner_id = first admin's user_id to any
pdf_collections / chat_sessions rows created before ownership tracking
existed (owner_id IS NULL).

Run manually once after deploying the owner_id columns:
    python scripts/backfill_owner_id.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg2
from psycopg2.extras import RealDictCursor


def _database_url() -> str:
    url = os.getenv("DATABASE_URL")
    if not url:
        raise SystemExit("DATABASE_URL is not set")
    if "sslmode=" not in url:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}sslmode=require"
    return url


def main() -> None:
    conn = psycopg2.connect(_database_url(), cursor_factory=RealDictCursor, connect_timeout=10)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT user_id, email FROM users WHERE role = 'admin' ORDER BY created_at ASC LIMIT 1"
            )
            admin = cur.fetchone()
            if not admin:
                raise SystemExit("No admin user found — register one first, then rerun this script.")

            admin_id = admin["user_id"]
            print(f"Backfilling owner_id = {admin_id} ({admin['email']})")

            cur.execute(
                "UPDATE pdf_collections SET owner_id = %s WHERE owner_id IS NULL",
                (admin_id,),
            )
            print(f"pdf_collections: {cur.rowcount} row(s) updated")

            cur.execute(
                "UPDATE chat_sessions SET owner_id = %s WHERE owner_id IS NULL",
                (admin_id,),
            )
            print(f"chat_sessions: {cur.rowcount} row(s) updated")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
