from __future__ import annotations

from fastapi import APIRouter, HTTPException, status, Depends
from urllib.parse import parse_qs, unquote, urlparse
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timezone
import logging
import os
import uuid
import re
from html import unescape
import httpx
from config import config
from ssrf_guard import assert_public_url_safe
from router.auth import get_current_user, UserRecord

from models import (
    CreatePublicLinkRequest,
    SetPublicLinkActiveRequest,
    PublicLinkItem,
    PublicLinkSource,
    PublicLinksResponse,
)

router = APIRouter()
logger = logging.getLogger(__name__)

_tables_ensured = False
GOOGLE_DRIVE_HOSTS = {"drive.google.com", "docs.google.com"}


def _database_url() -> str | None:
    url = os.getenv("DATABASE_URL") or getattr(config, "database_url", None)
    if not url:
        return None
    if "sslmode=" not in url:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}sslmode=require"
    return url


def _ts(value: Any) -> str:
    if value is None:
        return datetime.now(timezone.utc).isoformat()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _derive_title(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.netloc or "Public Link"
    path = parsed.path.strip("/")
    if path:
        return f"{host}/{path.split('/')[-1]}"
    return host


def _validate_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise HTTPException(status_code=400, detail="Invalid public URL")


def _name_from_url(url: str) -> str:
    parsed = urlparse(url)
    tail = parsed.path.strip("/").split("/")[-1] if parsed.path else ""
    return tail or parsed.netloc or "item"


def _normalize_host(raw_host: str) -> str:
    host = raw_host.lower()
    return host[4:] if host.startswith("www.") else host


def _extract_google_drive_file_id(raw_url: str) -> str | None:
    parsed = urlparse(raw_url)
    if _normalize_host(parsed.netloc) not in GOOGLE_DRIVE_HOSTS:
        return None

    path_match = re.search(r"/file/d/([a-zA-Z0-9_-]+)", parsed.path)
    if path_match:
        return path_match.group(1)

    query = parse_qs(parsed.query)
    return query.get("id", [None])[0]


def _extract_google_drive_folder_id(raw_url: str) -> str | None:
    parsed = urlparse(raw_url)
    if _normalize_host(parsed.netloc) not in GOOGLE_DRIVE_HOSTS:
        return None

    path_match = re.search(r"/drive/folders/([a-zA-Z0-9_-]+)", parsed.path)
    if path_match:
        return path_match.group(1)

    query = parse_qs(parsed.query)
    return query.get("id", [None])[0]


def _to_drive_absolute_url(href: str) -> str:
    if href.startswith("http://") or href.startswith("https://"):
        return href
    if href.startswith("/"):
        return f"https://drive.google.com{href}"
    return f"https://drive.google.com/{href}"


def _parse_drive_folder_items(html_text: str) -> tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    anchor_pattern = re.compile(r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', re.IGNORECASE | re.DOTALL)
    files: List[Dict[str, str]] = []
    folders: List[Dict[str, str]] = []
    seen_files = set()
    seen_folders = set()

    for href, raw_label in anchor_pattern.findall(html_text):
        full_url = _to_drive_absolute_url(unescape(href))
        label = re.sub(r"<[^>]+>", "", unescape(raw_label)).strip()

        file_id = _extract_google_drive_file_id(full_url)
        if file_id:
            if file_id in seen_files:
                continue
            seen_files.add(file_id)
            files.append({
                "name": label or f"Google Drive file {len(files) + 1}",
                "url": full_url,
                "item_type": "file",
            })
            continue

        folder_id = _extract_google_drive_folder_id(full_url)
        if folder_id:
            if folder_id in seen_folders:
                continue
            seen_folders.add(folder_id)
            folders.append({
                "name": label or f"Google Drive folder {len(folders) + 1}",
                "url": full_url,
                "item_type": "folder",
            })

    return files, folders


def _is_probably_pdf(item_url: str, item_name: str) -> bool:
    if item_name.lower().endswith(".pdf"):
        return True
    parsed = urlparse(item_url)
    return "/file/d/" in parsed.path


async def _fetch_drive_folder_page(folder_id: str) -> tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    embed_url = f"https://drive.google.com/embeddedfolderview?id={folder_id}#list"
    timeout = httpx.Timeout(30.0, connect=10.0)
    async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
        response = await client.get(
            embed_url,
            headers={"User-Agent": "DocuLens/1.0 (+public-links)"},
        )
        response.raise_for_status()
    return _parse_drive_folder_items(response.text)


async def _extract_items_from_source(url: str) -> List[Dict[str, str]]:
    folder_id = _extract_google_drive_folder_id(url)
    if folder_id:
        try:
            files, folders = await _fetch_drive_folder_page(folder_id)
            pdf_files = [entry for entry in files if _is_probably_pdf(entry["url"], entry["name"])]
            return pdf_files + folders
        except Exception as exc:
            logger.warning("public_links: failed extracting folder items from %s: %s", url, exc)
            return []

    file_id = _extract_google_drive_file_id(url)
    if file_id:
        return [{"name": _name_from_url(url), "url": url, "item_type": "file"}]

    return []


def _replace_items(cur, link_id: str, items: List[Dict[str, str]]) -> None:
    cur.execute("DELETE FROM public_link_items WHERE link_id = %s", (link_id,))
    for item in items:
        cur.execute(
            """
            INSERT INTO public_link_items (item_id, link_id, name, url, item_type)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (
                str(uuid.uuid4()),
                link_id,
                item["name"],
                item["url"],
                item.get("item_type", "file"),
            ),
        )


def _fetch_link_detail(cur, link_id: str, user_id: str, is_admin: bool = False) -> Dict[str, Any] | None:
    if not is_admin:
        cur.execute(
            """
            SELECT link_id, workspace_id, title, url, status, created_at
            FROM public_links
            WHERE link_id = %s AND user_id = %s
            """,
            (link_id, user_id),
        )
    else:
        cur.execute(
            """
            SELECT link_id, workspace_id, title, url, status, created_at
            FROM public_links
            WHERE link_id = %s
            """,
            (link_id,),
        )
    link = cur.fetchone()
    if not link:
        return None

    cur.execute(
        """
        SELECT item_id, name, url, item_type
        FROM public_link_items
        WHERE link_id = %s
        ORDER BY created_at ASC
        """,
        (link_id,),
    )
    items = cur.fetchall() or []

    return {
        "link_id": link["link_id"],
        "workspace_id": link.get("workspace_id"),
        "title": link["title"],
        "url": link["url"],
        "status": link["status"],
        "item_count": len(items),
        "created_at": _ts(link.get("created_at")),
        "items": [
            {
                "id": row["item_id"],
                "name": row["name"],
                "url": row["url"],
                "item_type": row["item_type"],
            }
            for row in items
        ],
    }


def _get_conn():
    database_url = _database_url()
    if not database_url:
        return None
    try:
        psycopg2 = __import__("psycopg2")
        extras = __import__("psycopg2.extras", fromlist=["RealDictCursor"])
        real_dict_cursor = getattr(extras, "RealDictCursor")

        url = database_url
        if "sslmode=" not in url:
            sep = "&" if "?" in url else "?"
            url = url + sep + "sslmode=require"

        conn = psycopg2.connect(url, cursor_factory=real_dict_cursor, connect_timeout=10)
        conn.autocommit = True
        return conn
    except Exception as exc:
        logger.warning("public_links: DB connection failed: %s", exc)
        return None


def _ensure_tables(conn) -> None:
    global _tables_ensured
    if _tables_ensured:
        return

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS public_links (
                    id            BIGSERIAL PRIMARY KEY,
                    link_id       TEXT        NOT NULL UNIQUE,
                    user_id       TEXT        NOT NULL,
                    workspace_id  TEXT,
                    title         TEXT        NOT NULL,
                    url           TEXT        NOT NULL,
                    status        TEXT        NOT NULL DEFAULT 'inactive'
                                              CHECK (status IN ('active', 'inactive')),
                    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
                );

                CREATE TABLE IF NOT EXISTS public_link_items (
                    id            BIGSERIAL PRIMARY KEY,
                    item_id       TEXT        NOT NULL UNIQUE,
                    link_id       TEXT        NOT NULL REFERENCES public_links(link_id) ON DELETE CASCADE,
                    name          TEXT        NOT NULL,
                    url           TEXT        NOT NULL,
                    item_type     TEXT        NOT NULL DEFAULT 'file'
                                              CHECK (item_type IN ('file', 'folder')),
                    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
                );

                CREATE INDEX IF NOT EXISTS idx_public_links_user_created
                    ON public_links (user_id, created_at DESC);

                CREATE INDEX IF NOT EXISTS idx_public_link_items_link
                    ON public_link_items (link_id);

                CREATE OR REPLACE FUNCTION _set_public_links_updated_at()
                RETURNS TRIGGER LANGUAGE plpgsql AS $$
                BEGIN
                    NEW.updated_at = now();
                    RETURN NEW;
                END;
                $$;

                DROP TRIGGER IF EXISTS trg_public_links_updated_at ON public_links;
                CREATE TRIGGER trg_public_links_updated_at
                    BEFORE UPDATE ON public_links
                    FOR EACH ROW EXECUTE FUNCTION _set_public_links_updated_at();
                """
            )
        _tables_ensured = True
    except Exception as exc:
        logger.error("public_links: ensure table failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to initialize public links schema")


def _fetch_links(cur, user_id: str, is_admin: bool = False) -> List[Dict[str, Any]]:
    if not is_admin:
        cur.execute(
            """
            SELECT link_id, workspace_id, title, url, status, created_at
            FROM public_links
            WHERE user_id = %s
            ORDER BY created_at DESC
            """,
            (user_id,),
        )
    else:
        cur.execute(
            """
            SELECT link_id, workspace_id, title, url, status, created_at
            FROM public_links
            ORDER BY created_at DESC
            """,
        )
    links = cur.fetchall() or []

    result: List[Dict[str, Any]] = []
    for link in links:
        cur.execute(
            """
            SELECT item_id, name, url, item_type
            FROM public_link_items
            WHERE link_id = %s
            ORDER BY created_at ASC
            """,
            (link["link_id"],),
        )
        items = cur.fetchall() or []
        result.append(
            {
                "link_id": link["link_id"],
                "workspace_id": link.get("workspace_id"),
                "title": link["title"],
                "url": link["url"],
                "status": link["status"],
                "item_count": len(items),
                "created_at": _ts(link.get("created_at")),
                "items": [
                    {
                        "id": row["item_id"],
                        "name": row["name"],
                        "url": row["url"],
                        "item_type": row["item_type"],
                    }
                    for row in items
                ],
            }
        )
    return result


def _as_public_link_source(payload: Dict[str, Any]) -> PublicLinkSource:
    return PublicLinkSource(
        link_id=payload["link_id"],
        workspace_id=payload.get("workspace_id"),
        title=payload["title"],
        url=payload["url"],
        status=payload["status"],
        item_count=payload["item_count"],
        created_at=payload["created_at"],
        items=[
            PublicLinkItem(
                id=item["id"],
                name=item["name"],
                url=item["url"],
                item_type=item["item_type"],
            )
            for item in payload.get("items", [])
        ],
    )


async def _resolve_runtime_items(
    url: str,
    *,
    max_depth: int = 5,
    seen_urls: Optional[Set[str]] = None,
) -> List[Dict[str, str]]:
    seen_urls = seen_urls or set()
    if url in seen_urls:
        return []
    seen_urls.add(url)

    folder_id = _extract_google_drive_folder_id(url)
    if folder_id:
        if max_depth <= 0:
            return []

        try:
            files, folders = await _fetch_drive_folder_page(folder_id)
        except Exception as exc:
            logger.warning("public_links: runtime folder expansion failed for %s: %s", url, exc)
            return []

        resolved: List[Dict[str, str]] = []
        for entry in files:
            name = entry.get("name", "")
            item_url = entry.get("url", "")
            lowered = name.lower()
            if _is_probably_pdf(item_url, name) or lowered.endswith((".txt", ".md", ".log")):
                resolved.append(
                    {
                        "name": name or _name_from_url(item_url),
                        "url": item_url,
                        "item_type": "file",
                    }
                )

        for folder in folders:
            resolved.extend(
                await _resolve_runtime_items(
                    folder["url"],
                    max_depth=max_depth - 1,
                    seen_urls=seen_urls,
                )
            )

        return resolved

    return [
        {
            "name": _name_from_url(url),
            "url": url,
            "item_type": "file",
        }
    ]


async def resolve_active_public_link_sources(
    link_ids: Optional[List[str]] = None,
    user_id: Optional[str] = None,
    is_admin: bool = False,
) -> List[Dict[str, Any]]:
    conn = _get_conn()
    if not conn:
        raise HTTPException(status_code=503, detail="Database unavailable")

    _ensure_tables(conn)
    selected_ids = set(link_ids or [])

    try:
        with conn.cursor() as cur:
            raw_links = _fetch_links(cur, user_id or "", is_admin)
    finally:
        conn.close()

    active_links = [
        link for link in raw_links
        if link.get("status") == "active" and (not selected_ids or link.get("link_id") in selected_ids)
    ]

    resolved_links: List[Dict[str, Any]] = []
    for link in active_links:
        live_items = await _resolve_runtime_items(link["url"])
        fallback_items = link.get("items", [])
        dedup: Dict[str, Dict[str, str]] = {}

        for item in (live_items or fallback_items):
            item_url = item.get("url")
            if not item_url:
                continue
            dedup[item_url] = {
                "id": item.get("id") or str(uuid.uuid4()),
                "name": item.get("name") or _name_from_url(item_url),
                "url": item_url,
                "item_type": item.get("item_type", "file"),
            }

        resolved_links.append(
            {
                **link,
                "items": list(dedup.values()),
                "item_count": len(dedup),
            }
        )

    return resolved_links


@router.get("/public-links", response_model=PublicLinksResponse)
async def list_public_links(user: UserRecord = Depends(get_current_user)):
    conn = _get_conn()
    if not conn:
        raise HTTPException(status_code=503, detail="Database unavailable")

    _ensure_tables(conn)
    try:
        with conn.cursor() as cur:
            raw_links = _fetch_links(cur, user.user_id, user.role == "admin")
        links = [_as_public_link_source(entry) for entry in raw_links]
        return PublicLinksResponse(links=links, count=len(links))
    except Exception as exc:
        logger.error("public_links: list failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to list public links")
    finally:
        conn.close()


@router.post("/public-links", response_model=PublicLinkSource, status_code=status.HTTP_201_CREATED)
async def create_public_link(
    body: CreatePublicLinkRequest,
    user: UserRecord = Depends(get_current_user),
):
    _validate_url(body.url)
    assert_public_url_safe(body.url)

    conn = _get_conn()
    if not conn:
        raise HTTPException(status_code=503, detail="Database unavailable")

    _ensure_tables(conn)

    title = (body.title or "").strip() or _derive_title(body.url)
    link_id = str(uuid.uuid4())
    item_urls = body.item_urls or []

    extracted_items: List[Dict[str, str]] = []
    if item_urls:
        extracted_items = [
            {
                "name": _name_from_url(item_url),
                "url": item_url,
                "item_type": "file",
            }
            for item_url in item_urls
        ]
    else:
        extracted_items = await _extract_items_from_source(body.url)

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO public_links (link_id, user_id, workspace_id, title, url, status)
                VALUES (%s, %s, %s, %s, %s, 'active')
                RETURNING link_id, workspace_id, title, url, status, created_at
                """,
                (link_id, user.user_id, None, title, body.url),
            )
            row = cur.fetchone()

            dedup: Dict[str, Dict[str, str]] = {}
            for item in extracted_items:
                _validate_url(item["url"])
                assert_public_url_safe(item["url"])
                dedup[item["url"]] = item
            _replace_items(cur, link_id, list(dedup.values()))

            cur.execute(
                """
                SELECT item_id, name, url, item_type
                FROM public_link_items
                WHERE link_id = %s
                ORDER BY created_at ASC
                """,
                (link_id,),
            )
            items = cur.fetchall() or []

        return PublicLinkSource(
            link_id=row["link_id"],
            workspace_id=row.get("workspace_id"),
            title=row["title"],
            url=row["url"],
            status=row["status"],
            item_count=len(items),
            created_at=_ts(row.get("created_at")),
            items=[
                PublicLinkItem(
                    id=item["item_id"],
                    name=item["name"],
                    url=item["url"],
                    item_type=item["item_type"],
                )
                for item in items
            ],
        )
    except Exception as exc:
        logger.error("public_links: create failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to create public link")
    finally:
        conn.close()


@router.post("/public-link/activate")
async def set_public_link_active(
    body: SetPublicLinkActiveRequest,
    user: UserRecord = Depends(get_current_user),
):
    conn = _get_conn()
    if not conn:
        raise HTTPException(status_code=503, detail="Database unavailable")

    _ensure_tables(conn)

    try:
        with conn.cursor() as cur:
            if user.role != "admin":
                cur.execute(
                    """
                    UPDATE public_links
                    SET status = %s
                    WHERE link_id = %s AND user_id = %s
                    RETURNING link_id
                    """,
                    ("active" if body.active else "inactive", body.link_id, user.user_id),
                )
            else:
                cur.execute(
                    """
                    UPDATE public_links
                    SET status = %s
                    WHERE link_id = %s
                    RETURNING link_id
                    """,
                    ("active" if body.active else "inactive", body.link_id),
                )
            updated = cur.fetchone()

        if not updated:
            raise HTTPException(status_code=404, detail="Public link not found")

        return {"status": "success", "link_id": body.link_id, "active": body.active}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("public_links: activate failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to update active status")
    finally:
        conn.close()


@router.delete("/public-link/{link_id}")
async def delete_public_link(link_id: str, user: UserRecord = Depends(get_current_user)):
    conn = _get_conn()
    if not conn:
        raise HTTPException(status_code=503, detail="Database unavailable")

    _ensure_tables(conn)

    try:
        with conn.cursor() as cur:
            if user.role != "admin":
                cur.execute(
                    """
                    DELETE FROM public_links
                    WHERE link_id = %s AND user_id = %s
                    RETURNING link_id
                    """,
                    (link_id, user.user_id),
                )
            else:
                cur.execute(
                    """
                    DELETE FROM public_links
                    WHERE link_id = %s
                    RETURNING link_id
                    """,
                    (link_id,),
                )
            deleted = cur.fetchone()

        if not deleted:
            raise HTTPException(status_code=404, detail="Public link not found")

        return {"status": "success", "message": "Public link deleted", "link_id": link_id}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("public_links: delete failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to delete public link")
    finally:
        conn.close()
