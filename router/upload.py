# router/upload.py
from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Depends
from typing import List, Optional, Literal
import asyncio
import os
import shutil
import uuid
import re
from router.auth import get_current_user, UserRecord
from html import unescape
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import httpx

from utils import process_pdfs
from ssrf_guard import assert_public_url_safe
from models import (
    DriveFolderItem,
    DriveFolderItemsResponse,
    DriveFolderRequest,
    UploadFromUrlRequest,
    UploadFromUrlsRequest,
    UploadResponse,
)
from config import config
import logging
import storage as supabase_storage

router = APIRouter()
logger = logging.getLogger(__name__)

GOOGLE_DRIVE_HOSTS = {"drive.google.com", "docs.google.com"}


def _extract_google_drive_file_id(raw_url: str) -> Optional[str]:
    parsed = urlparse(raw_url)
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]

    if host not in GOOGLE_DRIVE_HOSTS:
        return None

    path_match = re.search(r"/file/d/([a-zA-Z0-9_-]+)", parsed.path)
    if path_match:
        return path_match.group(1)

    query = parse_qs(parsed.query)
    file_id = query.get("id", [None])[0]
    return file_id


def _normalize_remote_pdf_url(raw_url: str) -> str:
    parsed = urlparse(raw_url)
    if parsed.scheme not in {"http", "https"}:
        raise HTTPException(status_code=400, detail="Please provide a valid http(s) URL")

    file_id = _extract_google_drive_file_id(raw_url)
    if file_id:
        return f"https://drive.google.com/uc?export=download&id={file_id}"

    return raw_url


def _extract_google_drive_folder_id(raw_url: str) -> Optional[str]:
    parsed = urlparse(raw_url)
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]

    if host not in GOOGLE_DRIVE_HOSTS:
        return None

    path_match = re.search(r"/drive/folders/([a-zA-Z0-9_-]+)", parsed.path)
    if path_match:
        return path_match.group(1)

    query = parse_qs(parsed.query)
    folder_id = query.get("id", [None])[0]
    return folder_id


def _to_drive_absolute_url(href: str) -> str:
    if href.startswith("http://") or href.startswith("https://"):
        return href
    if href.startswith("/"):
        return f"https://drive.google.com{href}"
    return f"https://drive.google.com/{href}"


def _parse_drive_folder_items(html_text: str) -> tuple[List[DriveFolderItem], List[DriveFolderItem]]:
    anchor_pattern = re.compile(r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', re.IGNORECASE | re.DOTALL)
    seen = set()
    files: List[DriveFolderItem] = []
    folders: List[DriveFolderItem] = []

    for href, raw_label in anchor_pattern.findall(html_text):
        full_url = _to_drive_absolute_url(unescape(href))
        label = re.sub(r"<[^>]+>", "", unescape(raw_label)).strip()

        file_id = _extract_google_drive_file_id(full_url)
        if file_id:
            key = ("file", file_id)
            if key in seen:
                continue
            seen.add(key)
            files.append(
                DriveFolderItem(
                    id=file_id,
                    name=label or f"Google Drive file {len(files) + 1}",
                    url=full_url,
                    item_type="file",
                )
            )
            continue

        folder_id = _extract_google_drive_folder_id(full_url)
        if folder_id:
            key = ("folder", folder_id)
            if key in seen:
                continue
            seen.add(key)
            folders.append(
                DriveFolderItem(
                    id=folder_id,
                    name=label or f"Google Drive folder {len(folders) + 1}",
                    url=full_url,
                    item_type="folder",
                )
            )

    return files, folders


def _is_probably_pdf(item: DriveFolderItem) -> bool:
    lowered_name = item.name.lower()
    if lowered_name.endswith(".pdf"):
        return True
    parsed = urlparse(item.url)
    return "/file/d/" in parsed.path


async def _fetch_drive_folder_page(folder_id: str) -> tuple[List[DriveFolderItem], List[DriveFolderItem]]:
    embed_url = f"https://drive.google.com/embeddedfolderview?id={folder_id}#list"
    timeout = httpx.Timeout(30.0, connect=10.0)
    async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
        response = await client.get(
            embed_url,
            headers={"User-Agent": "DocuLens/1.0 (+drive-folder-browser)"},
        )
        response.raise_for_status()
    return _parse_drive_folder_items(response.text)


def _extract_filename_from_headers(headers: httpx.Headers) -> Optional[str]:
    content_disposition = headers.get("content-disposition", "")
    if not content_disposition:
        return None

    filename_star = re.search(r"filename\*=UTF-8''([^;]+)", content_disposition, re.IGNORECASE)
    if filename_star:
        return unquote(filename_star.group(1).strip())

    filename = re.search(r'filename="?([^";]+)"?', content_disposition, re.IGNORECASE)
    if filename:
        return filename.group(1).strip()

    return None


def _safe_pdf_filename(candidate: Optional[str], fallback_stem: str) -> str:
    raw_name = (candidate or "").strip()
    file_name = Path(raw_name).name if raw_name else ""
    file_name = re.sub(r"[^A-Za-z0-9._ -]", "_", file_name).strip(" ._")

    if not file_name:
        file_name = f"{fallback_stem}.pdf"
    elif not file_name.lower().endswith(".pdf"):
        file_name = f"{file_name}.pdf"

    return file_name


def _normalize_title(candidate: Optional[str]) -> Optional[str]:
    if not candidate:
        return None
    title = re.sub(r"\s+", " ", candidate).strip()
    return title[:150] if title else None


async def _download_remote_pdf(source_url: str, destination_path: str) -> tuple[str, str]:
    assert_public_url_safe(source_url)
    timeout = httpx.Timeout(60.0, connect=20.0)
    async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
        async with client.stream(
            "GET",
            source_url,
            headers={"User-Agent": "DocuLens/1.0 (+remote-pdf-import)"},
        ) as response:
            response.raise_for_status()

            content_type = (response.headers.get("content-type") or "").lower()
            resolved_name = _extract_filename_from_headers(response.headers)
            if not resolved_name:
                resolved_name = Path(urlparse(str(response.url)).path).name

            if "pdf" not in content_type and not (resolved_name or "").lower().endswith(".pdf"):
                raise HTTPException(
                    status_code=400,
                    detail="The provided link did not return a PDF file. Make sure the file is public and downloadable.",
                )

            with open(destination_path, "wb") as output:
                async for chunk in response.aiter_bytes():
                    output.write(chunk)

            return resolved_name or "", content_type


async def _download_pdf_to_collection(
    source_url: str,
    collection_path: str,
    fallback_stem: str,
) -> tuple[str, str]:
    normalized_url = _normalize_remote_pdf_url(source_url)
    temp_path = os.path.join(collection_path, f"{fallback_stem}.pdf")
    resolved_name, _ = await _download_remote_pdf(normalized_url, temp_path)
    file_name = _safe_pdf_filename(resolved_name, fallback_stem)
    final_path = os.path.join(collection_path, file_name)

    if os.path.abspath(final_path) != os.path.abspath(temp_path):
        os.replace(temp_path, final_path)
    else:
        final_path = temp_path

    return final_path, file_name


def _register_uploaded_collection(
    collection_id: str,
    saved_files: List[str],
    file_names: List[str],
    chunk_count: int,
    title: Optional[str] = None,
    persist_mode: Literal["auto", "local", "database"] = "auto",
    owner_id: Optional[str] = None,
):
    if persist_mode == "local":
        logger.info("Persist mode=local — skipping Supabase upload and DB registration")
        return

    storage_paths: List[str] = []
    if persist_mode == "database" and not supabase_storage.is_enabled():
        raise HTTPException(
            status_code=503,
            detail="Supabase Storage not configured — cannot persist with persist_mode=database",
        )

    if persist_mode in ("auto", "database") and supabase_storage.is_enabled():
        logger.info("Uploading collection %s to Supabase Storage…", collection_id)

        for file_path, fname in zip(saved_files, file_names):
            path = supabase_storage.upload_pdf(collection_id, file_path, fname)
            if path:
                storage_paths.append(path)

        index_dir = os.path.join(config.index_folder, collection_id)
        supabase_storage.upload_index(collection_id, index_dir)
        logger.info("Collection %s S3 upload done (%d file(s))", collection_id, len(storage_paths))
    elif persist_mode == "auto":
        logger.info("S3 not configured — skipping file upload to Supabase Storage")

    if supabase_storage.has_database():
        index_dir = os.path.join(config.index_folder, collection_id)
        supabase_storage.register_collection(
            collection_id=collection_id,
            file_names=file_names,
            chunk_count=chunk_count,
            title=title,
            storage_paths=storage_paths,
            owner_id=owner_id,
        )
        logger.info("Collection %s registered in Supabase DB", collection_id)
    elif persist_mode == "database":
        raise HTTPException(
            status_code=503,
            detail="DATABASE_URL not set — cannot persist with persist_mode=database",
        )
    else:
        logger.info("DATABASE_URL not set — skipping Supabase DB registration")


def _cleanup_local_artifacts(collection_id: str, collection_path: str):
    """Remove local upload + index artifacts after successful remote persistence."""
    try:
        shutil.rmtree(collection_path, ignore_errors=True)
        index_dir = os.path.join(config.index_folder, collection_id)
        shutil.rmtree(index_dir, ignore_errors=True)
        logger.info("Local artifacts cleaned for collection %s", collection_id)
    except Exception as exc:
        logger.warning("Failed cleaning local artifacts for %s: %s", collection_id, exc)


@router.post("/drive/folder-items", response_model=DriveFolderItemsResponse)
async def list_drive_folder_items(
    payload: DriveFolderRequest,
    user: UserRecord = Depends(get_current_user),
):
    folder_url = payload.url.strip()
    if not folder_url:
        raise HTTPException(status_code=400, detail="URL is required")

    folder_id = _extract_google_drive_folder_id(folder_url)
    if not folder_id:
        raise HTTPException(status_code=400, detail="Please provide a valid Google Drive folder URL")

    recursive = payload.recursive if payload.recursive is not None else True
    max_depth = payload.max_depth if payload.max_depth is not None else 5
    max_depth = max(0, min(max_depth, 10))

    try:
        queue: List[tuple[str, int]] = [(folder_id, 0)]
        visited_folders = {folder_id}
        discovered_files: List[DriveFolderItem] = []
        discovered_folders: List[DriveFolderItem] = []
        seen_files = set()
        seen_folders = set()

        while queue:
            current_folder_id, depth = queue.pop(0)
            files, folders = await _fetch_drive_folder_page(current_folder_id)

            for file_item in files:
                if not _is_probably_pdf(file_item):
                    continue
                if file_item.id in seen_files:
                    continue
                seen_files.add(file_item.id)
                discovered_files.append(file_item)

            for folder_item in folders:
                if folder_item.id in seen_folders:
                    continue
                seen_folders.add(folder_item.id)
                discovered_folders.append(folder_item)

                if recursive and depth < max_depth and folder_item.id not in visited_folders:
                    visited_folders.add(folder_item.id)
                    queue.append((folder_item.id, depth + 1))

        return DriveFolderItemsResponse(
            folder_id=folder_id,
            files=discovered_files,
            folders=discovered_folders,
            count=len(discovered_files),
        )
    except httpx.HTTPError as exc:
        logger.error("Drive folder listing failed for %s: %s", folder_url, exc)
        raise HTTPException(
            status_code=400,
            detail="Unable to load Google Drive folder items. Ensure the folder is public.",
        )


@router.post("/upload", response_model=UploadResponse)
async def upload_pdfs(
    files: List[UploadFile] = File(...),
    persist_mode: Literal["auto", "local", "database"] = Query(
        "auto",
        description="Persistence mode: auto (default), local (disk only), database (require DATABASE_URL)",
    ),
    user: UserRecord = Depends(get_current_user),
):
    """Upload and process PDF files, then persist to Supabase Storage."""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    collection_id = str(uuid.uuid4())
    collection_path = os.path.join(config.upload_folder, collection_id)
    os.makedirs(collection_path, exist_ok=True)

    saved_files: List[str] = []
    file_names: List[str] = []
    for file in files:
        file_name = file.filename or ""
        if not file_name.lower().endswith('.pdf'):
            continue
        safe_name = _safe_pdf_filename(file_name, f"upload-{len(saved_files) + 1}")
        file_path = os.path.join(collection_path, safe_name)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved_files.append(file_path)
        file_names.append(safe_name)

    if not saved_files:
        raise HTTPException(
            status_code=400, detail="No valid PDF files uploaded")

    try:
        chunk_count = await asyncio.to_thread(process_pdfs, saved_files, collection_id)
    except Exception as e:
        shutil.rmtree(collection_path, ignore_errors=True)
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process PDFs")

    _register_uploaded_collection(
        collection_id,
        saved_files,
        file_names,
        chunk_count,
        persist_mode=persist_mode,
        owner_id=user.user_id,
    )

    if persist_mode == "database":
        _cleanup_local_artifacts(collection_id, collection_path)

    return UploadResponse(
        collection_id=collection_id,
        file_count=len(saved_files),
        status="success",
        file_names=file_names,
    )


@router.post("/upload-from-url", response_model=UploadResponse)
async def upload_pdf_from_url(
    payload: UploadFromUrlRequest,
    persist_mode: Literal["auto", "local", "database"] = Query(
        "auto",
        description="Persistence mode: auto (default), local (disk only), database (require DATABASE_URL)",
    ),
    user: UserRecord = Depends(get_current_user),
):
    """Download a public PDF from a remote URL (including Google Drive public links) and process it as a collection."""
    source_url = payload.url.strip()
    if not source_url:
        raise HTTPException(status_code=400, detail="URL is required")

    collection_id = str(uuid.uuid4())
    collection_path = os.path.join(config.upload_folder, collection_id)
    os.makedirs(collection_path, exist_ok=True)

    collection_title = _normalize_title(payload.title)

    try:
        fallback_name = f"remote-{collection_id[:8]}"
        final_path, file_name = await _download_pdf_to_collection(
            source_url,
            collection_path,
            fallback_name,
        )

        chunk_count = await asyncio.to_thread(process_pdfs, [final_path], collection_id)
        if chunk_count <= 0:
            raise HTTPException(status_code=400, detail="No readable text was found in the PDF")

        _register_uploaded_collection(
            collection_id,
            [final_path],
            [file_name],
            chunk_count,
            title=collection_title,
            persist_mode=persist_mode,
            owner_id=user.user_id,
        )

        if persist_mode == "database":
            _cleanup_local_artifacts(collection_id, collection_path)

        return UploadResponse(
            collection_id=collection_id,
            file_count=1,
            status="success",
            file_names=[file_name],
            title=collection_title,
        )
    except HTTPException:
        shutil.rmtree(collection_path, ignore_errors=True)
        raise
    except httpx.HTTPError as exc:
        shutil.rmtree(collection_path, ignore_errors=True)
        logger.error("Remote upload failed for %s: %s", source_url, exc)
        raise HTTPException(
            status_code=400,
            detail="Unable to download the remote PDF. Check that the link is public and accessible.",
        )
    except Exception as exc:
        shutil.rmtree(collection_path, ignore_errors=True)
        logger.error("Remote PDF processing failed for %s: %s", source_url, exc)
        raise HTTPException(status_code=500, detail="Failed to process remote PDF")


@router.post("/upload-from-urls", response_model=UploadResponse)
async def upload_pdfs_from_urls(
    payload: UploadFromUrlsRequest,
    persist_mode: Literal["auto", "local", "database"] = Query(
        "auto",
        description="Persistence mode: auto (default), local (disk only), database (require DATABASE_URL)",
    ),
    user: UserRecord = Depends(get_current_user),
):
    urls = [item.strip() for item in (payload.urls or []) if item and item.strip()]
    if not urls:
        raise HTTPException(status_code=400, detail="At least one URL is required")

    collection_id = str(uuid.uuid4())
    collection_path = os.path.join(config.upload_folder, collection_id)
    os.makedirs(collection_path, exist_ok=True)
    collection_title = _normalize_title(payload.title)

    saved_files: List[str] = []
    file_names: List[str] = []
    failures: List[str] = []

    try:
        for index, source_url in enumerate(urls, start=1):
            fallback_name = f"remote-{collection_id[:8]}-{index}"
            try:
                final_path, file_name = await _download_pdf_to_collection(
                    source_url,
                    collection_path,
                    fallback_name,
                )
                saved_files.append(final_path)
                file_names.append(file_name)
            except HTTPException as exc:
                failures.append(f"{source_url} ({exc.detail})")
            except Exception as exc:
                failures.append(f"{source_url} ({str(exc)})")

        if not saved_files:
            failure_message = failures[0] if failures else "Unable to download selected files"
            raise HTTPException(status_code=400, detail=f"No valid PDF files imported: {failure_message}")

        chunk_count = await asyncio.to_thread(process_pdfs, saved_files, collection_id)
        if chunk_count <= 0:
            raise HTTPException(status_code=400, detail="No readable text was found in the selected PDFs")

        _register_uploaded_collection(
            collection_id,
            saved_files,
            file_names,
            chunk_count,
            title=collection_title,
            persist_mode=persist_mode,
            owner_id=user.user_id,
        )

        if persist_mode == "database":
            _cleanup_local_artifacts(collection_id, collection_path)

        return UploadResponse(
            collection_id=collection_id,
            file_count=len(saved_files),
            status="success",
            file_names=file_names,
            title=collection_title,
        )
    except HTTPException:
        shutil.rmtree(collection_path, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(collection_path, ignore_errors=True)
        logger.error("Remote PDFs processing failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to process selected remote PDFs")
