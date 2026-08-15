# Scope
FastAPI backend for DocuLens (hybrid BM25+FAISS RAG). Routers live in `router/`, one file per domain, all mounted under `/api/v1` in `main.py`.

# Route Convention
Every domain gets exactly ONE base path — list/create/delete/activate/upload/etc. are sub-paths under that one prefix, not scattered across different prefixes:

```
/api/v1/pdf-collections            GET (list) — router/collections.py
/api/v1/pdf-collections/{id}       DELETE     — router/collections.py
/api/v1/pdf-collections/activate   POST       — router/collections.py
/api/v1/pdf-collections/upload             POST — router/upload.py
/api/v1/pdf-collections/upload-from-url    POST — router/upload.py
/api/v1/pdf-collections/upload-from-urls   POST — router/upload.py
/api/v1/pdf-collections/drive/folder-items POST — router/upload.py

/api/v1/chat-collections            GET/DELETE/upload/activate/preview — router/chat.py

/api/v1/database-connections        GET/POST (list/create)
/api/v1/database-connections/{id}/tables  GET
/api/v1/database-connections/activate     POST
/api/v1/database-connections/{id}         DELETE
                                     — router/database_connections.py

/api/v1/telegram-connections        GET (list) + connect/start, connect/verify,
                                     {id}/dialogs, {id}/sync, activate, {id} DELETE
                                     — router/telegram.py

/api/v1/public-links                GET/POST (list/create) + activate, {id} DELETE
                                     — router/public_links.py

/api/v1/analysis/gap-analysis       POST (run) + /runs GET
                                     — router/compliance.py

/api/v1/auth                        register/login/me/change-password/admin/...
                                     — router/auth.py
```

**Do not** register a new endpoint for one of these domains under a different prefix (e.g. don't add `POST /pdf-collection/foo` — it goes under `/api/v1/pdf-collections/foo`). This was cleaned up FROM exactly that (plural/singular mismatches, unrelated prefixes like `pdf-collection` vs `collection` vs bare `/upload`) — keep it consolidated.

The one intentional exception: `GET /files/{collection_id}/{file_name}` and `GET /collection/{collection_id}/files` in `router/collections.py` stay on their own path — they're referenced as literal URL strings returned to the client (file download links), not called by name from the frontend's endpoint enum, so renaming them has no cleanup benefit and only risks breaking existing links.

# Two Backends Must Stay Identical
This repo (`pdf-reader`, local dev backend) and `hf-doculens-api` (deployed HF Space) are kept functionally identical — **any change to a router file here must be mirrored to the same file in `hf-doculens-api`, and vice versa.** Verify both with `python -m py_compile router/<file>.py` and `python -c "import main"` after changing either.

# Frontend Contract
`chat-ui` (Next.js) is the only consumer of this API. Its `services/endpoint.ts` is the single source of truth for every path this backend must serve — one entry per domain, matching the prefixes above. If you rename a route here, update `endpoint.ts` in `chat-ui` in the same change (and vice versa) — the two must never drift apart, since there's no separate API contract/OpenAPI doc to catch a mismatch.
