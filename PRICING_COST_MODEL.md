# DocuLens — Pricing vs. Operational Cost Model

Working notes to check whether the current pricing (Individual Rp65.000/mo, Team
Rp500.000/mo, Enterprise custom) actually holds up against real infrastructure
cost. All prices pulled live (July 2026); exchange rate assumed at **Rp16.000/USD**
— re-check this if it moves significantly.

## 1. Gemini cost per query

Current pipeline config (`processor.py` / `agnostic/generator.py`): top-8 retrieved
chunks × ~2000 chars each ≈ 16.000 chars of context per query.

**Gemini 2.5 Flash pricing** (ai.google.dev, July 2026):
| | Per 1M tokens | Per 1M tokens (IDR) |
|---|---|---|
| Input  | $0.30 | Rp4.800 |
| Output | $2.50 | Rp40.000 |

Token estimate per query (blended English/Indonesian, ~4 chars/token):
- Context (16.000 chars) ≈ 4.000 tokens
- System prompt + instructions ≈ 450 tokens
- Question ≈ 50 tokens
- **Input total ≈ 4.500 tokens**
- **Output (typical answer) ≈ 300 tokens**

**Cost per query:**
```
Input:  4.500 / 1.000.000 × Rp4.800  = Rp21,6
Output:   300 / 1.000.000 × Rp40.000 = Rp12,0
─────────────────────────────────────────────
Total ≈ Rp33,6 per query
```

This lines up closely with what you already observed in real testing (~Rp5.xxx
over 3 days of testing ≈ Rp33–40/query at ~150 queries) — the estimate isn't
theoretical, it matches lived data.

## 2. Monthly Gemini cost by plan

Usage isn't daily — it clusters at the start of a need, per your own observation.
Modeled as light vs. heavy:

| Plan | Users | Queries/mo (light → heavy) | Gemini cost/mo |
|---|---|---|---|
| Individual | 1 | 20 → 100 | Rp672 → Rp3.360 |
| Team | 6 (5 members + 1 admin) | 150 → 600 (pooled) | Rp5.040 → Rp20.160 |

Against revenue (Rp65.000 / Rp500.000), Gemini cost alone is **~1–5% of Individual
revenue** and **~1–4% of Team revenue**. Not a margin concern at any realistic
usage level — even a heavy outlier user querying 10× the assumed rate barely dents
the margin.

## 3. Storage cost model

Two different storage needs, priced very differently — worth splitting them
instead of putting everything in one bucket:

**a) Raw file blobs (PDFs, chat log exports)** — belongs in object storage, not
   a database row. **Cloudflare R2**: $0.015/GB-month, **zero egress fee** (unlike
   S3, which charges for data leaving the bucket — matters once users start
   viewing/downloading their own PDFs back through the app).
   → Rp240/GB-month

**b) Embeddings + relational data** (chunks, metadata, sessions, workspace/user
   tables — already Postgres via `psycopg2`/`DATABASE_URL` per the existing
   `public_links.py` / `database_connections.py` code) — needs a real database,
   not flat files. Two managed Postgres options with pgvector support:

   | | Included | Overage | Notes |
   |---|---|---|---|
   | **Supabase Pro** | $25/mo flat, 8GB included | $0.125/GB beyond 8GB | Bundles auth, storage, realtime — useful if you consolidate more of the stack onto it |
   | **Neon** | Pay-as-you-go, no flat fee | $0.35/GB-month metered | Serverless, instant branching — nice for per-environment (staging/prod) isolation |

   At current scale, **Supabase Pro's flat $25/mo (~Rp400.000) covers the vector
   +
   metadata table comfortably** (embeddings are a fraction of raw file size —
   a 384-dim MiniLM vector is ~1.5KB regardless of how big the source PDF was),
   so this cost is closer to a fixed cost than a per-GB variable one until you
   have a lot of users.

**Storage plan limits vs. actual cost:**
| Plan | Storage limit | Object storage cost (R2) at full usage |
|---|---|---|
| Individual | 5 GB | Rp1.200/mo |
| Team | 30 GB | Rp7.200/mo |
| Enterprise | Custom | Passed through / negotiated |

Even at 100% utilization of the advertised limit, storage costs a rounding error
against either plan's revenue.

## 4. Combined small-scale margin check

Modeled at an early-stage scale — 50 Individual + 3 Team subscribers:

```
Revenue:
  50 × Rp65.000                = Rp3.250.000
   3 × Rp500.000                = Rp1.500.000
                                  ───────────
                                  Rp4.750.000/mo

Cost:
  Postgres (Supabase Pro, flat)  = Rp400.000/mo
  R2 storage (50×5GB + 3×30GB
    = 340GB × Rp240)             = Rp81.600/mo
  Gemini (3.400 queries × Rp34)  = Rp115.600/mo
                                  ───────────
                                  Rp597.200/mo

Gross margin ≈ Rp4.152.800/mo  →  ~87% margin
```

**Verdict: the pricing makes sense.** At this scale the cost structure is
dominated by a flat Postgres fee, not usage — meaning margin should *improve*
as you add subscribers, not erode, right up until you outgrow the Supabase Pro
tier (8GB) or need to upgrade compute for concurrent query load. Gemini cost is
structurally small because it scales with *query volume*, not with *how much
is stored* — a user who uploads 5GB but rarely asks questions costs you almost
nothing beyond storage.

## 5. Database recommendation

**Split storage by type, don't force one system to do both jobs:**

1. **Raw files → Cloudflare R2** (or any S3-compatible object store). Cheapest
   per-GB, zero egress cost, and it's what object storage is for — don't store
   PDF/chat-log blobs as bytea columns in Postgres.
2. **Vectors + relational data → Postgres with `pgvector`** (Supabase or Neon).
   This is the bigger architectural recommendation: the current pipeline holds
   FAISS indexes in-process memory, rebuilt/cached per collection
   (`vector_store_cache` in `processor.py`). That works today but doesn't
   persist cleanly across restarts/scaling and doesn't fit the multi-tenant
   `workspace_id` model already present in `models.py`. Moving embeddings into
   `pgvector` means:
   - One system of record for both relational data (sessions, workspaces, DB
     connections) and vector search — less operational surface than running
     FAISS + Postgres as two separate stateful systems.
   - Vector data persists naturally with the rest of the tenant's data instead
     of needing a rebuild-on-cold-start step.
   - `workspace_id`-scoped rows work as a straightforward `WHERE` filter on the
     vector table — natural fit for the Team plan's shared-workspace model.
   - BM25 hybrid search (already built per project memory) still runs
     alongside — pgvector doesn't replace that, it replaces the FAISS layer.

   Between Supabase and Neon: **start with Supabase** — the flat $25/mo Pro tier
   is cheaper than Neon's metered $0.35/GB at your current data volumes, and the
   bundled auth/storage features reduce how much custom infra you maintain.
   Revisit Neon if/when you want per-environment branching or outgrow a single
   Postgres instance's compute.

## Caveats / things not modeled here

- Embedding *compute* cost isn't counted — `sentence-transformers` runs
  self-hosted (per project config), so it's a server/CPU cost, not a metered
  API cost. Worth revisiting if you move embedding generation to a hosted API.
- No rate-limiting/abuse model — a flat Rp65.000/mo with no query cap means a
  single abusive/scripted user could run far more than the "light/heavy"
  assumptions above. Worth a soft per-user monthly query cap before scaling
  paid signups, even if generous, just to bound worst-case Gemini spend.
- Numbers assume Gemini 2.5 Flash stays the primary model — the fallback chain
  in `generator.py` drops to `gemini-2.0-flash`/`gemini-1.5-flash` on retry,
  and 2.0 Flash is already noted as deprecated (shutdown June 2026), so that
  fallback tier may need re-checking.
