# router/payment.py
"""
Dummy/test-mode payment flow (Stripe Checkout, one-time payment) — MS-90.

Endpoints:
  POST /payments/checkout-session  — create a Stripe Checkout Session for a plan
  POST /payments/webhook           — Stripe webhook (no auth; verifies signature)
  GET  /payments/session/{id}      — this app's own record for a checkout session

Schema (auto-created via _ensure_tables, same convention as
router/telegram.py / router/database_connections.py):
  payments (payment_id, user_id, plan_id, amount, currency, status,
            stripe_checkout_session_id, stripe_payment_intent_id,
            created_at, updated_at)

Prices are defined here, server-side, and never trusted from the client —
otherwise a request could tamper with the amount actually charged.

This is a one-time payment (Checkout mode="payment"), not a subscription —
recurring billing/proration is a larger scope than validating the checkout
flow this ticket is about.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import stripe
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

import app_db
from config import config
from router.auth import get_current_user, require_role, UserRecord
from models import (
    CreateCheckoutSessionRequest,
    CheckoutSessionResponse,
    PaymentRecord,
    PaymentResponse,
    SubscriptionUsage,
    MemberTokenUsage,
    MyMemberUsageResponse,
    MembersUsageResponse,
    UpdateMemberAllocationRequest,
    UpdateMemberAllocationResponse,
    RateLimitStatus,
    CreateTokenRequestRequest,
    TokenRequestRecord,
    TokenRequestResponse,
    TokenRequestsResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()

stripe.api_key = config.stripe_secret_key

# The checkout flow (MS-90) has no login step in its path — Pricing → Select
# Plan → Payment → Gateway → Result — so it must work for a guest. Auth is
# accepted (attributes the payment to a real user_id) but never required:
# reuses get_current_user's own JWT validation instead of duplicating it,
# just swallows the 401 for a missing/invalid token instead of raising.
_optional_bearer = HTTPBearer(auto_error=False)


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_optional_bearer),
) -> Optional[UserRecord]:
    if not credentials:
        return None
    try:
        return get_current_user(credentials)
    except HTTPException:
        return None

# amount is already in Stripe's minor unit for IDR (Rupiah x 100) — IDR is a
# standard two-decimal currency for Stripe, not zero-decimal like JPY/KRW.
# Matches the marketing prices in chat-ui's lib/pricing-plans.ts. Enterprise
# is excluded — that plan is sales-assisted, not self-serve checkout.
PLAN_PRICES = {
    "individual": {"name": "Individual", "amount": 6_500_000},
    "team": {"name": "Team", "amount": 50_000_000},
}

# Token allowance granted per subscription period (MS-248). Same plan_ids as
# PLAN_PRICES above; Enterprise is sales-assisted/custom, no self-serve cap.
# "free" has no entry in PLAN_PRICES (nothing to check out) but does need one
# here — see _get_latest_plan_window's fallback branch below — so a workspace
# that's never paid still gets a real, enforced cap instead of relying on
# the flat safety-net alone.
PLAN_QUOTAS = {
    "free": {"name": "Free", "token_limit": config.free_plan_token_limit},
    "individual": {"name": "Individual", "token_limit": 2_000_000},
    "team": {"name": "Team", "token_limit": 10_000_000},
}

# No real recurring billing exists yet (see module docstring — Checkout is
# mode="payment", one-time) — the "current period" is a fixed window rolling
# forward from the most recent succeeded payment, not a Stripe-driven renewal.
SUBSCRIPTION_PERIOD_DAYS = 30

_tables_ensured = False
_usage_tables_ensured = False


def _get_app_conn():
    """Connection to THIS app's own database (metadata store, not a data source)."""
    return app_db.get_app_conn("payment")


def _ensure_tables(conn) -> None:
    global _tables_ensured
    if _tables_ensured:
        return
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS payments (
                    id                          BIGSERIAL PRIMARY KEY,
                    payment_id                  TEXT        NOT NULL UNIQUE,
                    user_id                     TEXT        NOT NULL,
                    plan_id                     TEXT        NOT NULL,
                    amount                      INTEGER     NOT NULL,
                    currency                    TEXT        NOT NULL DEFAULT 'idr',
                    status                      TEXT        NOT NULL DEFAULT 'pending'
                                                CHECK (status IN ('pending', 'succeeded', 'failed', 'cancelled')),
                    stripe_checkout_session_id  TEXT        UNIQUE,
                    stripe_payment_intent_id    TEXT,
                    created_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at                  TIMESTAMPTZ NOT NULL DEFAULT now()
                );

                CREATE INDEX IF NOT EXISTS idx_payments_user_created
                    ON payments (user_id, created_at DESC);

                -- Cancel-at-period-end (MS-248 follow-up): set on a
                -- SUCCEEDED payment when its admin cancels the subscription
                -- period it started. Distinct from the `status` column
                -- above (which only ever reaches 'cancelled' pre-success,
                -- via the Stripe webhook's checkout.session.expired case) —
                -- this instead marks a period that was paid for and is
                -- still running, just flagged not to be treated as
                -- renewable/resumable-by-default once it ends.
                ALTER TABLE payments
                    ADD COLUMN IF NOT EXISTS cancelled_at TIMESTAMPTZ;

                CREATE OR REPLACE FUNCTION _set_payments_updated_at()
                RETURNS TRIGGER LANGUAGE plpgsql AS $$
                BEGIN
                    NEW.updated_at = now();
                    RETURN NEW;
                END;
                $$;

                DROP TRIGGER IF EXISTS trg_payments_updated_at ON payments;
                CREATE TRIGGER trg_payments_updated_at
                    BEFORE UPDATE ON payments
                    FOR EACH ROW EXECUTE FUNCTION _set_payments_updated_at();
                """
            )
        _tables_ensured = True
    except Exception as exc:
        logger.error("payment: ensure table failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to initialize payments schema")


def _ensure_usage_tables(conn) -> None:
    """token_usage (append-only consumption ledger) + token_allocations (the
    admin-assigned per-member cap, one row per member) — same auto-create
    convention as _ensure_tables above."""
    global _usage_tables_ensured
    if _usage_tables_ensured:
        return
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS token_usage (
                    id             BIGSERIAL   PRIMARY KEY,
                    user_id        TEXT        NOT NULL,
                    admin_user_id  TEXT        NOT NULL,
                    tokens         INTEGER     NOT NULL,
                    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
                );

                CREATE INDEX IF NOT EXISTS idx_token_usage_admin_created
                    ON token_usage (admin_user_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_token_usage_user_created
                    ON token_usage (user_id, created_at);

                CREATE TABLE IF NOT EXISTS token_allocations (
                    id               BIGSERIAL   PRIMARY KEY,
                    admin_user_id    TEXT        NOT NULL,
                    user_id          TEXT        NOT NULL UNIQUE,
                    allocated_tokens INTEGER     NOT NULL DEFAULT 0,
                    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
                );

                CREATE INDEX IF NOT EXISTS idx_token_allocations_admin
                    ON token_allocations (admin_user_id);

                CREATE OR REPLACE FUNCTION _set_token_allocations_updated_at()
                RETURNS TRIGGER LANGUAGE plpgsql AS $$
                BEGIN
                    NEW.updated_at = now();
                    RETURN NEW;
                END;
                $$;

                DROP TRIGGER IF EXISTS trg_token_allocations_updated_at ON token_allocations;
                CREATE TRIGGER trg_token_allocations_updated_at
                    BEFORE UPDATE ON token_allocations
                    FOR EACH ROW EXECUTE FUNCTION _set_token_allocations_updated_at();

                -- "Request more tokens" (MS-248 follow-up) — a member who
                -- hit their admin-assigned cap can ask for more; the admin
                -- sees pending ones in the Billing tab and actually raises
                -- the cap via the existing allocation editor, then dismisses
                -- the request. In-app only for now (polling, no real push).
                CREATE TABLE IF NOT EXISTS token_requests (
                    id             BIGSERIAL   PRIMARY KEY,
                    request_id     TEXT        NOT NULL UNIQUE,
                    user_id        TEXT        NOT NULL,
                    admin_user_id  TEXT        NOT NULL,
                    message        TEXT,
                    status         TEXT        NOT NULL DEFAULT 'pending'
                                   CHECK (status IN ('pending', 'resolved')),
                    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
                    resolved_at    TIMESTAMPTZ
                );

                CREATE INDEX IF NOT EXISTS idx_token_requests_admin_status
                    ON token_requests (admin_user_id, status, created_at DESC);
                """
            )
        _usage_tables_ensured = True
    except Exception as exc:
        logger.error("payment: ensure usage tables failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to initialize token-usage schema")


def _resolve_admin_user_id(conn, user: UserRecord) -> str:
    """The workspace a user's usage counts against: admins own their
    workspace; a sub-user's workspace is whoever created them (`created_by`,
    see router/auth.py). Falls back to the user's own id if `created_by` is
    somehow unset, so usage still resolves to *some* workspace."""
    if user.role == "admin":
        return user.user_id
    with conn.cursor() as cur:
        cur.execute("SELECT created_by FROM users WHERE user_id = %s", (user.user_id,))
        row = cur.fetchone()
    return (row["created_by"] if row else None) or user.user_id


@dataclass
class PlanWindow:
    plan: dict
    payment_id: Optional[str]  # None for the synthetic free-tier window below
    period_start: datetime
    period_end: datetime
    status: str  # "active" | "expired" — free tier is always "active"
    cancel_at_period_end: bool


def _get_latest_plan_window(conn, admin_user_id: str) -> Optional[PlanWindow]:
    """The admin's most recent succeeded payment and the period it grants —
    or, if they've never paid successfully (or the plan_id on file isn't
    one we recognize), a synthetic Free-tier window (MS-248 follow-up) so
    every workspace has a real, enforced cap instead of relying on the
    flat safety-net rate limit alone. Only returns None if the admin's own
    user row can't be found at all, which shouldn't normally happen.
    `status` here only ever reflects whether a PAID period's `period_end`
    has passed — cancellation doesn't cut a period short (see
    cancel_at_period_end), it just stops it renewing. Free tier has
    nothing to expire from; it just keeps rolling to the next period."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT payment_id, plan_id, created_at, cancelled_at FROM payments
            WHERE user_id = %s AND status = 'succeeded'
            ORDER BY created_at DESC LIMIT 1
            """,
            (admin_user_id,),
        )
        row = cur.fetchone()
    plan = PLAN_QUOTAS.get(row["plan_id"]) if row else None

    if row and plan:
        period_start = row["created_at"]
        period_end = period_start + timedelta(days=SUBSCRIPTION_PERIOD_DAYS)
        now = datetime.now(timezone.utc)
        return PlanWindow(
            plan=plan,
            payment_id=row["payment_id"],
            period_start=period_start,
            period_end=period_end,
            status="active" if now <= period_end else "expired",
            cancel_at_period_end=row["cancelled_at"] is not None,
        )

    # Free tier: no succeeded payment on file (or an unrecognized plan_id).
    # Anchor the rolling window to the workspace admin's own account
    # creation date, since there's no payment date to anchor to, and
    # advance it in fixed SUBSCRIPTION_PERIOD_DAYS steps so a long-lived
    # free account keeps getting fresh periods automatically forever
    # rather than being stuck in (or blocked by) its very first one.
    with conn.cursor() as cur:
        cur.execute("SELECT created_at FROM users WHERE user_id = %s", (admin_user_id,))
        user_row = cur.fetchone()
    if not user_row:
        return None
    period_start = user_row["created_at"]
    now = datetime.now(timezone.utc)
    period_length = timedelta(days=SUBSCRIPTION_PERIOD_DAYS)
    periods_elapsed = max(0, (now - period_start) // period_length)
    period_start = period_start + periods_elapsed * period_length
    period_end = period_start + period_length
    return PlanWindow(
        plan=PLAN_QUOTAS["free"],
        payment_id=None,
        period_start=period_start,
        period_end=period_end,
        status="active",
        cancel_at_period_end=False,
    )


def _sum_tokens_for_admin(conn, admin_user_id: str, period_start, period_end) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COALESCE(SUM(tokens), 0) AS used FROM token_usage
            WHERE admin_user_id = %s AND created_at >= %s AND created_at < %s
            """,
            (admin_user_id, period_start, period_end),
        )
        return int(cur.fetchone()["used"] or 0)


def _sum_tokens_for_user(conn, user_id: str, period_start, period_end) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COALESCE(SUM(tokens), 0) AS used FROM token_usage
            WHERE user_id = %s AND created_at >= %s AND created_at < %s
            """,
            (user_id, period_start, period_end),
        )
        return int(cur.fetchone()["used"] or 0)


def _sum_tokens_by_user(conn, user_ids: list, period_start, period_end) -> dict:
    """Same as _sum_tokens_for_user but for a whole team in one query — used
    by get_members_usage so an admin's Billing tab doesn't issue one
    round-trip per member (N+1)."""
    if not user_ids:
        return {}
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT user_id, COALESCE(SUM(tokens), 0) AS used FROM token_usage
            WHERE user_id = ANY(%s) AND created_at >= %s AND created_at < %s
            GROUP BY user_id
            """,
            (user_ids, period_start, period_end),
        )
        rows = cur.fetchall()
    return {r["user_id"]: int(r["used"] or 0) for r in rows}


def _build_subscription_usage(
    conn, admin_user_id: str, window: Optional[PlanWindow] = None
) -> Optional[SubscriptionUsage]:
    """`window` lets a caller that already fetched one (e.g. to validate
    status before mutating something) pass it through instead of paying for
    a second identical `_get_latest_plan_window` query."""
    if window is None:
        window = _get_latest_plan_window(conn, admin_user_id)
    if not window:
        return None
    token_limit = window.plan["token_limit"]
    token_used = _sum_tokens_for_admin(conn, admin_user_id, window.period_start, window.period_end)
    # No auto-renewal exists regardless (see module docstring), so
    # next_reset_date only means "you can keep using this plan past
    # period_end without lifting a finger" — not true once cancelled.
    renews = window.status == "active" and not window.cancel_at_period_end
    return SubscriptionUsage(
        plan_name=window.plan["name"],
        subscription_status=window.status,
        token_limit=token_limit,
        token_used=token_used,
        token_remaining=max(0, token_limit - token_used),
        period_start=app_db.ts(window.period_start),
        period_end=app_db.ts(window.period_end),
        next_reset_date=app_db.ts(window.period_end) if renews else None,
        cancel_at_period_end=window.cancel_at_period_end,
        is_paid=window.payment_id is not None,
    )


def _get_rate_limit_status(conn, user_id: str) -> RateLimitStatus:
    """Flat, plan-independent safety-net rate limit — sliding window over
    just THIS user's own token_usage rows in the last
    config.rate_limit_window_hours (not the workspace-wide allocation pool
    used by _build_subscription_usage). No fixed reset clock: usage simply
    ages out of the window over time, so `reset_at` below is the earliest
    moment that happens naturally, not a scheduled job."""
    window_hours = config.effective_rate_limit_window_hours
    cap = config.rate_limit_token_cap
    window_start = datetime.now(timezone.utc) - timedelta(hours=window_hours)

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT tokens, created_at FROM token_usage
            WHERE user_id = %s AND created_at >= %s
            ORDER BY created_at ASC
            """,
            (user_id, window_start),
        )
        rows = cur.fetchall()

    used = sum(r["tokens"] for r in rows)
    blocked = cap > 0 and used >= cap
    reset_at = None
    if blocked:
        # Drop rows oldest-first until the remaining sum clears the cap —
        # the row that tips it under is the one whose own age-out moment
        # (its timestamp + the window) is when the user can send again.
        running = used
        for row in rows:
            running -= row["tokens"]
            if running < cap:
                reset_at = row["created_at"] + timedelta(hours=window_hours)
                break

    return RateLimitStatus(
        used_tokens=used,
        cap_tokens=cap,
        window_hours=window_hours,
        blocked=blocked,
        reset_at=app_db.ts(reset_at) if reset_at else None,
    )


def enforce_rate_limit(user_id: str) -> None:
    """Raise HTTPException(429) if this user has hit the flat safety-net
    rate limit. Call this from router/agnostic.py BEFORE the LLM is
    invoked — unlike log_token_usage (best-effort, never raises), this one
    is meant to actually block overage, so callers should let it propagate
    rather than swallowing it. Fails open (never blocks) if the metering DB
    itself is unreachable — a metering outage shouldn't take down chat."""
    conn = _get_app_conn()
    if not conn:
        return
    try:
        _ensure_usage_tables(conn)
        status = _get_rate_limit_status(conn, user_id)
    finally:
        conn.close()
    if status.blocked:
        raise HTTPException(
            status_code=429,
            detail="Batas token untuk akun kamu sudah tercapai untuk saat ini. Coba lagi setelah beberapa saat.",
        )


# Serializes a workspace's check-work-log cycle across enforce_rate_limit /
# enforce_plan_limit / enforce_member_allocation so concurrent requests
# can't all read "not yet blocked" before any of them has logged its
# usage — the classic check-then-act race a flat "check once, log after
# the LLM call returns" design otherwise leaves open.
#
# Keyed by WORKSPACE (the resolved admin_user_id — see resolve_workspace_id
# below), not by the calling user_id: enforce_plan_limit and
# enforce_member_allocation both check state shared across an entire team
# (the workspace's total usage vs. its plan's token_limit, and the
# allocation pool carved out of it), not just the caller's own. Two
# different members of the same team racing concurrently must serialize
# against EACH OTHER for those checks to mean anything — locking only on
# each member's own user_id (as an earlier version of this did) would let
# them race straight past a shared workspace ceiling together. This still
# never serializes different workspaces against each other.
#
# In-process only (fine for this app's current single-worker deployment; a
# multi-worker/multi-instance deployment would need a DB-level lock
# instead — see the module docstring on the one-time-payment model this
# whole file is built on for the same "not built for scale yet" caveat).
# Entries are never evicted; each is one tiny asyncio.Lock object, an
# acceptable tradeoff at this app's scale rather than added complexity.
_workspace_locks: dict[str, asyncio.Lock] = {}


def resolve_workspace_id(user: UserRecord) -> str:
    """The lock key for `user`'s workspace — their own user_id if they're
    an admin, otherwise whichever admin created them. Opens its own
    short-lived connection; call this BEFORE acquiring the lock (it can't
    be resolved from inside it, since resolving it needs a query and the
    point of the lock is to serialize queries). Fails open to the caller's
    own user_id (still safe, just narrower-than-ideal serialization) if
    the metering DB is unreachable."""
    if user.role == "admin":
        return user.user_id
    conn = _get_app_conn()
    if not conn:
        return user.user_id
    try:
        return _resolve_admin_user_id(conn, user)
    finally:
        conn.close()


def get_workspace_lock(workspace_id: str) -> asyncio.Lock:
    lock = _workspace_locks.get(workspace_id)
    if lock is None:
        lock = asyncio.Lock()
        _workspace_locks[workspace_id] = lock
    return lock


def enforce_plan_limit(user: UserRecord) -> None:
    """Raise HTTPException(402) once the WHOLE workspace has used up its
    plan's own token_limit for the current period (MS-248 follow-up) —
    Free/Individual/Team's own ceiling, separate from the flat per-user
    safety net (enforce_rate_limit) and the optional per-member allocation
    (enforce_member_allocation). Every workspace always has a plan window
    now (Free is the fallback in _get_latest_plan_window when nobody's
    paid), so this always applies, not just to paying workspaces. Fails
    open if the metering DB is unreachable, same as the other checks."""
    conn = _get_app_conn()
    if not conn:
        return
    try:
        _ensure_tables(conn)
        _ensure_usage_tables(conn)
        admin_user_id = _resolve_admin_user_id(conn, user)
        window = _get_latest_plan_window(conn, admin_user_id)
        if not window or window.status != "active":
            return
        used = _sum_tokens_for_admin(conn, admin_user_id, window.period_start, window.period_end)
        token_limit = window.plan["token_limit"]
        plan_name = window.plan["name"]
    finally:
        conn.close()
    if used >= token_limit:
        raise HTTPException(
            status_code=402,
            detail=f"Jatah token workspace untuk plan {plan_name} sudah habis untuk periode ini.",
        )


def enforce_member_allocation(user: UserRecord) -> None:
    """Raise HTTPException(403) if this user has an assigned token cap (a
    row in token_allocations) and has used it up this period. No row at
    all means nobody has capped this account individually — they're still
    subject to the flat rate limit (enforce_rate_limit), just not this
    per-member one. Applies to admins too (MS-248 follow-up): an admin can
    optionally allocate themselves a slice of the workspace pool for their
    own budget discipline, same mechanism as any team member, and can
    always raise it back up to whatever's unallocated since they're the
    one who controls it. Fails open if the metering DB is unreachable,
    same as enforce_rate_limit."""
    conn = _get_app_conn()
    if not conn:
        return
    try:
        _ensure_tables(conn)
        _ensure_usage_tables(conn)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT allocated_tokens FROM token_allocations WHERE user_id = %s",
                (user.user_id,),
            )
            alloc_row = cur.fetchone()
        if not alloc_row:
            return
        admin_user_id = _resolve_admin_user_id(conn, user)
        window = _get_latest_plan_window(conn, admin_user_id)
        if not window or window.status != "active":
            return
        used = _sum_tokens_for_user(conn, user.user_id, window.period_start, window.period_end)
    finally:
        conn.close()
    if used >= alloc_row["allocated_tokens"]:
        raise HTTPException(
            status_code=403,
            detail="Token cap yang diberikan admin untuk akun kamu sudah habis untuk periode ini.",
        )


def log_token_usage(user_id: str, tokens: int) -> None:
    """Best-effort: append one row to the consumption ledger for a query
    that just ran. Called from router/agnostic.py right after a live LLM
    answer is generated. Never raises — a metering hiccup must not break a
    chat response that already succeeded; callers should still wrap this in
    their own try/except as a second line of defense."""
    if tokens <= 0:
        return
    conn = _get_app_conn()
    if not conn:
        logger.warning("payment: log_token_usage skipped, no DB connection")
        return
    try:
        _ensure_usage_tables(conn)
        with conn.cursor() as cur:
            cur.execute("SELECT role, created_by FROM users WHERE user_id = %s", (user_id,))
            row = cur.fetchone()
        admin_user_id = user_id
        if row and row["role"] != "admin":
            admin_user_id = row["created_by"] or user_id
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO token_usage (user_id, admin_user_id, tokens) VALUES (%s, %s, %s)",
                (user_id, admin_user_id, tokens),
            )
    except Exception as exc:
        logger.warning("payment: log_token_usage failed for user %s: %s", user_id, exc)
    finally:
        conn.close()


def _as_record(row) -> PaymentRecord:
    return PaymentRecord(
        payment_id=row["payment_id"],
        plan_id=row["plan_id"],
        amount=row["amount"],
        currency=row["currency"],
        status=row["status"],
        created_at=app_db.ts(row.get("created_at")),
    )


@router.post(
    "/payments/checkout-session",
    response_model=CheckoutSessionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_checkout_session(
    body: CreateCheckoutSessionRequest,
    user: Optional[UserRecord] = Depends(get_optional_user),
):
    user_id = user.user_id if user else "guest"
    plan = PLAN_PRICES.get(body.plan_id)
    if not plan:
        raise HTTPException(status_code=400, detail="Unknown plan")

    if not config.stripe_secret_key:
        raise HTTPException(status_code=503, detail="Payments are not configured")

    conn = _get_app_conn()
    if not conn:
        raise HTTPException(status_code=503, detail="Database unavailable")

    _ensure_tables(conn)
    payment_id = f"pay_{uuid.uuid4().hex}"

    try:
        session = stripe.checkout.Session.create(
            mode="payment",
            payment_method_types=["card"],
            line_items=[
                {
                    "price_data": {
                        "currency": "idr",
                        "product_data": {"name": f"DocuLens {plan['name']} plan"},
                        "unit_amount": plan["amount"],
                    },
                    "quantity": 1,
                }
            ],
            success_url=(
                f"{config.frontend_url}/payment/result"
                "?status=success&session_id={CHECKOUT_SESSION_ID}"
            ),
            cancel_url=f"{config.frontend_url}/payment/result?status=cancelled",
            metadata={"user_id": user_id, "plan_id": body.plan_id, "payment_id": payment_id},
        )
    except Exception as exc:
        logger.error("payment: checkout session creation failed: %s", exc)
        conn.close()
        raise HTTPException(status_code=502, detail="Could not start checkout")

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO payments (payment_id, user_id, plan_id, amount, currency, status, stripe_checkout_session_id)
                VALUES (%s, %s, %s, %s, 'idr', 'pending', %s)
                """,
                (payment_id, user_id, body.plan_id, plan["amount"], session.id),
            )
    except Exception as exc:
        logger.error("payment: failed to record pending payment: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to start checkout")
    finally:
        conn.close()

    return CheckoutSessionResponse(checkout_url=session.url, payment_id=payment_id)


@router.post("/payments/webhook")
async def stripe_webhook(request: Request):
    """No auth — Stripe calls this directly. First raw-body endpoint in this
    codebase: signature verification needs the exact raw bytes, not a parsed
    Pydantic model."""
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")

    if not config.stripe_webhook_secret:
        # Not configured (e.g. local dev without `stripe listen` yet) — ack
        # rather than 500, since Stripe retries failed webhooks aggressively.
        logger.warning("payment: webhook received but STRIPE_WEBHOOK_SECRET is not set")
        return {"received": True}

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, config.stripe_webhook_secret)
    except Exception as exc:
        logger.warning("payment: webhook signature verification failed: %s", exc)
        raise HTTPException(status_code=400, detail="Invalid webhook signature")

    conn = _get_app_conn()
    if not conn:
        # Let Stripe retry rather than silently losing the event.
        raise HTTPException(status_code=503, detail="Database unavailable")

    _ensure_tables(conn)
    try:
        event_type = event["type"]
        # stripe-python 15.x's typed objects (e.g. Session) support [] item
        # access but not .get() — .to_dict() gives a plain dict so both the
        # .get() calls and the ["id"] access below behave as expected.
        data = event["data"]["object"].to_dict()

        # Only these two are handled: `checkout.session.completed` is the
        # reliable, documented success signal for mode="payment" Checkout;
        # `checkout.session.expired` covers an abandoned/timed-out session.
        # payment_intent.payment_failed is deliberately not handled — we
        # don't have a stripe_payment_intent_id on file to match against
        # until a session actually completes, so it can't reliably find the
        # right row anyway.
        if event_type == "checkout.session.completed":
            new_status = "succeeded" if data.get("payment_status") == "paid" else "failed"
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE payments SET status = %s, stripe_payment_intent_id = %s
                    WHERE stripe_checkout_session_id = %s
                    """,
                    (new_status, data.get("payment_intent"), data["id"]),
                )
        elif event_type == "checkout.session.expired":
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE payments SET status = 'cancelled' WHERE stripe_checkout_session_id = %s",
                    (data["id"],),
                )
    except Exception as exc:
        logger.error("payment: webhook handling failed: %s", exc)
    finally:
        conn.close()

    return {"received": True}


@router.get("/payments/session/{session_id}", response_model=PaymentResponse)
async def get_payment_by_session(session_id: str):
    """No auth required — same guest-friendly model as checkout-session.
    Authorization here is possession of the Stripe-generated session_id
    itself (only known to the browser Stripe just redirected back), the
    same trust model as a typical guest order-confirmation link."""
    conn = _get_app_conn()
    if not conn:
        raise HTTPException(status_code=503, detail="Database unavailable")

    _ensure_tables(conn)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM payments WHERE stripe_checkout_session_id = %s",
                (session_id,),
            )
            row = cur.fetchone()
    finally:
        conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Payment not found")

    return PaymentResponse(payment=_as_record(row))


# ===================== TOKEN USAGE & ALLOCATION (MS-248) =====================


@router.get("/payments/subscription/me", response_model=MyMemberUsageResponse)
async def get_my_usage(user: UserRecord = Depends(get_current_user)):
    """Any authenticated user — their own allocation/consumption within
    their workspace's current subscription period, for the Usage tab."""
    conn = _get_app_conn()
    if not conn:
        raise HTTPException(status_code=503, detail="Database unavailable")
    _ensure_tables(conn)
    _ensure_usage_tables(conn)
    try:
        admin_user_id = _resolve_admin_user_id(conn, user)
        window = _get_latest_plan_window(conn, admin_user_id)
        if not window or window.status != "active":
            return MyMemberUsageResponse(usage=None)

        with conn.cursor() as cur:
            cur.execute(
                "SELECT allocated_tokens FROM token_allocations WHERE user_id = %s",
                (user.user_id,),
            )
            alloc_row = cur.fetchone()
        allocated = alloc_row["allocated_tokens"] if alloc_row else 0
        used = _sum_tokens_for_user(conn, user.user_id, window.period_start, window.period_end)
    finally:
        conn.close()

    usage = MemberTokenUsage(
        user_id=user.user_id,
        email=user.email,
        allocated_tokens=allocated,
        used_tokens=used,
        remaining_tokens=max(0, allocated - used),
        usage_percent=round(used / allocated * 100, 2) if allocated > 0 else 0.0,
    )
    return MyMemberUsageResponse(usage=usage)


@router.get("/payments/subscription/members", response_model=MembersUsageResponse)
async def get_members_usage(admin: UserRecord = Depends(require_role("admin"))):
    """Admin-only — subscription overview plus every team member's own
    allocation/consumption, for the Billing tab's allocation editor. The
    admin is included as the first row (MS-248 follow-up) — they're a
    participant in the same shared pool as their team, not a special case,
    so they can optionally cap their own usage for budget discipline and
    raise it back up themselves whenever they want."""
    conn = _get_app_conn()
    if not conn:
        raise HTTPException(status_code=503, detail="Database unavailable")
    _ensure_tables(conn)
    _ensure_usage_tables(conn)
    try:
        window = _get_latest_plan_window(conn, admin.user_id)
        subscription = _build_subscription_usage(conn, admin.user_id, window=window)

        with conn.cursor() as cur:
            cur.execute(
                "SELECT user_id, email FROM users WHERE created_by = %s ORDER BY created_at DESC",
                (admin.user_id,),
            )
            team_rows = cur.fetchall()
        pool_rows = [{"user_id": admin.user_id, "email": admin.email}] + list(team_rows)

        with conn.cursor() as cur:
            cur.execute(
                "SELECT user_id, allocated_tokens FROM token_allocations WHERE admin_user_id = %s",
                (admin.user_id,),
            )
            alloc_rows = cur.fetchall()
        allocations = {r["user_id"]: r["allocated_tokens"] for r in alloc_rows}

        members: list[MemberTokenUsage] = []
        if window:
            used_by_user = _sum_tokens_by_user(
                conn, [row["user_id"] for row in pool_rows], window.period_start, window.period_end
            )
            for row in pool_rows:
                allocated = allocations.get(row["user_id"], 0)
                used = used_by_user.get(row["user_id"], 0)
                members.append(
                    MemberTokenUsage(
                        user_id=row["user_id"],
                        email=row["email"],
                        allocated_tokens=allocated,
                        used_tokens=used,
                        remaining_tokens=max(0, allocated - used),
                        usage_percent=round(used / allocated * 100, 2) if allocated > 0 else 0.0,
                    )
                )

        token_limit = subscription.token_limit if subscription else 0
        unallocated = max(0, token_limit - sum(allocations.values()))
    finally:
        conn.close()

    return MembersUsageResponse(subscription=subscription, members=members, unallocated_tokens=unallocated)


@router.post("/payments/subscription/cancel", response_model=SubscriptionUsage)
async def cancel_subscription(admin: UserRecord = Depends(require_role("admin"))):
    """Admin-only — cancel-at-period-end (not immediate): the admin already
    paid for the current period, so access/token_limit are untouched until
    period_end. This just stops next_reset_date implying it'll keep going
    past that — there's no auto-renewal to actually cancel (see module
    docstring), so all this does is flag the period as non-renewable."""
    conn = _get_app_conn()
    if not conn:
        raise HTTPException(status_code=503, detail="Database unavailable")
    _ensure_tables(conn)
    _ensure_usage_tables(conn)
    try:
        window = _get_latest_plan_window(conn, admin.user_id)
        if not window or window.status != "active" or window.payment_id is None:
            raise HTTPException(status_code=400, detail="No paid subscription to cancel")
        if window.cancel_at_period_end:
            raise HTTPException(status_code=400, detail="Subscription is already set to cancel")

        with conn.cursor() as cur:
            cur.execute(
                "UPDATE payments SET cancelled_at = now() WHERE payment_id = %s",
                (window.payment_id,),
            )
        subscription = _build_subscription_usage(conn, admin.user_id)
    finally:
        conn.close()
    return subscription


@router.post("/payments/subscription/resume", response_model=SubscriptionUsage)
async def resume_subscription(admin: UserRecord = Depends(require_role("admin"))):
    """Admin-only — undo a pending cancellation, as long as the paid period
    hasn't ended yet (matches standard "resume before it lapses" UX)."""
    conn = _get_app_conn()
    if not conn:
        raise HTTPException(status_code=503, detail="Database unavailable")
    _ensure_tables(conn)
    _ensure_usage_tables(conn)
    try:
        window = _get_latest_plan_window(conn, admin.user_id)
        if not window or window.status != "active":
            raise HTTPException(status_code=400, detail="Subscription period has already ended")
        if not window.cancel_at_period_end:
            raise HTTPException(status_code=400, detail="Subscription isn't set to cancel")

        with conn.cursor() as cur:
            cur.execute(
                "UPDATE payments SET cancelled_at = NULL WHERE payment_id = %s",
                (window.payment_id,),
            )
        subscription = _build_subscription_usage(conn, admin.user_id)
    finally:
        conn.close()
    return subscription


@router.post("/payments/subscription/allocations", response_model=UpdateMemberAllocationResponse)
async def set_member_allocation(
    body: UpdateMemberAllocationRequest,
    admin: UserRecord = Depends(require_role("admin")),
):
    """Admin-only — set one team member's token cap, carved out of the
    workspace's token_limit. Scoped to created_by = admin.user_id, same
    guard as auth.py's other per-member admin mutations — except the admin
    can also target their own user_id, to allocate themselves a slice of
    the same pool for their own budget discipline (MS-248 follow-up)."""
    if body.allocated_tokens < 0:
        raise HTTPException(status_code=400, detail="allocated_tokens must be >= 0")

    conn = _get_app_conn()
    if not conn:
        raise HTTPException(status_code=503, detail="Database unavailable")
    _ensure_tables(conn)
    _ensure_usage_tables(conn)
    try:
        if body.user_id == admin.user_id:
            member_row = {"user_id": admin.user_id, "email": admin.email}
        else:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT user_id, email FROM users WHERE user_id = %s AND created_by = %s",
                    (body.user_id, admin.user_id),
                )
                member_row = cur.fetchone()
            if not member_row:
                raise HTTPException(status_code=404, detail="Team member not found")

        window = _get_latest_plan_window(conn, admin.user_id)
        if not window or window.status != "active":
            raise HTTPException(status_code=400, detail="No active subscription to allocate tokens from")
        subscription = _build_subscription_usage(conn, admin.user_id, window=window)

        with conn.cursor() as cur:
            cur.execute(
                "SELECT user_id, allocated_tokens FROM token_allocations WHERE admin_user_id = %s",
                (admin.user_id,),
            )
            alloc_rows = cur.fetchall()
        allocations = {r["user_id"]: r["allocated_tokens"] for r in alloc_rows}
        already_allocated_elsewhere = sum(v for k, v in allocations.items() if k != body.user_id)
        if already_allocated_elsewhere + body.allocated_tokens > subscription.token_limit:
            raise HTTPException(status_code=400, detail="Allocation exceeds the workspace's token pool")

        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO token_allocations (admin_user_id, user_id, allocated_tokens)
                VALUES (%s, %s, %s)
                ON CONFLICT (user_id) DO UPDATE SET allocated_tokens = EXCLUDED.allocated_tokens
                """,
                (admin.user_id, body.user_id, body.allocated_tokens),
            )

        used = _sum_tokens_for_user(conn, body.user_id, window.period_start, window.period_end)
        unallocated = max(0, subscription.token_limit - already_allocated_elsewhere - body.allocated_tokens)
    finally:
        conn.close()

    member = MemberTokenUsage(
        user_id=body.user_id,
        email=member_row["email"],
        allocated_tokens=body.allocated_tokens,
        used_tokens=used,
        remaining_tokens=max(0, body.allocated_tokens - used),
        usage_percent=round(used / body.allocated_tokens * 100, 2) if body.allocated_tokens > 0 else 0.0,
    )
    return UpdateMemberAllocationResponse(member=member, unallocated_tokens=unallocated)


@router.get("/payments/rate-limit/me", response_model=RateLimitStatus)
async def get_my_rate_limit(user: UserRecord = Depends(get_current_user)):
    """Any authenticated user — lets the frontend pre-emptively disable the
    chat composer (and show a reset countdown) instead of only finding out
    they're blocked after a query already 429s."""
    conn = _get_app_conn()
    if not conn:
        raise HTTPException(status_code=503, detail="Database unavailable")
    _ensure_usage_tables(conn)
    try:
        return _get_rate_limit_status(conn, user.user_id)
    finally:
        conn.close()


@router.post("/payments/subscription/request-more", response_model=TokenRequestResponse)
async def request_more_tokens(
    body: CreateTokenRequestRequest,
    user: UserRecord = Depends(get_current_user),
):
    """Any authenticated user — ask their workspace admin for a bigger
    allocation. In-app only for now (the admin sees it next time they open
    Billing, via polling) — a real push-notification channel is a
    separate, larger follow-up. One pending request at a time per user;
    dismissing an old one (admin-side) frees them up to ask again."""
    conn = _get_app_conn()
    if not conn:
        raise HTTPException(status_code=503, detail="Database unavailable")
    _ensure_usage_tables(conn)
    try:
        admin_user_id = _resolve_admin_user_id(conn, user)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT request_id FROM token_requests WHERE user_id = %s AND status = 'pending'",
                (user.user_id,),
            )
            existing = cur.fetchone()
        if existing:
            raise HTTPException(
                status_code=400,
                detail="You already have a pending request — wait for your admin to respond to it first.",
            )

        request_id = f"treq_{uuid.uuid4().hex}"
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO token_requests (request_id, user_id, admin_user_id, message)
                VALUES (%s, %s, %s, %s)
                RETURNING created_at
                """,
                (request_id, user.user_id, admin_user_id, body.message),
            )
            created_at = cur.fetchone()["created_at"]
    finally:
        conn.close()

    return TokenRequestResponse(
        request=TokenRequestRecord(
            request_id=request_id,
            user_id=user.user_id,
            email=user.email,
            message=body.message,
            status="pending",
            created_at=app_db.ts(created_at),
        )
    )


@router.get("/payments/subscription/requests", response_model=TokenRequestsResponse)
async def list_token_requests(admin: UserRecord = Depends(require_role("admin"))):
    """Admin-only — pending (and recently resolved) token requests from
    their team, for the Billing tab and the sidebar's pending-count badge."""
    conn = _get_app_conn()
    if not conn:
        raise HTTPException(status_code=503, detail="Database unavailable")
    _ensure_usage_tables(conn)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT r.request_id, r.user_id, u.email, r.message, r.status, r.created_at
                FROM token_requests r
                JOIN users u ON u.user_id = r.user_id
                WHERE r.admin_user_id = %s
                ORDER BY (r.status = 'pending') DESC, r.created_at DESC
                LIMIT 50
                """,
                (admin.user_id,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    requests = [
        TokenRequestRecord(
            request_id=row["request_id"],
            user_id=row["user_id"],
            email=row["email"],
            message=row["message"],
            status=row["status"],
            created_at=app_db.ts(row["created_at"]),
        )
        for row in rows
    ]
    pending_count = sum(1 for r in requests if r.status == "pending")
    return TokenRequestsResponse(requests=requests, pending_count=pending_count)


@router.post("/payments/subscription/requests/{request_id}/dismiss", response_model=TokenRequestResponse)
async def dismiss_token_request(request_id: str, admin: UserRecord = Depends(require_role("admin"))):
    """Admin-only — mark a request as handled (whether or not they actually
    raised the member's allocation via the allocation editor elsewhere in
    the same Billing tab) so it stops showing as pending, and frees that
    member up to send a new request later if they need to."""
    conn = _get_app_conn()
    if not conn:
        raise HTTPException(status_code=503, detail="Database unavailable")
    _ensure_usage_tables(conn)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE token_requests SET status = 'resolved', resolved_at = now()
                WHERE request_id = %s AND admin_user_id = %s
                RETURNING user_id, message, created_at
                """,
                (request_id, admin.user_id),
            )
            row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Request not found")

        with conn.cursor() as cur:
            cur.execute("SELECT email FROM users WHERE user_id = %s", (row["user_id"],))
            user_row = cur.fetchone()
    finally:
        conn.close()

    return TokenRequestResponse(
        request=TokenRequestRecord(
            request_id=request_id,
            user_id=row["user_id"],
            email=user_row["email"] if user_row else "",
            message=row["message"],
            status="resolved",
            created_at=app_db.ts(row["created_at"]),
        )
    )
