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

import logging
import uuid
from typing import Optional

import stripe
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

import app_db
from config import config
from router.auth import get_current_user, UserRecord
from models import (
    CreateCheckoutSessionRequest,
    CheckoutSessionResponse,
    PaymentRecord,
    PaymentResponse,
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

_tables_ensured = False


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
