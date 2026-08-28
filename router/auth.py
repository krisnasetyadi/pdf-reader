# router/auth.py
"""
Authentication & RBAC
---------------------
Endpoints:
  POST /auth/register   — create account (role defaults to "user")
  POST /auth/login      — returns JWT access token
  GET  /auth/me         — returns current user's full profile, refreshed from the DB
  POST /auth/me         — update own display name and/or avatar
  GET  /auth/admin/users           — admin-only: list team members this admin created
  POST /auth/admin/users           — admin-only: add a team member (role "user"), capped by max_sub_users
  POST /auth/admin/users/activate  — admin-only: activate/deactivate a team member this admin created

RBAC dependency helpers (importable by other routers):
  get_current_user(token)        → UserRecord (any authenticated user)
  require_role("admin")          → raises 403 if role doesn't match
  CurrentUser                    → FastAPI Depends shortcut
  AdminOnly                      → FastAPI Depends shortcut (admin only)

Schema (auto-created via ensure_schema in storage.py):
  users (user_id, email, password_hash, role, is_active, name, avatar_url, created_by, max_sub_users, created_at, updated_at)
  name           — display name, editable via POST /auth/me
  avatar_url     — small avatar image as a data: URL, editable via POST /auth/me
  created_by     — user_id of the admin who added this account (NULL for self-registered users)
  max_sub_users  — how many team members this user may add as an admin; set manually per admin for now
"""

from __future__ import annotations

import os
import time
import uuid
import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, field_validator

logger = logging.getLogger(__name__)
router = APIRouter()

# ---------------------------------------------------------------------------
# Config — read from env with sensible defaults
# ---------------------------------------------------------------------------

SECRET_KEY = os.getenv("JWT_SECRET")
if not SECRET_KEY:
    raise RuntimeError(
        "JWT_SECRET environment variable is not set. Refusing to start with a "
        "guessable default secret — set JWT_SECRET before running the server."
    )
ALGORITHM   = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "1440"))  # 24 h
PASSWORD_MAX_BYTES = 72  # bcrypt hard limit

_bearer = HTTPBearer(auto_error=False)

# ---------------------------------------------------------------------------
# Login rate limiting — in-memory, per (email, IP). Resets on process restart;
# fine for a single-instance deployment, not shared across multiple workers.
# ---------------------------------------------------------------------------

LOGIN_ATTEMPT_LIMIT = 5
LOGIN_ATTEMPT_WINDOW_SECONDS = 15 * 60  # 15 minutes

_failed_login_attempts: dict[str, list[float]] = defaultdict(list)

# Cached hash used to equalize verify() timing when no user row is found,
# so login doesn't leak "this email doesn't exist" via response latency.
_dummy_hash_cache: dict[str, str] = {}


def _rate_limit_key(email: str, request: Request) -> str:
    client_ip = request.client.host if request.client else "unknown"
    return f"{email.strip().lower()}:{client_ip}"


def _check_rate_limit(key: str) -> None:
    now = time.monotonic()
    # .get() instead of indexing the defaultdict directly — a plain read
    # must not auto-vivify a permanent entry for keys that never fail again
    # (e.g. an attacker probing many distinct emails), or this dict grows
    # for the life of the process with no cleanup path.
    attempts = [t for t in _failed_login_attempts.get(key, []) if now - t < LOGIN_ATTEMPT_WINDOW_SECONDS]
    if attempts:
        _failed_login_attempts[key] = attempts
    else:
        _failed_login_attempts.pop(key, None)
    if len(attempts) >= LOGIN_ATTEMPT_LIMIT:
        retry_minutes = LOGIN_ATTEMPT_WINDOW_SECONDS // 60
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Too many failed login attempts. Try again in {retry_minutes} minutes.",
        )


def _record_failed_login(key: str) -> None:
    _failed_login_attempts[key].append(time.monotonic())


def _clear_failed_logins(key: str) -> None:
    _failed_login_attempts.pop(key, None)


# ---------------------------------------------------------------------------
# Lazy imports so startup doesn't fail if packages missing
# ---------------------------------------------------------------------------

def _jose():
    from jose import jwt, JWTError  # noqa: F401
    return jwt

def _passlib():
    from passlib.context import CryptContext
    return CryptContext(schemes=["bcrypt"], deprecated="auto")

def _hash_password(pwd_ctx, password: str) -> str:
    """Hash password (length already validated to fit bcrypt's 72-byte limit)."""
    return pwd_ctx.hash(password)

def _dummy_password_hash(pwd_ctx) -> str:
    """A real bcrypt hash of a fixed, unused password — used only so failed
    logins for a non-existent email take the same time as a wrong-password
    login for a real one."""
    if "hash" not in _dummy_hash_cache:
        _dummy_hash_cache["hash"] = pwd_ctx.hash("dummy-password-for-timing-equalization")
    return _dummy_hash_cache["hash"]


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

def _validate_password_length(v: str) -> str:
    if len(v.encode("utf-8")) > PASSWORD_MAX_BYTES:
        raise ValueError(f"Password must be at most {PASSWORD_MAX_BYTES} bytes long")
    return v


NAME_MAX_LENGTH = 100


def _validate_name(v: Optional[str]) -> Optional[str]:
    if v is None:
        return v
    v = v.strip()
    if not v:
        raise ValueError("Name cannot be empty")
    if len(v) > NAME_MAX_LENGTH:
        raise ValueError(f"Name must be at most {NAME_MAX_LENGTH} characters")
    return v


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    role: str = "user"  # "user" | "admin"
    name: Optional[str] = None

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        if v not in ("user", "admin"):
            raise ValueError("role must be 'user' or 'admin'")
        return v

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        return _validate_password_length(v)

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: Optional[str]) -> Optional[str]:
        return _validate_name(v)

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    email: str
    role: str
    name: Optional[str] = None

class UserRecord(BaseModel):
    user_id: str
    email: str
    role: str
    is_active: bool
    name: Optional[str] = None
    avatar_url: Optional[str] = None

class UpdateProfileRequest(BaseModel):
    name: Optional[str] = None
    avatar_url: Optional[str] = None
    remove_avatar: bool = False  # explicit signal to clear avatar_url — None on avatar_url just means "leave as-is"

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: Optional[str]) -> Optional[str]:
        return _validate_name(v)

    @field_validator("avatar_url")
    @classmethod
    def validate_avatar_url(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        if not v.startswith("data:image/"):
            raise ValueError("Avatar must be an image data URL")
        if len(v) > 500_000:
            raise ValueError("Avatar image is too large")
        return v

class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str

    @field_validator("new_password")
    @classmethod
    def validate_new_password(cls, v: str) -> str:
        return _validate_password_length(v)

class AdminResetRequest(BaseModel):
    user_id: str
    new_password: str

    @field_validator("new_password")
    @classmethod
    def validate_new_password(cls, v: str) -> str:
        return _validate_password_length(v)


class AdminCreateUserRequest(BaseModel):
    email: EmailStr
    password: str

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        return _validate_password_length(v)


class AdminUserStatusRequest(BaseModel):
    user_id: str
    active: bool


class TeamMember(BaseModel):
    user_id: str
    email: str
    role: str
    is_active: bool
    created_at: datetime


class TeamMembersResponse(BaseModel):
    members: list[TeamMember]
    max_sub_users: int


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _get_conn():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        return None
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
        url = database_url
        if "sslmode=" not in url:
            sep = "&" if "?" in url else "?"
            url = url + sep + "sslmode=require"
        conn = psycopg2.connect(url, cursor_factory=RealDictCursor, connect_timeout=10)
        conn.autocommit = True
        return conn
    except Exception as e:
        logger.warning("auth: DB connection failed: %s", e)
        return None


def _ensure_users_table(conn):
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id            BIGSERIAL    PRIMARY KEY,
                    user_id       TEXT         NOT NULL UNIQUE DEFAULT gen_random_uuid()::text,
                    email         TEXT         NOT NULL UNIQUE,
                    password_hash TEXT         NOT NULL,
                    role          TEXT         NOT NULL DEFAULT 'user'
                                  CHECK (role IN ('user', 'admin')),
                    is_active     BOOLEAN      NOT NULL DEFAULT true,
                    created_at    TIMESTAMPTZ  NOT NULL DEFAULT now(),
                    updated_at    TIMESTAMPTZ  NOT NULL DEFAULT now()
                );
                ALTER TABLE users
                    ADD COLUMN IF NOT EXISTS created_by TEXT;
                ALTER TABLE users
                    ADD COLUMN IF NOT EXISTS max_sub_users INTEGER NOT NULL DEFAULT 5;
                ALTER TABLE users
                    ADD COLUMN IF NOT EXISTS name TEXT;
                ALTER TABLE users
                    ADD COLUMN IF NOT EXISTS avatar_url TEXT;
                CREATE INDEX IF NOT EXISTS idx_users_email      ON users (email);
                CREATE INDEX IF NOT EXISTS idx_users_user_id    ON users (user_id);
                CREATE INDEX IF NOT EXISTS idx_users_created_by ON users (created_by);
            """)
    except Exception as e:
        logger.warning("auth: ensure users table failed: %s", e)


# ---------------------------------------------------------------------------
# JWT helpers
# ---------------------------------------------------------------------------

def _create_token(user_id: str, email: str, role: str, name: Optional[str] = None) -> str:
    jwt = _jose()
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": user_id, "email": email, "role": role, "exp": expire}
    if name:
        payload["name"] = name
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def _decode_token(token: str) -> dict:
    from jose import JWTError
    jwt = _jose()
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e


# ---------------------------------------------------------------------------
# RBAC dependency helpers (use these in other routers)
# ---------------------------------------------------------------------------

def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> UserRecord:
    """Require a valid JWT. Returns the decoded user record."""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    payload = _decode_token(credentials.credentials)
    return UserRecord(
        user_id=payload["sub"],
        email=payload["email"],
        role=payload["role"],
        is_active=True,
    )


def require_role(*roles: str):
    """
    Factory for role-gated dependencies.

    Usage:
        @router.get("/admin-only")
        def admin_route(user = Depends(require_role("admin"))):
            ...
    """
    def _dep(user: UserRecord = Depends(get_current_user)) -> UserRecord:
        if user.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Required role: {', '.join(roles)}",
            )
        return user
    return _dep


# Convenience aliases
CurrentUser = Depends(get_current_user)
AdminOnly   = Depends(require_role("admin"))


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/auth/register", response_model=TokenResponse, status_code=201)
async def register(body: RegisterRequest):
    """Register a new user. First registered user becomes admin automatically."""
    pwd_ctx = _passlib()
    conn = _get_conn()

    if not conn:
        raise HTTPException(status_code=503, detail="Database unavailable")

    try:
        _ensure_users_table(conn)
        with conn.cursor() as cur:
            # Check email already exists
            cur.execute("SELECT user_id FROM users WHERE email = %s", (body.email,))
            if cur.fetchone():
                raise HTTPException(status_code=409, detail="Email already registered")

            # First user in DB gets admin role regardless of request
            cur.execute("SELECT COUNT(*) AS cnt FROM users")
            count = cur.fetchone()["cnt"]
            role = "admin" if count == 0 else body.role

            user_id = str(uuid.uuid4())
            hashed  = _hash_password(pwd_ctx, body.password)

            cur.execute("""
                INSERT INTO users (user_id, email, password_hash, role, name)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING user_id, email, role, name
            """, (user_id, body.email, hashed, role, body.name))
            row = cur.fetchone()
    except HTTPException:
        raise
    except Exception as e:
        logger.error("auth register error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Registration failed: {e}")
    finally:
        conn.close()

    token = _create_token(row["user_id"], row["email"], row["role"], row["name"])
    return TokenResponse(
        access_token=token,
        user_id=row["user_id"],
        email=row["email"],
        role=row["role"],
        name=row["name"],
    )


@router.post("/auth/login", response_model=TokenResponse)
async def login(body: LoginRequest, request: Request):
    """Login and receive a JWT access token."""
    rl_key = _rate_limit_key(body.email, request)
    _check_rate_limit(rl_key)

    pwd_ctx = _passlib()
    conn = _get_conn()

    if not conn:
        raise HTTPException(status_code=503, detail="Database unavailable")

    try:
        _ensure_users_table(conn)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT user_id, email, password_hash, role, is_active, name FROM users WHERE email = %s",
                (body.email,)
            )
            row = cur.fetchone()
    except Exception as e:
        logger.error("auth login db error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Login failed: {e}")
    finally:
        conn.close()

    # Always run a bcrypt verify, even for an unknown email, so response time
    # doesn't reveal whether the address is registered.
    password_hash = row["password_hash"] if row else _dummy_password_hash(pwd_ctx)
    password_ok = pwd_ctx.verify(body.password, password_hash)

    if not row or not password_ok:
        _record_failed_login(rl_key)
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if not row["is_active"]:
        raise HTTPException(status_code=403, detail="Account is disabled")

    _clear_failed_logins(rl_key)
    token = _create_token(row["user_id"], row["email"], row["role"], row["name"])
    return TokenResponse(
        access_token=token,
        user_id=row["user_id"],
        email=row["email"],
        role=row["role"],
        name=row["name"],
    )


@router.get("/auth/me", response_model=UserRecord)
async def me(user: UserRecord = Depends(get_current_user)):
    """Return the current user's full profile, refreshed from the DB — the
    JWT alone doesn't carry avatar_url and its is_active claim can go stale."""
    conn = _get_conn()
    if not conn:
        return user
    try:
        _ensure_users_table(conn)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT user_id, email, role, is_active, name, avatar_url FROM users WHERE user_id = %s",
                (user.user_id,),
            )
            row = cur.fetchone()
        return UserRecord(**row) if row else user
    finally:
        conn.close()


@router.post("/auth/me", response_model=UserRecord)
async def update_profile(
    body: UpdateProfileRequest,
    user: UserRecord = Depends(get_current_user),
):
    """Update the current user's display name and/or avatar. Pass
    remove_avatar=true to clear the avatar back to the initials fallback."""
    if body.name is None and body.avatar_url is None and not body.remove_avatar:
        raise HTTPException(status_code=400, detail="Nothing to update")
    conn = _get_conn()
    if not conn:
        raise HTTPException(status_code=503, detail="Database unavailable")
    try:
        _ensure_users_table(conn)
        set_clauses = []
        params = []
        if body.name is not None:
            set_clauses.append("name = %s")
            params.append(body.name)
        if body.remove_avatar:
            set_clauses.append("avatar_url = NULL")
        elif body.avatar_url is not None:
            set_clauses.append("avatar_url = %s")
            params.append(body.avatar_url)
        set_clauses.append("updated_at = now()")
        params.append(user.user_id)
        with conn.cursor() as cur:
            cur.execute(
                f"UPDATE users SET {', '.join(set_clauses)} WHERE user_id = %s "
                f"RETURNING user_id, email, role, is_active, name, avatar_url",
                params,
            )
            row = cur.fetchone()
        return UserRecord(**row)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("update_profile error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to update profile: {e}")
    finally:
        conn.close()


@router.post("/auth/change-password", status_code=200)
async def change_password(
    body: ChangePasswordRequest,
    user: UserRecord = Depends(get_current_user),
):
    """Change the current user's password (requires valid token)."""
    pwd_ctx = _passlib()
    conn = _get_conn()
    if not conn:
        raise HTTPException(status_code=503, detail="Database unavailable")
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT password_hash FROM users WHERE user_id = %s",
                (user.user_id,)
            )
            row = cur.fetchone()
            if not row or not pwd_ctx.verify(body.current_password, row["password_hash"]):
                raise HTTPException(status_code=401, detail="Current password is incorrect")
            new_hash = _hash_password(pwd_ctx, body.new_password)
            cur.execute(
                "UPDATE users SET password_hash = %s, updated_at = now() WHERE user_id = %s",
                (new_hash, user.user_id)
            )
        return {"message": "Password updated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("change_password error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to change password: {e}")
    finally:
        conn.close()


@router.post("/auth/admin/reset-password", status_code=200)
async def admin_reset_password(
    body: AdminResetRequest,
    admin: UserRecord = Depends(require_role("admin")),
):
    """Admin-only: reset a team member's password. Scoped to created_by =
    admin.user_id — an admin can only reset passwords for members they
    created themselves, not any arbitrary user in the system."""
    pwd_ctx = _passlib()
    conn = _get_conn()
    if not conn:
        raise HTTPException(status_code=503, detail="Database unavailable")
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT user_id, email FROM users WHERE user_id = %s AND created_by = %s",
                (body.user_id, admin.user_id),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Team member not found")
            new_hash = _hash_password(pwd_ctx, body.new_password)
            cur.execute(
                "UPDATE users SET password_hash = %s, updated_at = now() WHERE user_id = %s",
                (new_hash, body.user_id)
            )
        return {"message": f"Password reset for {row['email']}"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("admin_reset_password error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to reset password: {e}")
    finally:
        conn.close()


@router.get("/auth/admin/users", response_model=TeamMembersResponse)
async def list_admin_users(admin: UserRecord = Depends(require_role("admin"))):
    """Admin-only: list the team members this admin has added, plus their quota."""
    conn = _get_conn()
    if not conn:
        raise HTTPException(status_code=503, detail="Database unavailable")
    try:
        _ensure_users_table(conn)
        with conn.cursor() as cur:
            cur.execute("SELECT max_sub_users FROM users WHERE user_id = %s", (admin.user_id,))
            admin_row = cur.fetchone()
            max_sub_users = admin_row["max_sub_users"] if admin_row else 0

            cur.execute(
                """
                SELECT user_id, email, role, is_active, created_at
                FROM users WHERE created_by = %s ORDER BY created_at DESC
                """,
                (admin.user_id,),
            )
            members = cur.fetchall()
        return TeamMembersResponse(members=members, max_sub_users=max_sub_users)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("list_admin_users error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list team members: {e}")
    finally:
        conn.close()


@router.post("/auth/admin/users", response_model=TeamMember, status_code=201)
async def add_admin_user(
    body: AdminCreateUserRequest,
    admin: UserRecord = Depends(require_role("admin")),
):
    """Admin-only: add a team member under this admin, capped by max_sub_users. New members always get role 'user'."""
    pwd_ctx = _passlib()
    conn = _get_conn()
    if not conn:
        raise HTTPException(status_code=503, detail="Database unavailable")
    try:
        _ensure_users_table(conn)
        with conn.cursor() as cur:
            cur.execute("SELECT user_id FROM users WHERE email = %s", (body.email,))
            if cur.fetchone():
                raise HTTPException(status_code=409, detail="Email already registered")

            cur.execute("SELECT max_sub_users FROM users WHERE user_id = %s", (admin.user_id,))
            admin_row = cur.fetchone()
            max_sub_users = admin_row["max_sub_users"] if admin_row else 0

            cur.execute(
                "SELECT COUNT(*) AS cnt FROM users WHERE created_by = %s AND is_active = true",
                (admin.user_id,),
            )
            current_count = cur.fetchone()["cnt"]
            if current_count >= max_sub_users:
                raise HTTPException(status_code=403, detail="Team member limit reached")

            user_id = str(uuid.uuid4())
            hashed = _hash_password(pwd_ctx, body.password)
            cur.execute(
                """
                INSERT INTO users (user_id, email, password_hash, role, created_by)
                VALUES (%s, %s, %s, 'user', %s)
                RETURNING user_id, email, role, is_active, created_at
                """,
                (user_id, body.email, hashed, admin.user_id),
            )
            row = cur.fetchone()
        return TeamMember(**row)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("add_admin_user error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to add team member: {e}")
    finally:
        conn.close()


@router.post("/auth/admin/users/activate")
async def set_admin_user_status(
    body: AdminUserStatusRequest,
    admin: UserRecord = Depends(require_role("admin")),
):
    """Admin-only: activate/deactivate a team member this admin created.
    Scoped to created_by = admin.user_id, so an admin can never touch a
    member outside their own team (and can never deactivate themselves,
    since their own row has no created_by pointing at themselves)."""
    conn = _get_conn()
    if not conn:
        raise HTTPException(status_code=503, detail="Database unavailable")
    try:
        _ensure_users_table(conn)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT user_id FROM users WHERE user_id = %s AND created_by = %s",
                (body.user_id, admin.user_id),
            )
            if not cur.fetchone():
                raise HTTPException(status_code=404, detail="Team member not found")

            cur.execute(
                "UPDATE users SET is_active = %s, updated_at = now() WHERE user_id = %s",
                (body.active, body.user_id),
            )
        return {"status": "success", "user_id": body.user_id, "active": body.active}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("set_admin_user_status error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to update team member: {e}")
    finally:
        conn.close()

