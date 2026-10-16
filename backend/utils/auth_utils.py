"""
=============================================================================
backend/utils/auth_utils.py — Authentication Utilities
=============================================================================

Provides:
    hash_password()     — bcrypt password hashing
    verify_password()   — constant-time password verification
    create_access_token() — JWT generation
    decode_access_token() — JWT decoding + validation
    get_current_user()  — FastAPI dependency that extracts the current user
                          from the Authorization: Bearer <token> header

Security Notes:
    - bcrypt automatically salts passwords — no need to manage salt manually
    - JWT tokens are signed with HS256 (HMAC-SHA256)
    - Token expiry is enforced by python-jose during decode
    - Passwords are NEVER stored in plaintext anywhere
=============================================================================
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import settings
from backend.database.connection import get_db
from backend.database.models import User

# ---------------------------------------------------------------------------
# Password hashing context
# ---------------------------------------------------------------------------
# CryptContext manages the hashing algorithm. Using bcrypt with 12 rounds
# (the default) — each verification takes ~100–400 ms, which is intentionally
# slow to resist brute-force attacks.
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ---------------------------------------------------------------------------
# OAuth2 scheme — tells FastAPI where to find the token
# ---------------------------------------------------------------------------
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


def hash_password(plain_password: str) -> str:
    """
    Hash a plaintext password using bcrypt.

    bcrypt:
        1. Generates a random salt (22 characters)
        2. Runs the password + salt through the Blowfish cipher
        3. Returns a 60-character string containing algorithm + cost + salt + hash

    The resulting hash is deterministic per (password, salt) pair but
    different salts produce different hashes for the same password.
    """
    return pwd_context.hash(plain_password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plaintext password against a bcrypt hash.

    Uses constant-time comparison internally (via passlib) to prevent
    timing attacks where an attacker could guess characters by measuring
    how long the comparison takes.
    """
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(
    subject: str,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Create a signed JWT access token.

    JWT Structure:
        Header:  {"alg": "HS256", "typ": "JWT"}
        Payload: {"sub": user_id, "exp": expiry_timestamp, "iat": issued_at}
        Signature: HMAC-SHA256(base64(header) + "." + base64(payload), SECRET_KEY)

    The token is NOT encrypted — the payload is base64-encoded and visible.
    Never put sensitive data (passwords, PII) in JWT claims.
    Security comes from the signature, which cannot be forged without SECRET_KEY.

    Args:
        subject: The user ID (primary key from database)
        expires_delta: Custom expiry duration; defaults to settings value

    Returns:
        Signed JWT string
    """
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=settings.access_token_expire_minutes)
    )
    payload = {
        "sub": subject,
        "exp": expire,
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)


def decode_access_token(token: str) -> Optional[str]:
    """
    Decode and validate a JWT token.

    Returns the subject (user ID) if valid, None if invalid/expired.

    python-jose automatically:
        - Verifies the signature using SECRET_KEY
        - Checks the "exp" claim (expiry)
        - Raises JWTError for tampered or expired tokens
    """
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
        return payload.get("sub")
    except JWTError:
        return None


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    """
    FastAPI dependency — extracts and validates the current user.

    Flow:
        1. FastAPI reads Bearer token from Authorization header
        2. Token is decoded → user_id extracted
        3. User is loaded from database
        4. Active check performed
        5. User object returned (or 401 raised)

    Usage:
        @router.get("/protected")
        async def protected(current_user: User = Depends(get_current_user)):
            ...
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    user_id = decode_access_token(token)
    if user_id is None:
        raise credentials_exception

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if user is None or not user.is_active:
        raise credentials_exception

    return user
