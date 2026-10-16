"""
=============================================================================
backend/routers/auth.py — Authentication API Routes
=============================================================================

Endpoints:
    POST /api/auth/register  — Create new user account
    POST /api/auth/login     — Login and receive JWT token
    GET  /api/auth/me        — Get current user profile
    PUT  /api/auth/me        — Update username
    POST /api/auth/logout    — Invalidate session (client-side token deletion)
=============================================================================
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.connection import get_db
from backend.database.models import User
from backend.schemas.auth import (
    LoginRequest,
    MessageResponse,
    RegisterRequest,
    TokenResponse,
    UserPublic,
)
from backend.utils.auth_utils import (
    create_access_token,
    get_current_user,
    hash_password,
    verify_password,
)

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user",
)
async def register(
    payload: RegisterRequest,
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    """
    Create a new user account.

    Steps:
        1. Check if email already registered (unique constraint)
        2. Hash the password with bcrypt
        3. Create User record in database
        4. Issue JWT access token (user is auto-logged-in after registration)

    Returns:
        JWT token + public user data
    """
    # Check for existing email
    existing = await db.execute(select(User).where(User.email == payload.email))
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        )

    user = User(
        email=payload.email,
        username=payload.username,
        hashed_password=hash_password(payload.password),
    )
    db.add(user)
    await db.flush()  # Get the generated UUID without committing
    await db.refresh(user)

    access_token = create_access_token(user.id)
    logger.info(f"New user registered: {user.email}")

    return TokenResponse(
        access_token=access_token,
        expires_in=86400,  # 24 hours in seconds
        user=UserPublic.model_validate(user),
    )


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Login and receive access token",
)
async def login(
    payload: LoginRequest,
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    """
    Authenticate a user and return a JWT access token.

    Security:
        - Uses constant-time password comparison (bcrypt.verify)
        - Returns the SAME error message for "email not found" and
          "wrong password" to prevent user enumeration attacks
    """
    result = await db.execute(select(User).where(User.email == payload.email))
    user = result.scalar_one_or_none()

    # Deliberate vague error: prevents email enumeration
    if user is None or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is disabled.",
        )

    # Update last login timestamp
    user.last_login = datetime.now(timezone.utc)
    await db.flush()

    access_token = create_access_token(user.id)
    logger.info(f"User logged in: {user.email}")

    return TokenResponse(
        access_token=access_token,
        expires_in=86400,
        user=UserPublic.model_validate(user),
    )


@router.get(
    "/me",
    response_model=UserPublic,
    summary="Get current user profile",
)
async def get_me(
    current_user: User = Depends(get_current_user),
) -> UserPublic:
    """Return the authenticated user's public profile."""
    return UserPublic.model_validate(current_user)


@router.put(
    "/me",
    response_model=UserPublic,
    summary="Update username",
)
async def update_me(
    username: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> UserPublic:
    """Update the current user's display name."""
    current_user.username = username.strip()
    await db.flush()
    await db.refresh(current_user)
    return UserPublic.model_validate(current_user)


@router.post(
    "/logout",
    response_model=MessageResponse,
    summary="Logout (client-side token invalidation)",
)
async def logout(
    current_user: User = Depends(get_current_user),
) -> MessageResponse:
    """
    Logout endpoint.

    JWT tokens are stateless — the server cannot truly invalidate them.
    This endpoint exists for:
        1. Frontend UX (call to signify session end)
        2. Future implementation of a token blacklist (Redis-based)

    The client MUST delete the stored token from localStorage/cookies.
    """
    logger.info(f"User logged out: {current_user.email}")
    return MessageResponse(message="Logged out successfully. Please clear your session token.")
