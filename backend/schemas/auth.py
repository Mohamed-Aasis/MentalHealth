"""
=============================================================================
backend/schemas/auth.py — Pydantic Schemas for Authentication
=============================================================================

These schemas define the shape of request/response bodies for auth endpoints.

Pydantic v2 validates all incoming data automatically — if a required field
is missing or has the wrong type, FastAPI returns a 422 Unprocessable Entity
before any business logic runs.
=============================================================================
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field, field_validator


# ---------------------------------------------------------------------------
# Request schemas (incoming from client)
# ---------------------------------------------------------------------------

class RegisterRequest(BaseModel):
    """User registration payload."""
    email: EmailStr = Field(..., description="Valid email address — used as login")
    username: str = Field(..., min_length=2, max_length=50, description="Display name")
    password: str = Field(..., min_length=8, max_length=100, description="Minimum 8 characters")

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        """Enforce basic password complexity."""
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v

    @field_validator("username")
    @classmethod
    def username_no_spaces(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Username cannot be empty")
        return v


class LoginRequest(BaseModel):
    """User login payload."""
    email: EmailStr
    password: str


class PasswordChangeRequest(BaseModel):
    """Change password payload."""
    current_password: str
    new_password: str = Field(..., min_length=8)


# ---------------------------------------------------------------------------
# Response schemas (outgoing to client)
# ---------------------------------------------------------------------------

class TokenResponse(BaseModel):
    """JWT token returned after successful login or registration."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds until expiry
    user: "UserPublic"


class UserPublic(BaseModel):
    """Public-safe user data (never includes password hash)."""
    id: str
    email: str
    username: str
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime] = None
    total_messages: int = 0

    model_config = {"from_attributes": True}


class MessageResponse(BaseModel):
    """Generic message response for simple API confirmations."""
    message: str
    success: bool = True
