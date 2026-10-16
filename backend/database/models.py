"""
=============================================================================
backend/database/models.py — SQLAlchemy ORM Models
=============================================================================

Defines every database table used by the application.

Tables:
    User            — registered users (hashed passwords, metadata)
    Conversation    — a named chat session belonging to a user
    Message         — individual turns within a conversation
    EmotionLog      — timestamped emotion/intent/sentiment readings
                      (used for mood timeline visualisation)

Architecture Note:
    We use SQLAlchemy 2.0 with the modern "mapped_column" API (DeclarativeBase).
    The async driver (aiosqlite) allows FastAPI's async request handlers to query
    the database without blocking the event loop — critical for an app that also
    runs heavy ML inference in a thread pool.
=============================================================================
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    JSON,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_uuid() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Base class — all models inherit from this
# ---------------------------------------------------------------------------
class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# User model
# ---------------------------------------------------------------------------
class User(Base):
    """
    Represents a registered user.

    Columns:
        id            — UUID primary key
        email         — unique login identifier
        username      — display name
        hashed_password — bcrypt hash (never store plaintext!)
        is_active     — soft delete / account disable flag
        created_at    — registration timestamp
        last_login    — most recent successful login
        total_messages — denormalised counter for quick stats
    """
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    username: Mapped[str] = mapped_column(String(100), nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    last_login: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    total_messages: Mapped[int] = mapped_column(Integer, default=0)

    # Relationships
    conversations: Mapped[List["Conversation"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


# ---------------------------------------------------------------------------
# Conversation model
# ---------------------------------------------------------------------------
class Conversation(Base):
    """
    A named chat session.

    Each user can have multiple conversations.
    A conversation has many messages.

    Columns:
        id          — UUID primary key
        user_id     — FK to User
        title       — auto-generated or user-renamed title
        created_at  — session start time
        updated_at  — last message time (for sorting)
        is_deleted  — soft delete (allow recovery)
        message_count — denormalised counter
        avg_emotion — predominant emotion across session (for summary)
    """
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(255), default="New Conversation")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)
    is_deleted: Mapped[bool] = mapped_column(Boolean, default=False)
    message_count: Mapped[int] = mapped_column(Integer, default=0)
    avg_emotion: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    # Relationships
    user: Mapped["User"] = relationship(back_populates="conversations")
    messages: Mapped[List["Message"]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan", order_by="Message.created_at"
    )
    emotion_logs: Mapped[List["EmotionLog"]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan"
    )


# ---------------------------------------------------------------------------
# Message model
# ---------------------------------------------------------------------------
class Message(Base):
    """
    A single message turn in a conversation.

    Stores both the user input and the AI response as separate rows
    (role = "user" | "assistant") — consistent with OpenAI message format
    for easy future migration.

    Columns:
        role            — "user" or "assistant"
        content         — raw text content
        created_at      — timestamp
        emotion         — detected emotion (user messages only)
        emotion_conf    — emotion confidence score (0–1)
        intent          — detected intent (user messages only)
        intent_conf     — intent confidence score (0–1)
        sentiment_score — valence score (−1 to +1)
        safety_flag     — True if safety classifier triggered
        safety_category — which safety category triggered
        metadata_json   — flexible JSON store for future fields
    """
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    conversation_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("conversations.id"), nullable=False, index=True
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)  # "user" | "assistant"
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    # Analysis fields (populated for user messages)
    emotion: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    emotion_conf: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    intent: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    intent_conf: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sentiment_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    safety_flag: Mapped[bool] = mapped_column(Boolean, default=False)
    safety_category: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    metadata_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # Relationships
    conversation: Mapped["Conversation"] = relationship(back_populates="messages")


# ---------------------------------------------------------------------------
# EmotionLog model — for timeline/chart visualisation
# ---------------------------------------------------------------------------
class EmotionLog(Base):
    """
    Timestamped emotion reading for mood visualisation.

    Separate from Message to allow efficient time-series queries
    without loading full message content.

    Used by the frontend emotion timeline chart.
    """
    __tablename__ = "emotion_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("conversations.id"), nullable=False, index=True
    )
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    emotion: Mapped[str] = mapped_column(String(50), nullable=False)
    emotion_conf: Mapped[float] = mapped_column(Float, nullable=False)
    sentiment_score: Mapped[float] = mapped_column(Float, default=0.0)

    # Relationships
    conversation: Mapped["Conversation"] = relationship(back_populates="emotion_logs")
