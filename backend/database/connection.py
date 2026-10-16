"""
=============================================================================
backend/database/connection.py — Async Database Connection & Session Factory
=============================================================================

Provides:
    engine          — SQLAlchemy async engine (aiosqlite)
    AsyncSessionLocal — async session factory
    get_db()        — FastAPI dependency that yields a session per request
    init_db()       — creates all tables on startup (idempotent)

Architecture Note:
    We use SQLAlchemy's async extension with aiosqlite.
    Each HTTP request gets its own session via the FastAPI dependency injection
    system. Sessions are committed on success and rolled back on exception,
    then always closed — preventing connection leaks.
=============================================================================
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool

from backend.config import settings
from backend.database.models import Base


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
# StaticPool is used for SQLite in-process to ensure all connections see
# the same in-memory/file state (avoids "database is locked" on async).
engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,          # Log SQL in debug mode
    connect_args={"check_same_thread": False},  # Required for SQLite
    poolclass=StaticPool,
)

# ---------------------------------------------------------------------------
# Session Factory
# ---------------------------------------------------------------------------
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,   # Don't expire objects after commit
    autocommit=False,
    autoflush=False,
)


# ---------------------------------------------------------------------------
# FastAPI Dependency
# ---------------------------------------------------------------------------
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency that provides a database session per request.

    Usage:
        @router.get("/items")
        async def get_items(db: AsyncSession = Depends(get_db)):
            ...

    The session is:
        1. Created fresh for each request
        2. Automatically committed if no exception
        3. Rolled back on any exception
        4. Closed in the finally block
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------
async def init_db() -> None:
    """
    Create all tables defined in models.py if they don't already exist.
    Called once at application startup via the lifespan context manager.

    Using `checkfirst=True` (the default for create_all) makes this
    idempotent — safe to call on every restart.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
