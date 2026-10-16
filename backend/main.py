"""
=============================================================================
backend/main.py — FastAPI Application Entry Point
=============================================================================

This is the root of the FastAPI application. It:
    1. Defines the lifespan context manager (startup / shutdown hooks)
    2. Initialises the database (create tables)
    3. Loads all ML models at startup (avoids cold-start latency in production)
    4. Configures CORS (Cross-Origin Resource Sharing)
    5. Registers all API routers
    6. Mounts the frontend static files (serves HTML/CSS/JS)
    7. Adds a health check endpoint

Run with:
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
=============================================================================
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from backend.config import settings
from backend.database.connection import init_db
from backend.routers import auth, chat, export
from backend.services.pipeline import load_all_models

# ---------------------------------------------------------------------------
# Project root for resolving static files
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"


# ---------------------------------------------------------------------------
# Lifespan context manager (replaces deprecated @app.on_event)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Startup and shutdown logic.

    FastAPI's lifespan replaces @app.on_event("startup") / ("shutdown").
    Code before 'yield' runs on startup; after 'yield' runs on shutdown.

    Startup:
        1. Create database tables (idempotent)
        2. Load all ML models into memory
           (done ONCE here so every request reuses loaded models)

    Shutdown:
        - FastAPI handles graceful worker shutdown automatically
        - Any explicit cleanup (GPU memory, file handles) goes after yield
    """
    logger.info("=" * 60)
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Debug mode: {settings.debug}")
    logger.info("=" * 60)

    # Step 1: Initialise database
    logger.info("Initialising database...")
    await init_db()
    logger.info("Database ready.")

    # Step 2: Load all ML models
    await load_all_models()

    yield  # Application is running and serving requests

    # Shutdown
    logger.info("Shutting down MindBridge AI...")


# ---------------------------------------------------------------------------
# FastAPI Application Instance
# ---------------------------------------------------------------------------
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Mental Health Chatbot API powered by Deep Learning. "
        "Uses DistilBERT (emotion/safety), BiLSTM+Attention (intent), "
        "RoBERTa (sentiment), and FLAN-T5 (response generation)."
    ),
    docs_url="/api/docs",       # Swagger UI
    redoc_url="/api/redoc",     # ReDoc UI
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# CORS Middleware
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# API Routers
# ---------------------------------------------------------------------------
app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(export.router)


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------
@app.get("/api/health", tags=["System"])
async def health_check() -> dict:
    """
    Health check endpoint for monitoring / load balancers.

    Returns application status and version. Useful for:
        - Docker HEALTHCHECK directive
        - Kubernetes liveness probes
        - CI/CD deployment verification
    """
    return {
        "status": "healthy",
        "app": settings.app_name,
        "version": settings.app_version,
        "debug": settings.debug,
    }


# ---------------------------------------------------------------------------
# Static Frontend Files
# ---------------------------------------------------------------------------
# Mount frontend assets (css/, js/, assets/) as static files
if FRONTEND_DIR.exists():
    app.mount(
        "/static",
        StaticFiles(directory=str(FRONTEND_DIR)),
        name="static",
    )

    # Serve frontend HTML pages
    @app.get("/", include_in_schema=False)
    async def serve_index() -> FileResponse:
        """Serve the login/landing page."""
        return FileResponse(str(FRONTEND_DIR / "index.html"))

    @app.get("/register", include_in_schema=False)
    async def serve_register() -> FileResponse:
        return FileResponse(str(FRONTEND_DIR / "register.html"))

    @app.get("/chat", include_in_schema=False)
    async def serve_chat() -> FileResponse:
        return FileResponse(str(FRONTEND_DIR / "chat.html"))
else:
    logger.warning(f"Frontend directory not found at {FRONTEND_DIR}. Serving API only.")


# ---------------------------------------------------------------------------
# Entry point for direct execution (development)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info",
    )
