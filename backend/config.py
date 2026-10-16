"""
=============================================================================
backend/config.py — Application Configuration
=============================================================================

Centralised configuration using Pydantic Settings.
All values are read from environment variables (or .env file).
This is the single source of truth for all configurable parameters.

Architecture Note:
    Using pydantic-settings ensures type-safe, validated configuration
    at startup. Any misconfiguration (wrong type, missing required value)
    raises a clear error immediately rather than failing deep in inference.
=============================================================================
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root is two levels up from this file: backend/config.py → backend/ → project root
PROJECT_ROOT = Path(__file__).parent.parent


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables / .env file.

    Every attribute maps directly to a variable in .env.example.
    Pydantic automatically casts types (e.g., "false" → False).
    """

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -------------------------------------------------------------------------
    # Application
    # -------------------------------------------------------------------------
    app_name: str = "MindBridge AI"
    app_version: str = "1.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000

    # -------------------------------------------------------------------------
    # Security
    # -------------------------------------------------------------------------
    secret_key: str = "dev-secret-key-please-change-in-production-32-chars-min"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 1440  # 24 hours

    # -------------------------------------------------------------------------
    # Database
    # -------------------------------------------------------------------------
    database_url: str = "sqlite+aiosqlite:///./mindbridge.db"

    # -------------------------------------------------------------------------
    # Model Paths — resolved relative to project root
    # -------------------------------------------------------------------------
    emotion_model_path: str = "models/emotion"
    intent_model_path: str = "models/intent"
    response_model_path: str = "models/response"
    safety_model_path: str = "models/safety"

    # -------------------------------------------------------------------------
    # Model Configuration
    # -------------------------------------------------------------------------
    use_lite_mode: bool = False
    max_context_turns: int = 6
    max_new_tokens: int = 600
    generation_temperature: float = 0.8
    generation_top_p: float = 0.92

    # -------------------------------------------------------------------------
    # Ollama LLM Configuration
    # -------------------------------------------------------------------------
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "mindbridge"

    # -------------------------------------------------------------------------
    # Safety
    # -------------------------------------------------------------------------
    safety_threshold: float = 0.65
    crisis_hotline_number: str = "988"
    crisis_hotline_name: str = "Suicide & Crisis Lifeline"
    crisis_hotline_url: str = "https://988lifeline.org"

    # -------------------------------------------------------------------------
    # WandB (optional)
    # -------------------------------------------------------------------------
    wandb_api_key: str = ""
    wandb_project: str = "mindbridge-training"

    # -------------------------------------------------------------------------
    # CORS
    # -------------------------------------------------------------------------
    cors_origins: str = "http://localhost:3000,http://localhost:8000,http://127.0.0.1:8000"

    # =========================================================================
    # Computed Properties
    # =========================================================================

    @property
    def cors_origins_list(self) -> List[str]:
        """Return CORS origins as a list."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def emotion_model_dir(self) -> Path:
        return PROJECT_ROOT / self.emotion_model_path

    @property
    def intent_model_dir(self) -> Path:
        return PROJECT_ROOT / self.intent_model_path

    @property
    def response_model_dir(self) -> Path:
        return PROJECT_ROOT / self.response_model_path

    @property
    def safety_model_dir(self) -> Path:
        return PROJECT_ROOT / self.safety_model_path

    # =========================================================================
    # HuggingFace Hub Model IDs
    # =========================================================================
    # These are the BASE pre-trained models used when fine-tuned checkpoints
    # are not yet available locally (i.e., before training).
    # After running training/train_*.py scripts the local directories will
    # take precedence over these hub IDs.

    # Emotion classifier base
    EMOTION_BASE_MODEL: str = "distilbert-base-uncased"
    # Number of emotion classes (GoEmotions has 28)
    EMOTION_NUM_LABELS: int = 28
    # GoEmotions label list (canonical order)
    EMOTION_LABELS: List[str] = [
        "admiration", "amusement", "anger", "annoyance", "approval",
        "caring", "confusion", "curiosity", "desire", "disappointment",
        "disapproval", "disgust", "embarrassment", "excitement", "fear",
        "gratitude", "grief", "joy", "love", "nervousness",
        "optimism", "pride", "realization", "relief", "remorse",
        "sadness", "surprise", "neutral",
    ]

    # Intent classifier
    INTENT_BASE_MODEL: str = "distilbert-base-uncased"
    INTENT_LABELS: List[str] = [
        "seeking_support", "expressing_distress", "asking_for_advice",
        "sharing_experience", "greeting", "farewell", "gratitude",
        "confusion", "crisis_signal", "small_talk",
    ]
    INTENT_NUM_LABELS: int = 10

    # Response generator
    RESPONSE_BASE_MODEL: str = "google/flan-t5-small"
    RESPONSE_LITE_MODEL: str = "microsoft/DialoGPT-small"

    # Safety classifier
    SAFETY_BASE_MODEL: str = "distilbert-base-uncased"
    SAFETY_LABELS: List[str] = [
        "safe", "suicidal_ideation", "self_harm", "violence", "abuse",
    ]
    SAFETY_NUM_LABELS: int = 5

    # Sentiment model (zero-shot, no fine-tuning needed)
    SENTIMENT_MODEL: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"

    # =========================================================================
    # Training Hyperparameters (used by training scripts)
    # =========================================================================
    TRAIN_BATCH_SIZE: int = 16
    EVAL_BATCH_SIZE: int = 32
    LEARNING_RATE: float = 2e-5
    WEIGHT_DECAY: float = 0.01
    WARMUP_RATIO: float = 0.1
    MAX_SEQ_LENGTH: int = 128
    NUM_EPOCHS_CLASSIFIER: int = 5
    NUM_EPOCHS_GENERATOR: int = 3
    GRADIENT_CLIP_NORM: float = 1.0
    DROPOUT_RATE: float = 0.3
    EARLY_STOPPING_PATIENCE: int = 2

    # =========================================================================
    # Intent BiLSTM Hyperparameters
    # =========================================================================
    INTENT_VOCAB_SIZE: int = 30000
    INTENT_EMBED_DIM: int = 256
    INTENT_HIDDEN_DIM: int = 512
    INTENT_NUM_LAYERS: int = 2
    INTENT_MAX_SEQ_LEN: int = 64


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the singleton Settings instance.
    Using @lru_cache ensures this is only created once per process,
    which is important because loading settings reads disk I/O.
    """
    return Settings()


# Convenience alias used throughout the backend
settings = get_settings()
