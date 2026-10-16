"""
=============================================================================
backend/schemas/chat.py — Pydantic Schemas for Chat & Conversation
=============================================================================
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Message schemas
# ---------------------------------------------------------------------------

class MessageIn(BaseModel):
    """Payload to send a new user message."""
    conversation_id: Optional[str] = Field(
        None,
        description="Existing conversation ID. If None, a new conversation is created."
    )
    content: str = Field(..., min_length=1, max_length=2000, description="User message text")


class EmotionResult(BaseModel):
    """Emotion detection result from DistilBERT classifier."""
    label: str                          # e.g. "sadness"
    confidence: float                   # 0.0 – 1.0
    top_emotions: List[Dict[str, Any]]  # [{"label": ..., "score": ...}, ...]


class IntentResult(BaseModel):
    """Intent classification result from BiLSTM classifier."""
    label: str                          # e.g. "seeking_support"
    confidence: float
    top_intents: List[Dict[str, Any]]


class SafetyResult(BaseModel):
    """Safety classifier output."""
    is_safe: bool
    category: str                       # "safe" | "suicidal_ideation" | ...
    confidence: float
    triggered: bool                     # True if threshold exceeded


class SentimentResult(BaseModel):
    """RoBERTa sentiment analysis output."""
    label: str                          # "positive" | "negative" | "neutral"
    score: float                        # 0.0 – 1.0 (confidence for that label)
    valence: float                      # normalised to −1.0 (negative) … +1.0 (positive)


class AnalysisResult(BaseModel):
    """Aggregated NLP analysis for a single user message."""
    emotion: EmotionResult
    intent: IntentResult
    safety: SafetyResult
    sentiment: SentimentResult


class MessageOut(BaseModel):
    """A single message (user or assistant) returned by the API."""
    id: str
    conversation_id: str
    role: str                           # "user" | "assistant"
    content: str
    created_at: datetime
    analysis: Optional[AnalysisResult] = None   # Populated for user messages
    metadata_json: Optional[Dict[str, Any]] = None

    model_config = {"from_attributes": True}


class ChatResponse(BaseModel):
    """Full chat turn response — user message + AI reply + analysis."""
    user_message: MessageOut
    assistant_message: MessageOut
    analysis: AnalysisResult
    conversation_id: str
    is_crisis: bool = False


# ---------------------------------------------------------------------------
# Conversation schemas
# ---------------------------------------------------------------------------

class ConversationSummary(BaseModel):
    """Brief conversation info for the sidebar list."""
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int
    avg_emotion: Optional[str] = None

    model_config = {"from_attributes": True}


class ConversationDetail(BaseModel):
    """Full conversation with all messages."""
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int
    messages: List[MessageOut]

    model_config = {"from_attributes": True}


class ConversationRenameRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)


# ---------------------------------------------------------------------------
# Statistics schema
# ---------------------------------------------------------------------------

class ConversationStats(BaseModel):
    """Session statistics displayed in the chat UI."""
    total_messages: int
    session_duration_seconds: int
    dominant_emotion: Optional[str]
    emotion_counts: Dict[str, int]
    avg_sentiment: float
    safety_events: int


# ---------------------------------------------------------------------------
# Emotion Timeline
# ---------------------------------------------------------------------------

class EmotionTimePoint(BaseModel):
    """Single data point for the mood timeline chart."""
    timestamp: datetime
    emotion: str
    confidence: float
    sentiment_score: float


class EmotionTimeline(BaseModel):
    """Ordered list of emotion readings for a conversation."""
    conversation_id: str
    points: List[EmotionTimePoint]
