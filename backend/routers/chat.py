"""
=============================================================================
backend/routers/chat.py — Chat & Conversation API Routes
=============================================================================

Endpoints:
    POST   /api/chat/send              — Send a message, get AI response
    GET    /api/chat/conversations     — List user's conversations
    GET    /api/chat/conversations/{id} — Get conversation with messages
    DELETE /api/chat/conversations/{id} — Soft-delete a conversation
    PUT    /api/chat/conversations/{id}/rename — Rename conversation
    GET    /api/chat/conversations/{id}/timeline — Emotion timeline
    GET    /api/chat/conversations/{id}/stats    — Session statistics
    DELETE /api/chat/all                — Delete all user conversations
=============================================================================
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from loguru import logger
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.database.connection import get_db
from backend.database.models import Conversation, EmotionLog, Message, User
from backend.schemas.auth import MessageResponse
from backend.schemas.chat import (
    ChatResponse,
    ConversationDetail,
    ConversationRenameRequest,
    ConversationStats,
    ConversationSummary,
    EmotionTimeline,
    EmotionTimePoint,
    MessageIn,
    MessageOut,
)
from backend.services.pipeline import NLPPipeline
from backend.utils.auth_utils import get_current_user
from backend.utils.text_utils import extract_conversation_title

router = APIRouter(prefix="/api/chat", tags=["Chat"])

# Module-level pipeline instance (reuses model singletons)
_pipeline = NLPPipeline()


# ---------------------------------------------------------------------------
# Send Message
# ---------------------------------------------------------------------------

@router.post(
    "/send",
    response_model=ChatResponse,
    summary="Send a user message and receive AI response",
)
async def send_message(
    payload: MessageIn,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ChatResponse:
    """
    Core chat endpoint.

    Flow:
        1. Resolve or create conversation
        2. Load existing history from DB
        3. Run full NLP pipeline
        4. Save user message + AI response to DB
        5. Update conversation metadata
        6. Log emotion to timeline
        7. Return structured response

    This is the most complex endpoint — it orchestrates database access
    and ML inference in a single request.
    """
    # ---- 1. Resolve conversation ----
    conversation: Optional[Conversation] = None

    if payload.conversation_id:
        result = await db.execute(
            select(Conversation).where(
                Conversation.id == payload.conversation_id,
                Conversation.user_id == current_user.id,
                Conversation.is_deleted == False,
            )
        )
        conversation = result.scalar_one_or_none()
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found.")
    else:
        # Create a new conversation with auto-generated title
        title = extract_conversation_title(payload.content)
        conversation = Conversation(
            user_id=current_user.id,
            title=title,
        )
        db.add(conversation)
        await db.flush()

    # ---- 2. Load existing history ----
    history_result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation.id)
        .order_by(Message.created_at)
    )
    existing_messages = history_result.scalars().all()
    history_dicts = [{"role": m.role, "content": m.content} for m in existing_messages]

    # ---- 3. Run NLP pipeline ----
    logger.info(f"Processing message for user {current_user.id}, conv {conversation.id}")
    pipeline_result = await _pipeline.process(
        user_text=payload.content,
        conversation_history=history_dicts,
    )

    analysis = pipeline_result["analysis"]
    response_text = pipeline_result["response"]
    is_crisis = pipeline_result["is_crisis"]

    # ---- 4. Save messages to database ----
    user_message = Message(
        conversation_id=conversation.id,
        role="user",
        content=payload.content,
        emotion=analysis.emotion.label,
        emotion_conf=analysis.emotion.confidence,
        intent=analysis.intent.label,
        intent_conf=analysis.intent.confidence,
        sentiment_score=analysis.sentiment.valence,
        safety_flag=analysis.safety.triggered,
        safety_category=analysis.safety.category if analysis.safety.triggered else None,
        metadata_json={
            "top_emotions": analysis.emotion.top_emotions,
            "top_intents": analysis.intent.top_intents,
        },
    )
    db.add(user_message)

    assistant_message = Message(
        conversation_id=conversation.id,
        role="assistant",
        content=response_text,
        metadata_json={"is_crisis_response": is_crisis},
    )
    db.add(assistant_message)

    # ---- 5. Update conversation metadata ----
    conversation.message_count += 2
    conversation.updated_at = datetime.now(timezone.utc)
    conversation.avg_emotion = analysis.emotion.label

    # ---- 6. Update user message counter ----
    current_user.total_messages += 1

    # ---- 7. Log emotion to timeline ----
    emotion_log = EmotionLog(
        conversation_id=conversation.id,
        emotion=analysis.emotion.label,
        emotion_conf=analysis.emotion.confidence,
        sentiment_score=analysis.sentiment.valence,
    )
    db.add(emotion_log)

    await db.flush()
    await db.refresh(user_message)
    await db.refresh(assistant_message)

    # ---- 8. Build response ----
    from backend.schemas.chat import AnalysisResult

    user_msg_out = MessageOut(
        id=user_message.id,
        conversation_id=conversation.id,
        role="user",
        content=payload.content,
        created_at=user_message.created_at,
        analysis=analysis,
        metadata_json=user_message.metadata_json,
    )

    assistant_msg_out = MessageOut(
        id=assistant_message.id,
        conversation_id=conversation.id,
        role="assistant",
        content=response_text,
        created_at=assistant_message.created_at,
    )

    return ChatResponse(
        user_message=user_msg_out,
        assistant_message=assistant_msg_out,
        analysis=analysis,
        conversation_id=conversation.id,
        is_crisis=is_crisis,
    )


# ---------------------------------------------------------------------------
# Conversation Management
# ---------------------------------------------------------------------------

@router.get(
    "/conversations",
    response_model=List[ConversationSummary],
    summary="List user conversations",
)
async def list_conversations(
    search: Optional[str] = Query(None, description="Search conversation titles"),
    limit: int = Query(50, le=100),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> List[ConversationSummary]:
    """Return all non-deleted conversations for the current user, newest first."""
    query = (
        select(Conversation)
        .where(
            Conversation.user_id == current_user.id,
            Conversation.is_deleted == False,
        )
        .order_by(Conversation.updated_at.desc())
        .limit(limit)
        .offset(offset)
    )

    if search:
        query = query.where(Conversation.title.ilike(f"%{search}%"))

    result = await db.execute(query)
    conversations = result.scalars().all()
    return [ConversationSummary.model_validate(c) for c in conversations]


@router.get(
    "/conversations/{conversation_id}",
    response_model=ConversationDetail,
    summary="Get conversation with full message history",
)
async def get_conversation(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ConversationDetail:
    """Return a conversation with all its messages."""
    result = await db.execute(
        select(Conversation)
        .options(selectinload(Conversation.messages))
        .where(
            Conversation.id == conversation_id,
            Conversation.user_id == current_user.id,
            Conversation.is_deleted == False,
        )
    )
    conv = result.scalar_one_or_none()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    messages_out = [
        MessageOut.model_validate(m) for m in conv.messages
    ]

    return ConversationDetail(
        id=conv.id,
        title=conv.title,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
        message_count=conv.message_count,
        messages=messages_out,
    )


@router.delete(
    "/conversations/{conversation_id}",
    response_model=MessageResponse,
    summary="Delete (soft) a conversation",
)
async def delete_conversation(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> MessageResponse:
    """Soft-delete a conversation (hides it, data retained for potential recovery)."""
    result = await db.execute(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == current_user.id,
        )
    )
    conv = result.scalar_one_or_none()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    conv.is_deleted = True
    return MessageResponse(message="Conversation deleted.")


@router.put(
    "/conversations/{conversation_id}/rename",
    response_model=ConversationSummary,
    summary="Rename a conversation",
)
async def rename_conversation(
    conversation_id: str,
    payload: ConversationRenameRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ConversationSummary:
    """Update the title of a conversation."""
    result = await db.execute(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == current_user.id,
            Conversation.is_deleted == False,
        )
    )
    conv = result.scalar_one_or_none()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    conv.title = payload.title
    await db.flush()
    await db.refresh(conv)
    return ConversationSummary.model_validate(conv)


@router.get(
    "/conversations/{conversation_id}/timeline",
    response_model=EmotionTimeline,
    summary="Get emotion timeline for visualisation",
)
async def get_emotion_timeline(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> EmotionTimeline:
    """Return timestamped emotion readings for the mood timeline chart."""
    # Verify ownership
    conv_result = await db.execute(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == current_user.id,
        )
    )
    if not conv_result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Conversation not found.")

    logs_result = await db.execute(
        select(EmotionLog)
        .where(EmotionLog.conversation_id == conversation_id)
        .order_by(EmotionLog.timestamp)
    )
    logs = logs_result.scalars().all()

    points = [
        EmotionTimePoint(
            timestamp=log.timestamp,
            emotion=log.emotion,
            confidence=log.emotion_conf,
            sentiment_score=log.sentiment_score,
        )
        for log in logs
    ]

    return EmotionTimeline(conversation_id=conversation_id, points=points)


@router.get(
    "/conversations/{conversation_id}/stats",
    response_model=ConversationStats,
    summary="Get conversation statistics",
)
async def get_conversation_stats(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ConversationStats:
    """Compute statistics for the session statistics panel."""
    conv_result = await db.execute(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == current_user.id,
        )
    )
    conv = conv_result.scalar_one_or_none()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    # Load user messages only
    msgs_result = await db.execute(
        select(Message).where(
            Message.conversation_id == conversation_id,
            Message.role == "user",
        )
    )
    user_messages = msgs_result.scalars().all()

    emotion_counts: Counter = Counter()
    sentiment_scores = []
    safety_events = 0

    for msg in user_messages:
        if msg.emotion:
            emotion_counts[msg.emotion] += 1
        if msg.sentiment_score is not None:
            sentiment_scores.append(msg.sentiment_score)
        if msg.safety_flag:
            safety_events += 1

    dominant_emotion = emotion_counts.most_common(1)[0][0] if emotion_counts else None
    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0

    now = datetime.now(timezone.utc)
    duration_seconds = int((now - conv.created_at).total_seconds())

    return ConversationStats(
        total_messages=conv.message_count,
        session_duration_seconds=duration_seconds,
        dominant_emotion=dominant_emotion,
        emotion_counts=dict(emotion_counts),
        avg_sentiment=round(avg_sentiment, 3),
        safety_events=safety_events,
    )


@router.delete(
    "/all",
    response_model=MessageResponse,
    summary="Delete all conversations for current user",
)
async def delete_all_conversations(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> MessageResponse:
    """Soft-delete all conversations for the authenticated user."""
    result = await db.execute(
        select(Conversation).where(
            Conversation.user_id == current_user.id,
            Conversation.is_deleted == False,
        )
    )
    conversations = result.scalars().all()
    for conv in conversations:
        conv.is_deleted = True
    return MessageResponse(message=f"Deleted {len(conversations)} conversation(s).")
