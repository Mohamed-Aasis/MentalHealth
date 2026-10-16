"""
=============================================================================
backend/routers/export.py — Conversation Export Routes
=============================================================================

Endpoints:
    GET /api/export/{conversation_id}/json   — Export as JSON
    GET /api/export/{conversation_id}/txt    — Export as plain text
    GET /api/export/{conversation_id}/csv    — Export as CSV
=============================================================================
"""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response, StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.connection import get_db
from backend.database.models import Conversation, Message, User
from backend.utils.auth_utils import get_current_user

router = APIRouter(prefix="/api/export", tags=["Export"])


async def _get_conversation(
    conversation_id: str,
    user: User,
    db: AsyncSession,
) -> Conversation:
    """Helper to fetch and verify conversation ownership."""
    result = await db.execute(
        select(Conversation)
        .options(selectinload(Conversation.messages))
        .where(
            Conversation.id == conversation_id,
            Conversation.user_id == user.id,
            Conversation.is_deleted == False,
        )
    )
    conv = result.scalar_one_or_none()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return conv


@router.get(
    "/{conversation_id}/json",
    summary="Export conversation as JSON",
)
async def export_json(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Response:
    """Export a full conversation (messages + analysis) as structured JSON."""
    conv = await _get_conversation(conversation_id, current_user, db)

    export_data = {
        "export_metadata": {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "app": "MindBridge AI",
            "version": "1.0.0",
        },
        "conversation": {
            "id": conv.id,
            "title": conv.title,
            "created_at": conv.created_at.isoformat(),
            "updated_at": conv.updated_at.isoformat(),
            "message_count": conv.message_count,
        },
        "messages": [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "created_at": m.created_at.isoformat(),
                "emotion": m.emotion,
                "emotion_confidence": m.emotion_conf,
                "intent": m.intent,
                "intent_confidence": m.intent_conf,
                "sentiment_score": m.sentiment_score,
                "safety_flag": m.safety_flag,
            }
            for m in sorted(conv.messages, key=lambda x: x.created_at)
        ],
    }

    json_bytes = json.dumps(export_data, indent=2, ensure_ascii=False).encode("utf-8")
    filename = f"mindbridge_{conversation_id[:8]}.json"

    return Response(
        content=json_bytes,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get(
    "/{conversation_id}/txt",
    summary="Export conversation as plain text",
)
async def export_txt(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Response:
    """Export conversation as a readable text transcript."""
    conv = await _get_conversation(conversation_id, current_user, db)
    messages = sorted(conv.messages, key=lambda x: x.created_at)

    lines = [
        f"MindBridge AI — Conversation Export",
        f"Title: {conv.title}",
        f"Date: {conv.created_at.strftime('%Y-%m-%d %H:%M UTC')}",
        f"Messages: {conv.message_count}",
        "=" * 60,
        "",
    ]

    for msg in messages:
        role_label = "You" if msg.role == "user" else "MindBridge"
        time_str = msg.created_at.strftime("%H:%M")
        lines.append(f"[{time_str}] {role_label}:")
        lines.append(msg.content)
        if msg.role == "user" and msg.emotion:
            lines.append(f"  (Emotion: {msg.emotion}, Sentiment: {msg.sentiment_score:+.2f})")
        lines.append("")

    txt_content = "\n".join(lines).encode("utf-8")
    filename = f"mindbridge_{conversation_id[:8]}.txt"

    return Response(
        content=txt_content,
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get(
    "/{conversation_id}/csv",
    summary="Export conversation as CSV",
)
async def export_csv(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """Export conversation messages as a CSV file (useful for analysis)."""
    conv = await _get_conversation(conversation_id, current_user, db)
    messages = sorted(conv.messages, key=lambda x: x.created_at)

    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=[
            "timestamp", "role", "content", "emotion", "emotion_confidence",
            "intent", "intent_confidence", "sentiment_score", "safety_flag",
        ],
    )
    writer.writeheader()

    for msg in messages:
        writer.writerow({
            "timestamp": msg.created_at.isoformat(),
            "role": msg.role,
            "content": msg.content,
            "emotion": msg.emotion or "",
            "emotion_confidence": f"{msg.emotion_conf:.4f}" if msg.emotion_conf else "",
            "intent": msg.intent or "",
            "intent_confidence": f"{msg.intent_conf:.4f}" if msg.intent_conf else "",
            "sentiment_score": f"{msg.sentiment_score:.4f}" if msg.sentiment_score is not None else "",
            "safety_flag": str(msg.safety_flag),
        })

    output.seek(0)
    filename = f"mindbridge_{conversation_id[:8]}.csv"

    return StreamingResponse(
        io.BytesIO(output.getvalue().encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
