"""
=============================================================================
backend/services/memory_manager.py — Conversation Context Window
=============================================================================

Manages the conversation history passed to the response generator.

Why context matters:
    LLMs (including FLAN-T5) can generate coherent, contextually aware
    responses only if they "see" the conversation history. Without context,
    the model treats each message as a standalone prompt and produces
    disconnected, generic responses.

Context Window Strategy:
    We maintain a sliding window of the last N turns (configurable via
    MAX_CONTEXT_TURNS in settings). This balances:
        - Coherence: enough history to maintain conversation thread
        - Token budget: FLAN-T5-small has a 512-token input limit
        - Inference speed: shorter prompts = faster generation

Context Compression:
    When the window is full, we use a simple heuristic:
        - Always keep the FIRST message (original context/topic)
        - Always keep the LAST N-1 messages (recent context)
        - Drop oldest middle messages
    This "anchor + recency" strategy outperforms pure sliding window
    for long conversations where the initial topic is still relevant.
=============================================================================
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from backend.config import settings


class ConversationMemory:
    """
    Manages conversation history for a single chat session.

    Stores messages as a list of {"role": "user"|"assistant", "content": str}
    dicts — consistent with the OpenAI message format for portability.

    The memory is NOT persisted here — persistence is handled by the
    database layer. This class manages the IN-FLIGHT context for a single
    request cycle.
    """

    def __init__(self, max_turns: int = settings.max_context_turns) -> None:
        self.max_turns = max_turns
        self._history: List[Dict[str, str]] = []

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the in-memory history."""
        self._history.append({"role": role, "content": content})

    def get_context(self) -> List[Dict[str, str]]:
        """
        Return the context window (last max_turns messages).

        Context window strategy:
            If len(history) <= max_turns: return full history
            Else: return first message + last (max_turns - 1) messages
        """
        if len(self._history) <= self.max_turns:
            return list(self._history)

        # Anchor + recency strategy
        anchor = [self._history[0]]
        recent = self._history[-(self.max_turns - 1):]
        return anchor + recent

    def from_db_messages(self, db_messages: List[Any]) -> None:
        """
        Populate history from database Message ORM objects.

        Called when resuming an existing conversation — loads the
        most recent messages from the database into the in-flight context.

        Args:
            db_messages: List of Message ORM objects (ordered by created_at)
        """
        self._history = [
            {"role": msg.role, "content": msg.content}
            for msg in db_messages
        ]

    def clear(self) -> None:
        """Reset conversation history."""
        self._history = []

    @property
    def turn_count(self) -> int:
        """Number of turns (user + assistant messages)."""
        return len(self._history)

    @property
    def user_turn_count(self) -> int:
        """Number of user messages only."""
        return sum(1 for m in self._history if m["role"] == "user")
