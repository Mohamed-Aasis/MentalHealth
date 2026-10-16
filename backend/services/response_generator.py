"""
=============================================================================
backend/services/response_generator.py — Ollama-based Response Generator
=============================================================================

Uses a locally-hosted LLM via Ollama for empathetic response generation.

Architecture:
    Ollama runs as a local server (localhost:11434) and provides a
    chat API. We send the conversation history along with a concise
    system prompt.

    Critically: emotion/intent/sentiment classifier outputs are NOT
    forwarded to the LLM. They are used only for the analytics dashboard.
    Forwarding ML labels to the LLM caused "context contamination" — the
    model would invent emotions the user never expressed.

    Fallback: If Ollama is unavailable, returns curated empathetic
    responses based on the detected emotion.

Model: mindbridge (based on gemma2:2b) — custom Modelfile with our persona.
=============================================================================
"""

from __future__ import annotations

import random
import re
from typing import List, Optional

import httpx
from loguru import logger

from backend.config import settings


# ---------------------------------------------------------------------------
# Response sanitizer
# ---------------------------------------------------------------------------
_EMOJI_ONLY_LINE = re.compile(
    r'^[\U0001F000-\U0001FFFF\U00002600-\U000027BF\U0000FE00-\U0000FE0F\s]*$'
)

def _sanitize_response(text: str) -> str:
    """
    Strip leading lines that contain only emoji/whitespace.
    Gemma2 occasionally starts a response with a bare emoji (e.g. '🙂')
    before the actual content. When that happens and the frontend
    renders it, only the emoji shows — the rest appears empty.
    """
    if not text:
        return text
    lines = text.splitlines()
    # Drop leading emoji-only lines
    while lines and _EMOJI_ONLY_LINE.match(lines[0].strip()):
        lines.pop(0)
    result = '\n'.join(lines).strip()
    return result if result else text  # fall back to original if we stripped everything


# ---------------------------------------------------------------------------
# System prompt — kept short and non-conflicting intentionally.
# A long prompt with contradictory instructions causes the model to
# over-infer and introduce emotions the user never expressed.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are MindBridge, a warm and knowledgeable mental health support companion. "
    "You are like a trusted friend who genuinely listens and gives real, practical help.\n\n"
    "How to respond:\n"
    "- Read the user's message carefully and respond to exactly what they said.\n"
    "- Be warm and natural — not clinical, not robotic, not full of hollow affirmations.\n"
    "- When someone describes a problem, offer concrete advice: techniques, steps, reframes.\n"
    "- Write in flowing paragraphs. Avoid bullet lists unless listing clear steps.\n"
    "- Match the length to the complexity of what they shared. A short message gets a focused reply; a detailed question gets a thorough one.\n"
    "- Never attribute emotions the user did not express.\n"
    "- Never diagnose. Never prescribe medication. Never claim to be a therapist.\n"
    "- For serious distress, gently suggest speaking to a licensed professional.\n"
    "- If someone expresses thoughts of suicide or self-harm, always share the crisis line: call or text 988 (Suicide & Crisis Lifeline, available 24/7)."
)


# ---------------------------------------------------------------------------
# Curated fallback responses (used when Ollama is offline)
# ---------------------------------------------------------------------------
FALLBACK_RESPONSES = {
    "sadness": [
        "That sounds really difficult. Losing sleep, feeling low energy, struggling to concentrate — these things compound each other. One small thing that can help is establishing a consistent wind-down routine: dim the lights an hour before bed, avoid screens, and do something calming like light reading or slow breathing. Would it help to talk through what might be keeping you awake?",
        "I'm sorry you're going through this. When mood and sleep suffer together it becomes a cycle — poor sleep worsens mood, low mood makes sleep harder. Breaking that cycle usually starts with the sleep side. What does your evening routine look like right now?",
    ],
    "joy": [
        "That's genuinely great to hear. What's been making things feel good lately?",
        "I'm glad things are looking up. What's been the highlight?",
    ],
    "anger": [
        "That sounds really frustrating. When we're angry it's often because something important to us has been crossed. What happened?",
        "Anger like that usually has something underneath it — disappointment, feeling unheard, or a sense of injustice. What's going on?",
    ],
    "fear": [
        "Feeling nervous before something big is incredibly common — it means you care about the outcome. One thing that helps is shifting focus from the outcome to the preparation: what's one thing you can do right now to feel more ready?",
        "That kind of anxiety is your brain trying to protect you. A simple technique: take 5 slow breaths (4 counts in, hold 4, out 6). It activates the parasympathetic nervous system and takes the edge off almost immediately.",
    ],
    "neutral": [
        "I'm here. What's on your mind?",
        "What would you like to talk about today?",
    ],
    "default": [
        "Tell me more — I want to make sure I understand what you're going through.",
        "I'm listening. What's been on your mind?",
    ],
}


class ResponseGenerator:
    """
    Ollama-based empathetic response generator.

    Connects to a locally running Ollama instance to generate
    contextually aware, emotionally attuned responses.

    NOTE: Emotion/intent/sentiment labels from the ML pipeline are
    intentionally NOT passed to this generator. They caused context
    contamination — the LLM would respond to the label rather than
    the user's actual words. Use those labels for the analytics UI only.
    """

    def __init__(self) -> None:
        self._is_loaded: bool = False
        self._base_url: str = settings.ollama_base_url
        self._model: str = settings.ollama_model
        self._client: Optional[httpx.AsyncClient] = None

    def load(self) -> None:
        """Initialize the HTTP client for Ollama."""
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=120.0,  # Generating long responses can take 60-90s on small GPU
        )
        self._is_loaded = True
        logger.info(f"Response generator ready (Ollama: {self._model} @ {self._base_url})")

    async def _generate_ollama(
        self,
        conversation_history: List[dict],
    ) -> str:
        """
        Generate a response using Ollama's chat API.

        Only the system prompt + raw conversation history are sent.
        No ML classifier outputs are injected — they cause the model
        to hallucinate emotions the user never expressed.
        """
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Add raw conversation history — just role + content, nothing else
        for turn in conversation_history[-settings.max_context_turns:]:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

        try:
            response = await self._client.post(
                "/api/chat",
                json={
                    "model": self._model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": settings.generation_temperature,
                        "top_p": settings.generation_top_p,
                        "num_predict": settings.max_new_tokens,
                    },
                },
            )
            response.raise_for_status()
            data = response.json()
            raw = data.get("message", {}).get("content", "").strip()

            # Log the raw response so we can diagnose future issues
            logger.debug(
                f"Ollama raw response ({data.get('eval_count', '?')} tokens, "
                f"done_reason={data.get('done_reason')}): "
                f"{raw[:120]!r}{'...' if len(raw) > 120 else ''}"
            )

            return _sanitize_response(raw)

        except httpx.ConnectError:
            logger.error("Ollama is not running. Start it with: ollama serve")
            return ""
        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama API error: {e.response.status_code} — {e.response.text}")
            return ""
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            return ""

    def _get_fallback_response(self, emotion: str) -> str:
        """Return a curated response when Ollama is unavailable."""
        sad_emotions = {"sadness", "grief", "disappointment", "remorse", "embarrassment"}
        happy_emotions = {"joy", "amusement", "excitement", "love", "admiration", "gratitude", "relief", "pride", "optimism"}
        angry_emotions = {"anger", "annoyance", "disapproval", "disgust"}
        fear_emotions = {"fear", "nervousness", "confusion"}

        if emotion in sad_emotions:
            category = "sadness"
        elif emotion in happy_emotions:
            category = "joy"
        elif emotion in angry_emotions:
            category = "anger"
        elif emotion in fear_emotions:
            category = "fear"
        elif emotion == "neutral":
            category = "neutral"
        else:
            category = "default"

        return random.choice(FALLBACK_RESPONSES[category])

    async def generate(
        self,
        conversation_history: List[dict],
        emotion: str,
        intent: str,
        sentiment: str,
    ) -> str:
        """
        Generate an empathetic response.

        emotion/intent/sentiment are accepted for API compatibility but are
        NOT forwarded to the LLM. They are kept for the fallback path only.
        Tries Ollama first; falls back to curated responses if unavailable.
        """
        if not self._is_loaded:
            return "I'm still starting up. Please try again in a moment."

        # Only conversation history is passed — no classifier metadata
        response = await self._generate_ollama(conversation_history)

        if not response:
            logger.warning("Ollama returned empty response, using fallback")
            response = self._get_fallback_response(emotion)

        return response

    async def generate_title(self, first_message: str) -> str:
        """
        Generate a concise title for the conversation using Ollama.
        """
        if not self._is_loaded:
            return ""

        try:
            response = await self._client.post(
                "/api/chat",
                json={
                    "model": self._model,
                    "messages": [
                        {"role": "system", "content": "You are a title generator. Generate a short, 3-5 word title summarizing the user's message. Do NOT use quotes. Do NOT add any extra text."},
                        {"role": "user", "content": first_message}
                    ],
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 12,
                    },
                },
            )
            response.raise_for_status()
            data = response.json()
            title = data.get("message", {}).get("content", "").strip()
            title = title.strip('"\'')
            return title
        except Exception as e:
            logger.error(f"Ollama title generation failed: {e}")
            return ""

    async def close(self) -> None:
        """Clean up the HTTP client."""
        if self._client:
            await self._client.aclose()


_generator: Optional[ResponseGenerator] = None


def get_response_generator() -> ResponseGenerator:
    global _generator
    if _generator is None:
        _generator = ResponseGenerator()
    return _generator
