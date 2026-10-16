"""
=============================================================================
backend/services/pipeline.py — NLP Pipeline Orchestrator
=============================================================================

The pipeline runs all ML services in the correct order for each chat turn:

    Input Text
        │
        ▼
    [1] Keyword Crisis Check  ← Fast regex/keyword scan (safety net)
        │
        ├── CRISIS → return crisis response immediately
        │
        ▼
    [2] Safety Check (ML)     ← DistilBERT classifier
        │
        ├── CRISIS → return crisis response immediately
        │
        ▼
    [3] Run in parallel:
        ├── Emotion Detection   (DistilBERT)
        ├── Intent Classification (BiLSTM)
        └── Sentiment Analysis  (RoBERTa)
        │
        ▼
    [4] Response Generation (Ollama LLM)
        │
        ▼
    JSON Result

The keyword crisis check is a critical safety net — even if the ML safety
model fails (untrained, misclassifies), dangerous keywords will always
trigger a crisis response. This prevents the catastrophic failure where
the AI says "Happy to hear that" to "I am going to die".
=============================================================================
"""

from __future__ import annotations

import asyncio
import re
from typing import Any, Dict, List, Optional

from loguru import logger

from backend.schemas.chat import AnalysisResult
from backend.services.emotion_detector import get_emotion_detector
from backend.services.intent_classifier import get_intent_classifier
from backend.services.memory_manager import ConversationMemory
from backend.services.response_generator import get_response_generator
from backend.services.safety_classifier import get_crisis_response, get_safety_classifier
from backend.services.sentiment_analyzer import get_sentiment_analyzer


# ---------------------------------------------------------------------------
# Keyword-based crisis detection (safety net)
# ---------------------------------------------------------------------------
# These patterns catch obvious crisis language even if the ML model fails.
# They are intentionally broad — false positives here are SAFE (user gets
# a compassionate crisis response), while false negatives are DANGEROUS.

CRISIS_PATTERNS = [
    r"\b(kill\s*(my)?self|suicide|suicidal)\b",
    r"\b(want|going|plan(ning)?|trying)\s+to\s+(die|end\s+(it|my\s+life|everything))\b",
    r"\b(don'?t|do\s+not)\s+want\s+to\s+(live|be\s+alive|exist)\b",
    r"\bwant\s+to\s+die\b",
    r"\bend\s+(my|this)\s+life\b",
    r"\b(cut(ting)?\s+my(self)?|self[- ]?harm)\b",
    r"\b(overdose|hang(ing)?\s+myself|jump(ing)?\s+off)\b",
    r"\bno\s+(reason|point)\s+(to|in)\s+(live|living|go(ing)?\s+on)\b",
    r"\b(better\s+off\s+dead|world.*better.*without\s+me)\b",
    r"\bgoodbye\s+forever\b",
    r"\bgoing\s+to\s+die\b",
]

_CRISIS_REGEX = re.compile(
    "|".join(CRISIS_PATTERNS),
    re.IGNORECASE,
)


def check_crisis_keywords(text: str) -> bool:
    """Fast keyword-based crisis detection. Returns True if crisis detected."""
    return bool(_CRISIS_REGEX.search(text))


class NLPPipeline:
    """
    Orchestrates the full NLP pipeline for a single chat turn.
    """

    def __init__(self) -> None:
        self.emotion_detector = get_emotion_detector()
        self.intent_classifier = get_intent_classifier()
        self.safety_classifier = get_safety_classifier()
        self.sentiment_analyzer = get_sentiment_analyzer()
        self.response_generator = get_response_generator()

    async def process(
        self,
        user_text: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Process a single user message through the full pipeline.
        """
        if conversation_history is None:
            conversation_history = []

        # ------------------------------------------------------------------
        # Step 0: Keyword-based crisis check (SAFETY NET)
        # ------------------------------------------------------------------
        keyword_crisis = check_crisis_keywords(user_text)
        if keyword_crisis:
            logger.warning(f"KEYWORD CRISIS DETECTED in: {user_text[:80]}...")

        # ------------------------------------------------------------------
        # Step 1: ML Safety check
        # ------------------------------------------------------------------
        logger.debug(f"Pipeline: running safety check on: {user_text[:80]}...")
        safety_result = await self.safety_classifier.classify(user_text)

        # Either keyword or ML trigger → crisis mode
        if keyword_crisis or safety_result.triggered:
            logger.warning(
                f"CRISIS MODE: keyword={keyword_crisis}, "
                f"ml_safety={safety_result.category}({safety_result.confidence})"
            )
            crisis_response = get_crisis_response()

            # Still run emotion/intent/sentiment for the UI
            emotion_result, intent_result, sentiment_result = await asyncio.gather(
                self.emotion_detector.detect(user_text),
                self.intent_classifier.classify(user_text),
                self.sentiment_analyzer.analyze(user_text),
            )

            # Override safety result if keyword triggered but ML didn't
            if keyword_crisis and not safety_result.triggered:
                from backend.schemas.chat import SafetyResult
                safety_result = SafetyResult(
                    is_safe=False,
                    category="suicidal_ideation",
                    confidence=1.0,
                    triggered=True,
                )

            analysis = AnalysisResult(
                emotion=emotion_result,
                intent=intent_result,
                safety=safety_result,
                sentiment=sentiment_result,
            )

            return {
                "response": crisis_response,
                "analysis": analysis,
                "is_crisis": True,
                "context_turns": len(conversation_history),
            }

        # ------------------------------------------------------------------
        # Step 2: Parallel NLP analysis
        # ------------------------------------------------------------------
        logger.debug("Pipeline: running parallel emotion/intent/sentiment analysis")
        emotion_result, intent_result, sentiment_result = await asyncio.gather(
            self.emotion_detector.detect(user_text),
            self.intent_classifier.classify(user_text),
            self.sentiment_analyzer.analyze(user_text),
        )

        logger.debug(
            f"Pipeline: emotion={emotion_result.label}({emotion_result.confidence:.2f}), "
            f"intent={intent_result.label}({intent_result.confidence:.2f}), "
            f"sentiment={sentiment_result.label}({sentiment_result.valence:.2f})"
        )

        # ------------------------------------------------------------------
        # Step 3: Build context-enriched history
        # ------------------------------------------------------------------
        memory = ConversationMemory()
        memory.from_db_messages_dict(conversation_history)
        context = memory.get_context()

        # ------------------------------------------------------------------
        # Step 4: Generate response via Ollama
        # ------------------------------------------------------------------
        logger.debug("Pipeline: generating response with Ollama")
        response_text = await self.response_generator.generate(
            conversation_history=context,
            emotion=emotion_result.label,
            intent=intent_result.label,
            sentiment=sentiment_result.label,
        )

        # ------------------------------------------------------------------
        # Assemble result
        # ------------------------------------------------------------------
        analysis = AnalysisResult(
            emotion=emotion_result,
            intent=intent_result,
            safety=safety_result,
            sentiment=sentiment_result,
        )

        return {
            "response": response_text,
            "analysis": analysis,
            "is_crisis": False,
            "context_turns": len(context),
        }


# ---------------------------------------------------------------------------
# Startup initialisation
# ---------------------------------------------------------------------------

async def load_all_models() -> None:
    """
    Load all ML models at application startup.

    Note: Response generator (Ollama) does not load a model — it only
    initializes an HTTP client. The actual LLM is managed by Ollama.
    """
    logger.info("=" * 60)
    logger.info("Loading NLP models...")
    logger.info("=" * 60)

    loop = asyncio.get_event_loop()

    models = [
        ("Emotion Detector (DistilBERT)", get_emotion_detector),
        ("Safety Classifier (DistilBERT)", get_safety_classifier),
        ("Intent Classifier (BiLSTM)", get_intent_classifier),
        ("Sentiment Analyzer (RoBERTa)", get_sentiment_analyzer),
        ("Response Generator (Ollama)", get_response_generator),
    ]

    for name, getter_fn in models:
        logger.info(f"Loading: {name}...")
        instance = getter_fn()
        await loop.run_in_executor(None, instance.load)
        logger.info(f"✓ {name} ready")

    logger.info("=" * 60)
    logger.info("All models loaded. Application ready.")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Monkey-patch ConversationMemory to accept dict list
# ---------------------------------------------------------------------------
def _from_db_messages_dict(self, messages: List[Dict[str, str]]) -> None:
    """Accept a list of dicts (from DB or memory) instead of ORM objects."""
    self._history = list(messages)


from backend.services.memory_manager import ConversationMemory
ConversationMemory.from_db_messages_dict = _from_db_messages_dict
