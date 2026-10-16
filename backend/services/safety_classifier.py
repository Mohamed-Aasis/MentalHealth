"""
=============================================================================
backend/services/safety_classifier.py — Crisis & Safety Detection
=============================================================================

Architecture: DistilBERT → Multi-class Safety Head (5 classes)

Classes:
    0: safe               — no concerning content
    1: suicidal_ideation  — expressions of wanting to die or end one's life
    2: self_harm          — self-injury, cutting, eating disorders in crisis
    3: violence           — harm to others
    4: abuse              — reports of abuse (physical, emotional, sexual)

Why a separate safety model (not just keywords)?
    Keyword-based filters produce excessive false positives:
        "I want to kill this exam" → NOT a crisis
        "I've been cutting back on sleep" → NOT self-harm
    A DistilBERT classifier understands context and produces far fewer
    false positives/negatives than word lists.

Safety Protocol (implemented in pipeline.py):
    If safety_classifier triggers:
        1. DO NOT generate a normal response
        2. Return a compassionate, scripted crisis response
        3. Always include crisis hotline number
        4. Never claim to be a therapist
        5. Encourage contacting trusted people

Note on model loading:
    This is a SEPARATE DistilBERT instance with its own classification head.
    It is loaded independently from the emotion detector despite using the
    same base architecture — this allows the two to specialise independently
    through fine-tuning on different data.
=============================================================================
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from loguru import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from backend.config import settings
from backend.schemas.chat import SafetyResult
from backend.utils.text_utils import clean_text


# ---------------------------------------------------------------------------
# Crisis response templates
# ---------------------------------------------------------------------------
CRISIS_RESPONSES = [
    (
        "I hear you, and I'm really glad you reached out. What you're going through "
        "sounds incredibly difficult, and you don't have to face this alone. "
        "Please consider reaching out to the {hotline_name} at {hotline_number} — "
        "they are available 24/7 and can provide real support. "
        "You matter, and help is available right now."
    ),
    (
        "Thank you for trusting me with something so heavy. I want you to know that "
        "your feelings are valid, and there are people trained specifically to help "
        "with what you're experiencing. Please call or text {hotline_number} ({hotline_name}) "
        "to speak with someone right now. You don't have to go through this alone."
    ),
    (
        "I'm so sorry you're feeling this way. What you're describing is serious, "
        "and you deserve real, professional support. Please reach out to "
        "{hotline_name} ({hotline_number}) — they're available 24/7 and completely free. "
        "Is there someone nearby you trust that you could be with right now?"
    ),
]

import random


def get_crisis_response() -> str:
    """Return a randomised crisis response with the configured hotline details."""
    template = random.choice(CRISIS_RESPONSES)
    return template.format(
        hotline_name=settings.crisis_hotline_name,
        hotline_number=settings.crisis_hotline_number,
    )


class SafetyClassifier:
    """
    DistilBERT-based safety classifier.

    Runs BEFORE response generation in the pipeline to ensure
    potentially harmful content never reaches the response generator.

    Decision threshold:
        If the probability for any non-safe class exceeds SAFETY_THRESHOLD,
        the safety flag is raised. The threshold is configurable — lower values
        increase sensitivity (more false positives), higher values increase
        specificity (may miss edge cases).

        Default: 0.65 (tuned to balance FP/FN on mental health chat data)
    """

    def __init__(self) -> None:
        self.model: Optional[AutoModelForSequenceClassification] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.labels: List[str] = settings.SAFETY_LABELS
        self._is_loaded: bool = False

    def load(self) -> None:
        """Load safety classifier from local checkpoint or base model."""
        start = time.time()

        local_path = settings.safety_model_dir
        if local_path.exists() and any(local_path.iterdir()):
            model_name_or_path = str(local_path)
            logger.info(f"Loading safety model from local checkpoint: {local_path}")
        else:
            model_name_or_path = settings.SAFETY_BASE_MODEL
            logger.warning(
                f"No local safety checkpoint. Loading base model. "
                f"Run training/train_safety.py to fine-tune."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            num_labels=settings.SAFETY_NUM_LABELS,
            ignore_mismatched_sizes=True,
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        elapsed = time.time() - start
        logger.info(f"Safety classifier loaded in {elapsed:.2f}s on {self.device}")
        self._is_loaded = True

    @torch.no_grad()
    def _predict(self, text: str) -> Dict[str, Any]:
        """
        Classify text safety.

        Returns the safety category with highest probability.
        If the top category is "safe", checks whether any non-safe class
        exceeds the threshold (catches cases where "safe" is chosen but
        a concerning class has significant probability mass).

        Multi-label consideration:
            We treat this as single-label (argmax) but also check if any
            non-safe class exceeds the threshold independently.
            This catches cases like: P(safe)=0.55, P(suicidal)=0.72
            where argmax gives "safe" but the crisis threshold is triggered.
        """
        inputs = self.tokenizer(
            text,
            max_length=settings.MAX_SEQ_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)[0]

        predicted_class = torch.argmax(probs).item()
        predicted_label = self.labels[predicted_class]
        confidence = probs[predicted_class].item()

        # Check if any non-safe class exceeds the crisis threshold
        triggered = False
        triggered_category = "safe"
        triggered_conf = 0.0

        for i, label in enumerate(self.labels):
            if label != "safe" and probs[i].item() >= settings.safety_threshold:
                if probs[i].item() > triggered_conf:
                    triggered = True
                    triggered_category = label
                    triggered_conf = probs[i].item()

        if triggered:
            return {
                "is_safe": False,
                "category": triggered_category,
                "confidence": round(triggered_conf, 4),
                "triggered": True,
            }

        return {
            "is_safe": predicted_label == "safe",
            "category": predicted_label,
            "confidence": round(confidence, 4),
            "triggered": predicted_label != "safe" and confidence >= settings.safety_threshold,
        }

    async def classify(self, text: str) -> SafetyResult:
        """Async wrapper — runs inference in thread pool executor."""
        cleaned = clean_text(text, lowercase=True)
        if not cleaned:
            return SafetyResult(is_safe=True, category="safe", confidence=1.0, triggered=False)

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self._predict, cleaned)
        return SafetyResult(**result)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_classifier: Optional[SafetyClassifier] = None


def get_safety_classifier() -> SafetyClassifier:
    global _classifier
    if _classifier is None:
        _classifier = SafetyClassifier()
    return _classifier
