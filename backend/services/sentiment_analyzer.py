"""
=============================================================================
backend/services/sentiment_analyzer.py — RoBERTa Sentiment Analysis
=============================================================================

Architecture: RoBERTa-base (Twitter fine-tuned) → 3-class Sentiment Head
Model: cardiffnlp/twitter-roberta-base-sentiment-latest

Why RoBERTa for sentiment?
    - RoBERTa (Robustly Optimised BERT Pretraining) removes BERT's
      Next Sentence Prediction objective and trains on more data with
      larger batches. This improves sentence-level tasks like sentiment.
    - The Cardiff NLP Twitter variant is fine-tuned on ~124M tweets,
      making it robust to informal, colloquial mental health language
      that appears on social media and chat.

Why zero-shot (no fine-tuning)?
    Sentiment (positive/negative/neutral) is a well-defined 3-class
    problem where the Cardiff model already achieves ~77% accuracy on
    Twitter data out of the box. Fine-tuning a SEPARATE model for this
    (on top of emotion and intent fine-tuning) would add significant
    compute with marginal gain. The model provides a continuous
    valence score used for the mood timeline chart.

Valence Computation:
    Valence = P(positive) − P(negative)
    Range: [−1.0, +1.0]
    −1.0 = strongly negative sentiment
     0.0 = neutral
    +1.0 = strongly positive sentiment

    This continuous score is used for:
        1. Mood timeline chart (line graph over time)
        2. Session average sentiment statistic
        3. Conditioning response generation (via emotion context)
=============================================================================
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from loguru import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from backend.config import settings, PROJECT_ROOT
from backend.schemas.chat import SentimentResult
from backend.utils.text_utils import clean_text


class SentimentAnalyzer:
    """
    Zero-shot RoBERTa sentiment analyzer.

    Provides:
        - 3-class sentiment label: positive / negative / neutral
        - Confidence score for the predicted class
        - Continuous valence score for timeline visualisation
    """

    def __init__(self) -> None:
        self.model: Optional[AutoModelForSequenceClassification] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        # RoBERTa label order: 0=negative, 1=neutral, 2=positive
        self.id2label: Dict[int, str] = {0: "negative", 1: "neutral", 2: "positive"}
        self._is_loaded: bool = False

    def load(self) -> None:
        """Load the pre-trained sentiment model (no fine-tuning required)."""
        start = time.time()
        
        local_path = PROJECT_ROOT / "models" / "sentiment"
        model_name = str(local_path) if local_path.exists() else settings.SENTIMENT_MODEL
        logger.info(f"Loading sentiment model: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, low_cpu_mem_usage=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, low_cpu_mem_usage=True)
        self.model = self.model.to(self.device)
        self.model.eval()

        elapsed = time.time() - start
        logger.info(f"Sentiment analyzer loaded in {elapsed:.2f}s")
        self._is_loaded = True

    @torch.no_grad()
    def _predict(self, text: str) -> Dict[str, Any]:
        """
        Run sentiment inference.

        The Twitter-RoBERTa model expects text with special mention/URL tokens
        pre-processed. The tokenizer handles most of this, but we also clean
        URLs/HTML before passing to the model.
        """
        inputs = self.tokenizer(
            text,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)[0]

        # probs[0]=P(negative), probs[1]=P(neutral), probs[2]=P(positive)
        p_negative = probs[0].item()
        p_neutral = probs[1].item()
        p_positive = probs[2].item()

        predicted_class = torch.argmax(probs).item()
        label = self.id2label[predicted_class]
        score = probs[predicted_class].item()

        # Valence: continuous score from -1 (very negative) to +1 (very positive)
        valence = p_positive - p_negative

        return {
            "label": label,
            "score": round(score, 4),
            "valence": round(valence, 4),
        }

    async def analyze(self, text: str) -> SentimentResult:
        """Async wrapper for sentiment analysis."""
        cleaned = clean_text(text, lowercase=False)  # Keep case for RoBERTa
        if not cleaned:
            return SentimentResult(label="neutral", score=1.0, valence=0.0)

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self._predict, cleaned)
        return SentimentResult(**result)


_analyzer: Optional[SentimentAnalyzer] = None


def get_sentiment_analyzer() -> SentimentAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentAnalyzer()
    return _analyzer
