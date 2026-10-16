"""
=============================================================================
backend/services/emotion_detector.py — DistilBERT Emotion Classifier
=============================================================================

Architecture: Transformer Encoder → Classification Head
Model: distilbert-base-uncased fine-tuned on GoEmotions (28 classes)

Why DistilBERT over BERT?
    - DistilBERT retains 97% of BERT's performance at 40% fewer parameters
      (66M vs 110M). It uses knowledge distillation: a smaller "student" model
      is trained to mimic the output distribution of the larger "teacher" (BERT).
    - Faster inference is critical in a real-time chat application where
      multiple models run sequentially per message.

Why Transformers over traditional ML for emotion detection?
    - Traditional ML (TF-IDF + SVM) treats text as a bag of words —
      it loses word order and fails to capture context. "I am not happy"
      and "I am happy" would have similar TF-IDF vectors.
    - Self-attention in Transformers relates every token to every other token,
      capturing nuanced semantic context. The model learns that "not happy"
      carries negative valence through contextual embeddings.

How DistilBERT works:
    1. Tokenisation: Input text → WordPiece token IDs + attention mask
    2. Embedding layer: Token IDs → 768-dim dense vectors
       (token embedding + positional embedding + segment embedding)
    3. 6 Transformer encoder blocks, each containing:
       a. Multi-head self-attention (12 heads, 64 dims each)
          - Q, K, V projections computed for each token
          - Attention weights: softmax(QKᵀ / √d_k)
          - Output: weighted sum of values
       b. Feed-forward network (2 layers, 3072 hidden dim)
       c. Layer normalisation + residual connections
    4. [CLS] token representation → 768-dim sentence embedding
    5. Dropout (0.1) → Linear classification head (768 → 28)
    6. Softmax → probability distribution over 28 emotion labels

GoEmotions Dataset:
    58,000 Reddit comments labelled with 28 fine-grained emotions.
    Multi-label (a comment can express multiple emotions simultaneously).
    We treat it as single-label by taking the argmax for simplicity,
    but the top-k probabilities are returned for the UI.
=============================================================================
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from loguru import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from backend.config import settings
from backend.schemas.chat import EmotionResult
from backend.utils.text_utils import clean_text


class EmotionDetector:
    """
    DistilBERT-based emotion classifier.

    Manages model loading, inference, and result formatting.
    Loaded once at application startup and reused across all requests
    (model objects are not thread-safe but Python's GIL + asyncio
    thread pool ensures safe concurrent access).
    """

    def __init__(self) -> None:
        self.model: Optional[AutoModelForSequenceClassification] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.labels: List[str] = settings.EMOTION_LABELS
        self._is_loaded: bool = False

    def load(self) -> None:
        """
        Load the model from local checkpoint or HuggingFace Hub.

        Loading order:
            1. Check if fine-tuned checkpoint exists locally (models/emotion/)
            2. If yes → load from disk (faster, uses your trained weights)
            3. If no → load base pre-trained model from HuggingFace Hub
               (still functional, just not fine-tuned on GoEmotions)

        Model is moved to GPU if available, then set to eval() mode.
        eval() disables dropout layers which are only needed during training.
        """
        start = time.time()

        local_path = settings.emotion_model_dir
        if local_path.exists() and any(local_path.iterdir()):
            model_name_or_path = str(local_path)
            logger.info(f"Loading emotion model from local checkpoint: {local_path}")
        else:
            model_name_or_path = settings.EMOTION_BASE_MODEL
            logger.warning(
                f"No local emotion checkpoint found at {local_path}. "
                f"Loading base model '{model_name_or_path}' from HuggingFace Hub. "
                f"Run training/train_emotion.py to fine-tune."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        # If loading base model (not fine-tuned), we need to provide num_labels
        # so the classification head is initialised with the correct output size.
        if model_name_or_path == settings.EMOTION_BASE_MODEL:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name_or_path,
                num_labels=settings.EMOTION_NUM_LABELS,
                ignore_mismatched_sizes=True,
            )
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name_or_path
            )

        self.model = self.model.to(self.device)
        self.model.eval()

        elapsed = time.time() - start
        logger.info(f"Emotion detector loaded in {elapsed:.2f}s on {self.device}")
        self._is_loaded = True

    @torch.no_grad()
    def _predict(self, text: str) -> Dict[str, Any]:
        """
        Run synchronous inference on a single text input.

        Why @torch.no_grad():
            Disables gradient computation (not needed for inference).
            This reduces memory usage by ~50% and speeds up forward pass
            because PyTorch doesn't need to build the computation graph.

        Tokenisation details:
            - max_length=128: truncate long inputs (emotion is usually
              apparent from the first 50–100 tokens)
            - padding="max_length": pad short inputs (enables batching)
            - truncation=True: handle inputs longer than max_length
            - return_tensors="pt": return PyTorch tensors (not lists)

        Args:
            text: Cleaned user input text

        Returns:
            Dict with predicted label, confidence, and top-k emotions
        """
        if not self._is_loaded:
            raise RuntimeError("Emotion detector not loaded. Call load() first.")

        # Tokenise
        inputs = self.tokenizer(
            text,
            max_length=settings.MAX_SEQ_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        # Move tensors to the same device as the model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Forward pass through DistilBERT
        # outputs.logits shape: (1, num_labels) = (1, 28)
        outputs = self.model(**inputs)
        logits = outputs.logits  # Raw scores (unnormalised)

        # Convert logits to probabilities using softmax
        # Softmax: P(class_i) = exp(logit_i) / Σ exp(logit_j)
        # This ensures probabilities sum to 1.0
        probabilities = F.softmax(logits, dim=-1)

        # Get top-5 emotions for the UI
        top_k = torch.topk(probabilities[0], k=min(5, len(self.labels)))
        top_emotions = [
            {"label": self.labels[idx.item()], "score": round(score.item(), 4)}
            for score, idx in zip(top_k.values, top_k.indices)
        ]

        # Primary prediction
        predicted_class = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0][predicted_class].item()

        return {
            "label": self.labels[predicted_class],
            "confidence": round(confidence, 4),
            "top_emotions": top_emotions,
        }

    async def detect(self, text: str) -> EmotionResult:
        """
        Async wrapper around _predict().

        The synchronous _predict() runs in a thread pool executor to avoid
        blocking FastAPI's event loop during the CPU/GPU-intensive forward pass.

        asyncio.get_event_loop().run_in_executor(None, fn) uses the default
        ThreadPoolExecutor — appropriate for CPU-bound tasks that release the GIL
        (PyTorch operations do release the GIL during C++ execution).

        Args:
            text: Raw user input

        Returns:
            EmotionResult Pydantic model
        """
        # Clean text (lowercase for classification)
        cleaned = clean_text(text, lowercase=True, max_length=512)
        if not cleaned:
            # Fallback for empty input
            return EmotionResult(
                label="neutral",
                confidence=1.0,
                top_emotions=[{"label": "neutral", "score": 1.0}],
            )

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self._predict, cleaned)

        return EmotionResult(**result)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_detector: Optional[EmotionDetector] = None


def get_emotion_detector() -> EmotionDetector:
    """Return the module-level singleton EmotionDetector (loaded at startup)."""
    global _detector
    if _detector is None:
        _detector = EmotionDetector()
    return _detector
