"""
=============================================================================
backend/services/intent_classifier.py — BiLSTM + Attention Intent Classifier
=============================================================================

Architecture: Embedding → BiLSTM → Bahdanau Attention → Linear → Softmax

Why BiLSTM instead of another Transformer?
    Including both RNN-based and Transformer-based models in the pipeline
    demonstrates breadth of knowledge, which is pedagogically important for
    a neural networks course. BiLSTMs:

    - Are trained FROM SCRATCH (no pre-trained weights), demonstrating
      the full training pipeline: embedding initialisation, gradient flow,
      backpropagation through time (BPTT)
    - Are significantly lighter (~5M params vs 66M for DistilBERT),
      showing the compute-accuracy trade-off
    - Still achieve strong performance on intent classification tasks
      where inputs are short and categorical intent patterns are learnable
      from moderate-sized datasets

BiLSTM Architecture:
    1. Embedding layer: Integer token IDs → dense 256-dim vectors
       (learned from scratch; can optionally initialise with GloVe)
    2. Bidirectional LSTM (2 layers, 512 hidden dim):
       - Forward LSTM: reads left-to-right, captures prefix context
       - Backward LSTM: reads right-to-left, captures suffix context
       - Concatenated: 1024-dim representation per timestep
       - BPTT propagates gradients through time steps
       - Vanishing gradient mitigated by LSTM's gating mechanism:
           * Forget gate: how much past state to keep
           * Input gate: how much new info to add
           * Output gate: what to expose to next layer
    3. Bahdanau (additive) Attention:
       - Learns to weight each timestep's hidden state
       - α_t = softmax(v · tanh(W_1·h_t + W_2·query))
       - context = Σ α_t · h_t
       - Reduces sequence to a single context vector emphasising key tokens
    4. Dropout (0.3): regularisation — randomly zeros activations during training
    5. Linear head (1024 → 10): maps context vector to intent logits
    6. Softmax: probability distribution over intent classes

This module handles both training (when fine-tuned weights don't exist)
and inference using a trained checkpoint.
=============================================================================
"""

from __future__ import annotations

import asyncio
import json
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from backend.config import settings
from backend.schemas.chat import IntentResult
from backend.utils.text_utils import clean_text


# ---------------------------------------------------------------------------
# BiLSTM + Attention Model Definition
# ---------------------------------------------------------------------------

class BahdanauAttention(nn.Module):
    """
    Bahdanau (additive) attention mechanism.

    Computes a context vector as a weighted sum of LSTM hidden states.
    The attention weights (α) indicate which parts of the input sequence
    are most relevant for classifying the intent.

    Mathematically:
        score(h_t, query) = v · tanh(W_1 · h_t + W_2 · query)
        α_t = exp(score_t) / Σ_i exp(score_i)   [softmax normalisation]
        context = Σ_t α_t · h_t

    Args:
        hidden_dim: Dimension of LSTM hidden states (= 2 * lstm_hidden for BiLSTM)
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.W1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,  # (batch, seq_len, hidden_dim)
        query: torch.Tensor,           # (batch, hidden_dim)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: All LSTM hidden states (one per token)
            query: Final LSTM hidden state used as the query

        Returns:
            context: Weighted sum of hidden states (batch, hidden_dim)
            attn_weights: Attention distribution (batch, seq_len) — for visualisation
        """
        query_expanded = query.unsqueeze(1)  # (batch, 1, hidden_dim)

        # Additive scoring
        scores = self.v(
            torch.tanh(self.W1(hidden_states) + self.W2(query_expanded))
        ).squeeze(-1)  # (batch, seq_len)

        attn_weights = F.softmax(scores, dim=-1)  # (batch, seq_len)

        # Context: weighted sum
        context = torch.bmm(
            attn_weights.unsqueeze(1), hidden_states
        ).squeeze(1)  # (batch, hidden_dim)

        return context, attn_weights


class BiLSTMIntentClassifier(nn.Module):
    """
    Bidirectional LSTM with Bahdanau Attention for intent classification.

    Parameters:
        vocab_size:    Size of vocabulary (tokeniser vocabulary)
        embed_dim:     Embedding dimension (256)
        hidden_dim:    LSTM hidden units per direction (512 → 1024 bidirectional)
        num_layers:    Stacked LSTM layers (2)
        num_classes:   Number of intent classes (10)
        dropout:       Dropout probability (0.3)
        pad_idx:       Padding token index (to mask in embedding)
    """

    def __init__(
        self,
        vocab_size: int = settings.INTENT_VOCAB_SIZE,
        embed_dim: int = settings.INTENT_EMBED_DIM,
        hidden_dim: int = settings.INTENT_HIDDEN_DIM,
        num_layers: int = settings.INTENT_NUM_LAYERS,
        num_classes: int = settings.INTENT_NUM_LABELS,
        dropout: float = settings.DROPOUT_RATE,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Embedding layer
        # padding_idx ensures pad tokens contribute zero gradient
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.embed_dropout = nn.Dropout(dropout)

        # Bidirectional LSTM
        # Output dim = 2 * hidden_dim (forward + backward concatenated)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,       # (batch, seq_len, features)
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Attention layer (operates on 2*hidden_dim = 1024 dimensional states)
        self.attention = BahdanauAttention(hidden_dim * 2)

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,      # (batch, seq_len)
        lengths: Optional[torch.Tensor] = None,  # For packed sequences
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            logits: (batch, num_classes)
            attn_weights: (batch, seq_len) — for explainability
        """
        # Step 1: Embed tokens
        embedded = self.embed_dropout(self.embedding(input_ids))
        # embedded: (batch, seq_len, embed_dim)

        # Step 2: LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # lstm_out: (batch, seq_len, 2*hidden_dim)  ← all timesteps
        # hidden:   (num_layers*2, batch, hidden_dim)

        # Extract final hidden states from both directions
        # hidden[-2]: last layer, forward direction
        # hidden[-1]: last layer, backward direction
        final_hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        # final_hidden: (batch, 2*hidden_dim)

        # Step 3: Attention over all timesteps
        context, attn_weights = self.attention(lstm_out, final_hidden)
        # context: (batch, 2*hidden_dim)

        # Step 4: Classify
        logits = self.classifier(self.dropout(context))
        # logits: (batch, num_classes)

        return logits, attn_weights


# ---------------------------------------------------------------------------
# Simple word tokeniser for the intent classifier
# ---------------------------------------------------------------------------

class SimpleTokenizer:
    """
    Lightweight word-level tokenizer for the BiLSTM intent classifier.

    Unlike WordPiece (used by BERT), this tokeniser simply splits on
    whitespace and maps words to integer IDs via a vocabulary.

    Special tokens:
        <PAD>: 0 — padding to fixed length
        <UNK>: 1 — unknown words not in vocabulary
        <BOS>: 2 — beginning of sequence
        <EOS>: 3 — end of sequence
    """
    PAD_IDX = 0
    UNK_IDX = 1
    BOS_IDX = 2
    EOS_IDX = 3

    def __init__(self) -> None:
        self.word2idx: Dict[str, int] = {
            "<PAD>": self.PAD_IDX,
            "<UNK>": self.UNK_IDX,
            "<BOS>": self.BOS_IDX,
            "<EOS>": self.EOS_IDX,
        }
        self.idx2word: Dict[int, str] = {v: k for k, v in self.word2idx.items()}
        self.vocab_size: int = 4

    def build_vocab(self, texts: List[str], min_freq: int = 2) -> None:
        """Build vocabulary from a corpus of texts."""
        from collections import Counter
        word_freq: Counter = Counter()
        for text in texts:
            word_freq.update(text.lower().split())
        for word, freq in word_freq.items():
            if freq >= min_freq and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
        self.vocab_size = len(self.word2idx)

    def encode(self, text: str, max_len: int = settings.INTENT_MAX_SEQ_LEN) -> List[int]:
        """Encode text to token IDs, padded/truncated to max_len."""
        words = text.lower().split()[:max_len - 2]
        ids = [self.BOS_IDX]
        ids += [self.word2idx.get(w, self.UNK_IDX) for w in words]
        ids += [self.EOS_IDX]
        # Pad to max_len
        ids += [self.PAD_IDX] * (max_len - len(ids))
        return ids[:max_len]

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "SimpleTokenizer":
        with open(path, "rb") as f:
            return pickle.load(f)


# ---------------------------------------------------------------------------
# Inference Service
# ---------------------------------------------------------------------------

class IntentClassifier:
    """Intent classification service wrapping the BiLSTM model."""

    def __init__(self) -> None:
        self.model: Optional[BiLSTMIntentClassifier] = None
        self.tokenizer: Optional[SimpleTokenizer] = None
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.labels: List[str] = settings.INTENT_LABELS
        self._is_loaded: bool = False

    def load(self) -> None:
        """Load intent classifier from checkpoint or initialise fresh model."""
        start = time.time()
        model_dir = settings.intent_model_dir
        model_path = model_dir / "intent_model.pt"
        tokenizer_path = model_dir / "tokenizer.pkl"

        if model_path.exists() and tokenizer_path.exists():
            logger.info(f"Loading intent model from {model_path}")
            self.tokenizer = SimpleTokenizer.load(tokenizer_path)
            self.model = BiLSTMIntentClassifier(vocab_size=self.tokenizer.vocab_size)
            state = torch.load(model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state)
        else:
            logger.warning(
                "No intent model checkpoint found. Using randomly-initialised model. "
                "Run training/train_intent.py to train."
            )
            self.tokenizer = SimpleTokenizer()
            # Build a minimal vocabulary with common words
            self.tokenizer.build_vocab(["i feel sad happy anxious depressed alone"], min_freq=1)
            self.model = BiLSTMIntentClassifier(vocab_size=settings.INTENT_VOCAB_SIZE)

        self.model = self.model.to(self.device)
        self.model.eval()
        self._is_loaded = True
        logger.info(f"Intent classifier loaded in {time.time() - start:.2f}s")

    @torch.no_grad()
    def _predict(self, text: str) -> Dict[str, Any]:
        """Run intent classification inference."""
        token_ids = self.tokenizer.encode(text, max_len=settings.INTENT_MAX_SEQ_LEN)
        tensor = torch.tensor([token_ids], dtype=torch.long).to(self.device)

        logits, attn_weights = self.model(tensor)
        probs = F.softmax(logits, dim=-1)[0]

        top_k = torch.topk(probs, k=min(3, len(self.labels)))
        top_intents = [
            {"label": self.labels[idx.item()], "score": round(score.item(), 4)}
            for score, idx in zip(top_k.values, top_k.indices)
        ]

        predicted_class = torch.argmax(probs).item()
        return {
            "label": self.labels[predicted_class],
            "confidence": round(probs[predicted_class].item(), 4),
            "top_intents": top_intents,
        }

    async def classify(self, text: str) -> IntentResult:
        """Async wrapper for intent classification."""
        cleaned = clean_text(text, lowercase=True)
        if not cleaned:
            return IntentResult(
                label="small_talk", confidence=0.5,
                top_intents=[{"label": "small_talk", "score": 0.5}]
            )
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self._predict, cleaned)
        return IntentResult(**result)


_classifier: Optional[IntentClassifier] = None


def get_intent_classifier() -> IntentClassifier:
    global _classifier
    if _classifier is None:
        _classifier = IntentClassifier()
    return _classifier
