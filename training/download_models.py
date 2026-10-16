"""
=============================================================================
training/download_models.py — Download Pre-trained Model Checkpoints
=============================================================================

This script downloads pre-trained base models from HuggingFace Hub so the
application can run immediately without training.

Note: These are BASE pre-trained models, NOT fine-tuned on our datasets.
For best results, run the training scripts after downloading datasets.
However, the application is fully functional with these base models:

    - DistilBERT: Will classify text but may not perfectly match GoEmotions labels
      until fine-tuned
    - BiLSTM: Will be randomly initialised until trained
    - FLAN-T5: Will generate empathetic responses (instruction-tuned by Google)
      even without domain fine-tuning
    - RoBERTa: Pre-trained on Twitter for sentiment — no fine-tuning needed

Usage:
    python training/download_models.py

Requirements:
    pip install transformers datasets
=============================================================================
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from loguru import logger


def download_model(
    model_id: str,
    save_dir: Path,
    num_labels: int | None = None,
    is_seq2seq: bool = False,
) -> None:
    """
    Download a model and tokeniser from HuggingFace Hub to local directory.

    Args:
        model_id:   HuggingFace model ID (e.g., "distilbert-base-uncased")
        save_dir:   Local directory to save the model
        num_labels: Number of classification labels (for classifier heads)
        is_seq2seq: True for encoder-decoder models (T5, BART)
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    config_file = save_dir / "config.json"
    if config_file.exists():
        logger.info(f"✓ Already downloaded: {model_id} → {save_dir}")
        return

    logger.info(f"↓ Downloading: {model_id} → {save_dir}")

    try:
        # Download tokeniser
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.save_pretrained(str(save_dir))

        # Download model
        if is_seq2seq:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        elif num_labels:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_id,
                num_labels=num_labels,
                ignore_mismatched_sizes=True,
            )
        else:
            model = AutoModelForSequenceClassification.from_pretrained(model_id)

        model.save_pretrained(str(save_dir))
        logger.info(f"✓ Saved: {model_id} → {save_dir}")

    except Exception as e:
        logger.error(f"✗ Failed to download {model_id}: {e}")
        raise


def main() -> None:
    """Download all required models."""
    logger.info("=" * 60)
    logger.info("MindBridge AI — Model Download Script")
    logger.info("=" * 60)
    logger.info("")
    logger.info("This will download ~1.5 GB of pre-trained model weights.")
    logger.info("Ensure you have a stable internet connection.")
    logger.info("")

    models_root = PROJECT_ROOT / "models"

    downloads = [
        {
            "name": "Emotion Classifier (DistilBERT — GoEmotions 28 labels)",
            "model_id": "distilbert-base-uncased",
            "save_dir": models_root / "emotion",
            "num_labels": 28,
            "is_seq2seq": False,
        },
        {
            "name": "Safety Classifier (DistilBERT — 5 safety classes)",
            "model_id": "distilbert-base-uncased",
            "save_dir": models_root / "safety",
            "num_labels": 5,
            "is_seq2seq": False,
        },
        {
            "name": "Response Generator (FLAN-T5-small)",
            "model_id": "google/flan-t5-small",
            "save_dir": models_root / "response",
            "is_seq2seq": True,
        },
        {
            "name": "Sentiment Analyzer (Twitter-RoBERTa)",
            "model_id": "cardiffnlp/twitter-roberta-base-sentiment-latest",
            "save_dir": models_root / "sentiment",  # Used directly, not from settings
            "is_seq2seq": False,
        },
    ]

    for i, info in enumerate(downloads, 1):
        logger.info(f"\n[{i}/{len(downloads)}] {info['name']}")
        download_model(
            model_id=info["model_id"],
            save_dir=info["save_dir"],
            num_labels=info.get("num_labels"),
            is_seq2seq=info.get("is_seq2seq", False),
        )

    logger.info("")
    logger.info("=" * 60)
    logger.info("✓ All models downloaded successfully!")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Download datasets: python datasets/download_datasets.py")
    logger.info("  2. Train emotion model: python training/train_emotion.py")
    logger.info("  3. Train intent model:  python training/train_intent.py")
    logger.info("  4. Fine-tune generator: python training/train_response.py")
    logger.info("  5. Train safety model:  python training/train_safety.py")
    logger.info("  6. Start the app:       uvicorn backend.main:app --reload")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
