"""
=============================================================================
datasets/download_datasets.py — Download All Required Datasets
=============================================================================

Downloads all datasets used in the MindBridge AI training pipeline via
the HuggingFace `datasets` library.

Datasets:
    1. GoEmotions        → Emotion classification (28 classes)
    2. EmpatheticDialogues → Response generation fine-tuning
    3. DailyDialog       → Intent classification augmentation
    4. CounselChat        → Therapist-client Q&A (response augmentation)

Usage:
    python datasets/download_datasets.py
=============================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets import load_dataset, DatasetDict
from loguru import logger


DATASETS_DIR = PROJECT_ROOT / "datasets" / "cached"


def download_and_cache(name: str, *args, **kwargs) -> DatasetDict:
    """Download a dataset and cache it locally."""
    cache_path = DATASETS_DIR / name.replace("/", "_")
    if cache_path.exists():
        logger.info(f"✓ Already cached: {name}")
        return load_dataset(*args, cache_dir=str(DATASETS_DIR), **kwargs)

    logger.info(f"↓ Downloading: {name}...")
    dataset = load_dataset(*args, cache_dir=str(DATASETS_DIR), **kwargs)
    logger.info(f"✓ Downloaded: {name} — splits: {list(dataset.keys())}")
    return dataset


def main() -> None:
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("MindBridge AI — Dataset Download Script")
    logger.info("=" * 60)

    # 1. GoEmotions — for emotion classifier
    logger.info("\n[1/4] GoEmotions (Google)")
    logger.info("  58,000 Reddit comments, 28 fine-grained emotion labels")
    logger.info("  Used for: training the DistilBERT emotion classifier")
    go_emotions = download_and_cache(
        "go_emotions",
        "google-research-datasets/go_emotions",
        "simplified",
    )
    logger.info(f"  Train: {len(go_emotions['train'])}, Val: {len(go_emotions['validation'])}, Test: {len(go_emotions['test'])}")

    # 2. EmpatheticDialogues — for response generator
    logger.info("\n[2/4] EmpatheticDialogues (Facebook AI Research)")
    logger.info("  25,000 conversations grounded in emotional situations")
    logger.info("  Used for: fine-tuning FLAN-T5 response generator")
    empathetic = download_and_cache(
        "empathetic_dialogues",
        "empathetic_dialogues",
    )
    logger.info(f"  Train: {len(empathetic['train'])}, Val: {len(empathetic['validation'])}, Test: {len(empathetic['test'])}")

    # 3. DailyDialog — for intent classifier augmentation
    logger.info("\n[3/4] DailyDialog")
    logger.info("  13,000+ daily-life dialogues with intent and emotion labels")
    logger.info("  Used for: intent classification training data augmentation")
    daily_dialog = download_and_cache(
        "daily_dialog",
        "daily_dialog",
    )
    logger.info(f"  Train: {len(daily_dialog['train'])}, Val: {len(daily_dialog['validation'])}, Test: {len(daily_dialog['test'])}")

    # 4. CounselChat — therapist Q&A
    logger.info("\n[4/4] CounselChat")
    logger.info("  Real therapist-client question-answer pairs from online counselling")
    logger.info("  Used for: response generator fine-tuning augmentation")
    try:
        counsel = download_and_cache(
            "counsel_chat",
            "nbertagnolli/counsel-chat",
        )
        logger.info(f"  Entries: {len(counsel['train'])}")
    except Exception as e:
        logger.warning(f"  CounselChat download failed: {e}")
        logger.info("  Continuing without CounselChat (optional dataset).")

    logger.info("\n" + "=" * 60)
    logger.info("✓ All datasets downloaded and cached!")
    logger.info(f"  Cache location: {DATASETS_DIR}")
    logger.info("\nNext: Run the training scripts!")
    logger.info("  python training/train_emotion.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
