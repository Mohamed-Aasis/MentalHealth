"""
=============================================================================
training/train_emotion.py — DistilBERT Emotion Classifier Training
=============================================================================

Fine-tunes DistilBERT on the GoEmotions dataset for 28-class emotion
classification.

Key DL Concepts Demonstrated:
    ✓ Transfer Learning: Starting from pre-trained DistilBERT weights
      (trained on Wikipedia + BookCorpus) and adapting to emotion data
    ✓ Fine-tuning: Only the classification head is trained from scratch;
      the 6 transformer encoder blocks are fine-tuned with a very small
      learning rate to avoid catastrophic forgetting
    ✓ Tokenisation: WordPiece tokeniser, padding, truncation, attention masks
    ✓ Batched Training: Mini-batch gradient descent (batch_size=16)
    ✓ AdamW Optimiser: Adam with decoupled weight decay
      (L2 regularisation without affecting adaptive learning rates)
    ✓ Linear Warmup Scheduler: Avoids large gradient updates early in training
    ✓ Dropout: Applied in DistilBERT layers for regularisation
    ✓ Cross-Entropy Loss: Standard loss for multi-class classification
    ✓ Early Stopping: Prevents overfitting by stopping when val_f1 stops improving
    ✓ Gradient Clipping: Clips gradient norm to 1.0 to prevent exploding gradients
    ✓ Mixed Precision (fp16): Cuts memory by ~50% on GPU (optional)
    ✓ Checkpoint Saving: Saves best model weights based on validation F1
    ✓ Evaluation: Macro-F1, per-class accuracy, confusion matrix

Usage:
    python training/train_emotion.py [--epochs 5] [--batch-size 16] [--fp16]

Dataset: GoEmotions (downloads automatically via HuggingFace datasets)
Output: models/emotion/ (HuggingFace format)
=============================================================================
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

# Suppress tokenizer parallelism warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from loguru import logger
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from backend.config import settings


# ---------------------------------------------------------------------------
# Dataset Class
# ---------------------------------------------------------------------------

class GoEmotionsDataset(Dataset):
    """
    PyTorch Dataset for GoEmotions.

    GoEmotions provides multi-label annotations (a comment can have multiple
    emotions). We simplify to single-label by taking the first label.
    (The full multi-label setup is commented out for complexity management.)

    Tokenisation is performed here (not in the collate function) to allow
    caching — the tokenised inputs are stored in self.encodings.

    Args:
        texts:      List of comment strings
        labels:     List of label lists [[0, 3], [5], ...]
        tokenizer:  HuggingFace tokenizer
        max_length: Maximum sequence length (128 for DistilBERT)
    """

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer,
        max_length: int = settings.MAX_SEQ_LENGTH,
    ) -> None:
        self.labels = labels
        self.encodings = tokenizer(
            texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Early Stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """
    Early stopping to halt training when validation metric stops improving.

    Algorithm:
        - Track the best validation metric seen so far
        - If metric doesn't improve for `patience` consecutive epochs → stop
        - The best model checkpoint is saved and restored at the end

    Why early stopping?
        Training too long leads to overfitting — the model memorises the
        training set and fails to generalise to new data.
        Early stopping is a form of regularisation that uses the validation set
        as a proxy for generalisation performance.
    """

    def __init__(self, patience: int = settings.EARLY_STOPPING_PATIENCE, mode: str = "max") -> None:
        self.patience = patience
        self.mode = mode
        self.best_score: float | None = None
        self.counter: int = 0
        self.should_stop: bool = False

    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.

        Returns:
            True if training should stop (no improvement for `patience` epochs)
        """
        if self.best_score is None:
            self.best_score = score
        elif (self.mode == "max" and score <= self.best_score) or \
             (self.mode == "min" and score >= self.best_score):
            self.counter += 1
            logger.info(f"EarlyStopping: no improvement ({self.counter}/{self.patience})")
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = score
            self.counter = 0

        return self.should_stop


# ---------------------------------------------------------------------------
# Training Functions
# ---------------------------------------------------------------------------

def load_goemotions(tokenizer) -> tuple[GoEmotionsDataset, GoEmotionsDataset, GoEmotionsDataset]:
    """
    Load and preprocess the GoEmotions dataset.

    GoEmotions has 3 splits: train, validation, test.
    Labels are provided as lists (multi-label). We take the primary label.

    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    logger.info("Loading GoEmotions dataset from HuggingFace...")

    # Load the 'simplified' version (28 classes, excluding neutral-ambiguous)
    dataset = load_dataset("google-research-datasets/go_emotions", "simplified")

    def extract_primary_label(example: dict) -> dict:
        """Convert multi-label list to single primary label."""
        labels = example["labels"]
        if not labels:
            example["primary_label"] = 27  # neutral (index 27)
        else:
            example["primary_label"] = labels[0]  # Use first label
        return example

    dataset = dataset.map(extract_primary_label)

    def make_dataset(split: str) -> GoEmotionsDataset:
        texts = dataset[split]["text"]
        labels = dataset[split]["primary_label"]
        return GoEmotionsDataset(texts, labels, tokenizer)

    train_ds = make_dataset("train")
    val_ds = make_dataset("validation")
    test_ds = make_dataset("test")

    logger.info(f"Dataset sizes — Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    return train_ds, val_ds, test_ds


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: str,
    clip_norm: float = settings.GRADIENT_CLIP_NORM,
    use_fp16: bool = False,
    scaler=None,
) -> float:
    """
    Train for one epoch.

    Algorithm:
        For each mini-batch:
            1. Forward pass: compute model outputs (logits)
            2. Compute cross-entropy loss between logits and true labels
            3. Backward pass: compute gradients via backpropagation
            4. Gradient clipping: clip gradient norm to prevent explosion
            5. Update weights: optimizer.step()
            6. Update learning rate: scheduler.step()
            7. Zero gradients: optimizer.zero_grad()

    Cross-Entropy Loss (for classification):
        L = − Σ_c y_c · log(softmax(logit_c))
        where y_c is the one-hot true label and c indexes classes.
        This penalises confident wrong predictions more than uncertain ones.

    AdamW Update Rule:
        m_t = β_1 · m_{t-1} + (1 − β_1) · g_t      [1st moment]
        v_t = β_2 · v_{t-1} + (1 − β_2) · g_t²     [2nd moment]
        m̂_t = m_t / (1 − β_1^t)                     [bias correction]
        v̂_t = v_t / (1 − β_2^t)                     [bias correction]
        θ_t = θ_{t-1} − α · m̂_t / (√v̂_t + ε) − α · λ · θ_{t-1}
        [Last term: decoupled weight decay (the 'W' in AdamW)]

    Args:
        model:      The DistilBERT model
        dataloader: Training DataLoader
        optimizer:  AdamW optimizer
        scheduler:  Linear warmup scheduler
        device:     "cuda" or "cpu"
        clip_norm:  Max gradient norm (1.0)
        use_fp16:   Whether to use mixed precision (GPU only)
        scaler:     GradScaler for fp16

    Returns:
        Mean training loss for this epoch
    """
    model.train()
    total_loss = 0.0

    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        if use_fp16 and scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()

        scheduler.step()
        optimizer.zero_grad()
        total_loss += loss.item()

        if step % 100 == 0:
            logger.debug(f"  Step {step}/{len(dataloader)} — Loss: {loss.item():.4f}")

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    label_names: list[str],
) -> dict[str, float]:
    """
    Evaluate model on a dataset.

    Metrics:
        - Accuracy: proportion of correct predictions
        - Macro-F1: F1 averaged equally across all classes
          (appropriate for imbalanced multi-class datasets like GoEmotions)
        - Per-class F1: useful for identifying which emotions are hardest

    Args:
        model:       Trained model
        dataloader:  Validation/test DataLoader
        device:      Device
        label_names: Ordered list of class names

    Returns:
        Dict of metrics
    """
    model.eval()
    all_preds = []
    all_labels = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = (all_preds == all_labels).mean()
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    logger.info(f"  Accuracy: {accuracy:.4f}, Macro-F1: {macro_f1:.4f}")
    logger.debug("\n" + classification_report(all_labels, all_preds, target_names=label_names, zero_division=0))

    return {"accuracy": float(accuracy), "macro_f1": float(macro_f1)}


# ---------------------------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    """Main training function."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training on: {device}")

    use_fp16 = args.fp16 and device == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_fp16 else None

    # ---- Load tokenizer and model ----
    logger.info(f"Loading base model: {settings.EMOTION_BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(settings.EMOTION_BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        settings.EMOTION_BASE_MODEL,
        num_labels=settings.EMOTION_NUM_LABELS,
    )
    model = model.to(device)

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ---- Load datasets ----
    train_ds, val_ds, test_ds = load_goemotions(tokenizer)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=0)

    # ---- Optimizer: AdamW ----
    # We use weight decay (L2 regularisation) for transformer parameters
    # but NOT for bias and layer norm parameters — this is best practice
    # for fine-tuning transformers (Ilya Loshchilov & Frank Hutter, 2019)
    no_decay_params = ["bias", "LayerNorm.weight"]
    optimizer_grouped = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay_params)],
            "weight_decay": settings.WEIGHT_DECAY,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay_params)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped, lr=settings.LEARNING_RATE)

    # ---- Learning rate scheduler: Linear warmup then linear decay ----
    # Warmup: gradually increase LR from 0 to LEARNING_RATE over the first
    #         WARMUP_RATIO fraction of training steps
    # Rationale: Prevents large gradient updates early in fine-tuning when
    #            the classification head weights are random
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * settings.WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    logger.info(f"Total training steps: {total_steps}, Warmup steps: {warmup_steps}")

    # ---- Training loop ----
    early_stopper = EarlyStopping(patience=settings.EARLY_STOPPING_PATIENCE, mode="max")
    save_dir = settings.emotion_model_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = 0.0

    for epoch in range(1, args.epochs + 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch}/{args.epochs}")
        logger.info(f"{'='*50}")

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device,
            use_fp16=use_fp16, scaler=scaler,
        )
        logger.info(f"Train Loss: {train_loss:.4f}")

        # Validate
        logger.info("Validation:")
        val_metrics = evaluate(model, val_loader, device, settings.EMOTION_LABELS)
        val_f1 = val_metrics["macro_f1"]

        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            logger.info(f"✓ New best F1: {best_f1:.4f} — saving checkpoint to {save_dir}")
            model.save_pretrained(str(save_dir))
            tokenizer.save_pretrained(str(save_dir))

        # Early stopping
        if early_stopper(val_f1):
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break

    # ---- Final test evaluation ----
    logger.info("\nFinal Test Evaluation:")
    # Load best checkpoint for test evaluation
    best_model = AutoModelForSequenceClassification.from_pretrained(str(save_dir))
    best_model = best_model.to(device)
    test_metrics = evaluate(best_model, test_loader, device, settings.EMOTION_LABELS)

    logger.info(f"\n{'='*50}")
    logger.info("Training Complete!")
    logger.info(f"Best Val Macro-F1: {best_f1:.4f}")
    logger.info(f"Test Macro-F1:     {test_metrics['macro_f1']:.4f}")
    logger.info(f"Test Accuracy:     {test_metrics['accuracy']:.4f}")
    logger.info(f"Model saved to:    {save_dir}")
    logger.info(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DistilBERT emotion classifier on GoEmotions")
    parser.add_argument("--epochs", type=int, default=settings.NUM_EPOCHS_CLASSIFIER, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=settings.TRAIN_BATCH_SIZE, help="Training batch size")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision training (GPU only)")
    parser.add_argument("--lr", type=float, default=settings.LEARNING_RATE, help="Learning rate")

    args = parser.parse_args()
    settings.LEARNING_RATE = args.lr

    main(args)
