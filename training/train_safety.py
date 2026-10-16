"""
=============================================================================
training/train_safety.py — Safety Classifier Training
=============================================================================

Fine-tunes DistilBERT as a 5-class safety classifier to detect:
    0: safe               — normal mental health conversation
    1: suicidal_ideation  — expressions of suicidal thoughts
    2: self_harm          — self-injury related content
    3: violence           — threats/descriptions of violence to others
    4: abuse              — reports of abuse

Dataset Strategy:
    Since no single public dataset covers all 5 safety categories cleanly,
    we use a combination:
    1. Synthetic training examples (bootstrapping)
    2. CrisisNLP dataset (Twitter crisis messages)
    3. CLPsych 2015 shared task data (if available)
    4. Safe examples from DailyDialog and EmpatheticDialogues

Class Imbalance Handling:
    Safety data is highly imbalanced — most messages are "safe".
    We use:
    - Weighted cross-entropy loss: weight = 1 / class_frequency
    - Oversampling of minority classes

Usage:
    python training/train_safety.py [--epochs 5]
=============================================================================
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from backend.config import settings
from training.train_emotion import EarlyStopping


# ---------------------------------------------------------------------------
# Synthetic Training Data
# ---------------------------------------------------------------------------

SAFETY_EXAMPLES: List[Tuple[str, int]] = [
    # safe (0) - 40 examples
    ("i've been feeling anxious about my exams", 0),
    ("my relationship is going through a rough patch", 0),
    ("i feel sad and lonely sometimes", 0),
    ("i'm struggling with work-life balance", 0),
    ("i had a bad day today and need to vent", 0),
    ("i'm worried about my health", 0),
    ("i feel overwhelmed with responsibilities", 0),
    ("i'm having trouble sleeping due to stress", 0),
    ("i feel disconnected from my friends", 0),
    ("i've been experiencing mood swings", 0),
    ("how can i manage my anxiety better", 0),
    ("i feel really down today", 0),
    ("i'm going through a difficult time", 0),
    ("i feel like no one understands me", 0),
    ("i'm struggling to find motivation", 0),
    ("i feel empty inside sometimes", 0),
    ("my depression makes everyday tasks hard", 0),
    ("i'm having panic attacks occasionally", 0),
    ("i feel stuck in my life", 0),
    ("i'm grieving the loss of a loved one", 0),
    ("hello how are you today", 0),
    ("i want to talk about my feelings", 0),
    ("thank you for listening to me", 0),
    ("can you help me with coping strategies", 0),
    ("i'm feeling better than before", 0),
    ("i had a nice day today", 0),
    ("i'm working on improving my mental health", 0),
    ("i started therapy recently", 0),
    ("exercise has been helping my mood", 0),
    ("i'm learning to set boundaries", 0),
    ("i feel hopeful about my future", 0),
    ("i had a breakthrough in understanding myself", 0),
    ("mindfulness has been helping me", 0),
    ("i reached out to a friend today", 0),
    ("i'm practicing self-care", 0),
    ("i talked to my doctor about my medication", 0),
    ("i'm joining a support group", 0),
    ("journaling has been therapeutic for me", 0),
    ("i feel more stable than last month", 0),
    ("i want to understand my triggers better", 0),

    # suicidal_ideation (1) - 20 examples
    ("i've been thinking about ending my life", 1),
    ("i don't see a reason to keep living", 1),
    ("i want to kill myself", 1),
    ("i've been having thoughts of suicide", 1),
    ("i wish i could just die", 1),
    ("life doesn't feel worth living anymore", 1),
    ("i've thought about how i would do it", 1),
    ("everyone would be better off without me", 1),
    ("i'm planning to end things soon", 1),
    ("i've been saying my goodbyes", 1),
    ("there's no point in continuing to live", 1),
    ("i want to disappear permanently", 1),
    ("i've written a note to my family", 1),
    ("i don't want to wake up tomorrow", 1),
    ("i'm at the end of my rope and want to die", 1),
    ("i feel like the world would be better without me", 1),
    ("i've been googling methods", 1),
    ("i'm not afraid of death anymore i welcome it", 1),
    ("i've set a date to end my life", 1),
    ("suicide seems like the only way out", 1),

    # self_harm (2) - 20 examples
    ("i've been cutting myself to cope with pain", 2),
    ("i hurt myself when i feel overwhelmed", 2),
    ("i burn myself when things get too much", 2),
    ("i've been starving myself", 2),
    ("i hit myself when i'm angry", 2),
    ("i scratch my skin until it bleeds", 2),
    ("hurting myself gives me a sense of control", 2),
    ("i've been pulling my hair out from stress", 2),
    ("i bang my head against the wall when upset", 2),
    ("i've been using substances to hurt my body", 2),
    ("i relapsed on self-harm last night", 2),
    ("i have fresh cuts on my arms", 2),
    ("i hurt myself because i feel i deserve it", 2),
    ("i've been overdosing on medication on purpose", 2),
    ("i can't stop hurting myself", 2),
    ("self-harm is the only way i feel real", 2),
    ("i bruise myself deliberately", 2),
    ("i've been binge-purging to hurt my body", 2),
    ("picking at wounds makes me feel better temporarily", 2),
    ("i hurt myself three times this week", 2),

    # violence (3) - 15 examples
    ("i want to hurt someone", 3),
    ("i'm going to attack my coworker", 3),
    ("i feel like hurting people around me", 3),
    ("i've been fantasising about violence", 3),
    ("i want to make them pay", 3),
    ("i'm going to destroy everything", 3),
    ("i feel violent urges toward others", 3),
    ("i want to kill my abuser", 3),
    ("i'm going to hurt my ex partner", 3),
    ("i feel like lashing out physically", 3),
    ("i've threatened to hurt my family", 3),
    ("i want to attack the person who wronged me", 3),
    ("i can't control my violent impulses", 3),
    ("i've been violent toward my partner", 3),
    ("i plan to harm someone tomorrow", 3),

    # abuse (4) - 15 examples
    ("my partner hits me regularly", 4),
    ("i'm being emotionally abused at home", 4),
    ("someone is sexually abusing me", 4),
    ("my parent physically harms me", 4),
    ("i'm trapped in an abusive relationship", 4),
    ("my partner controls everything i do", 4),
    ("i'm being stalked by my ex", 4),
    ("someone is exploiting me financially", 4),
    ("i'm being bullied and harassed at work", 4),
    ("my caregiver hurts me when no one is watching", 4),
    ("i've been in an abusive situation for years", 4),
    ("someone forced me to do things i didn't want to", 4),
    ("i'm being gaslit by my partner constantly", 4),
    ("my family is abusing me psychologically", 4),
    ("i feel unsafe at home due to violence", 4),
]


def augment_safety_examples(
    examples: List[Tuple[str, int]], safe_factor: int = 5, crisis_factor: int = 15
) -> List[Tuple[str, int]]:
    """
    Augment data to address class imbalance.

    Strategy:
        - Safe class: augment by 5x (has most examples)
        - Crisis classes: augment by 15x (minority classes need more)
    """
    augmented = []
    prefixes = ["", "honestly ", "to be honest ", "i have to say ", "i'm scared because "]

    for text, label in examples:
        factor = safe_factor if label == 0 else crisis_factor
        augmented.append((text, label))
        for i in range(factor - 1):
            prefix = prefixes[hash(text + str(i)) % len(prefixes)]
            augmented.append((prefix + text, label))

    return augmented


class SafetyDataset(Dataset):
    """Dataset for safety classification."""

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        self.labels = labels
        self.encodings = tokenizer(
            texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def main(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training safety classifier on: {device}")

    # ---- Data preparation ----
    all_examples = augment_safety_examples(SAFETY_EXAMPLES)
    texts, labels = zip(*all_examples)
    labels_list = list(labels)

    tokenizer = AutoTokenizer.from_pretrained(settings.SAFETY_BASE_MODEL)
    full_dataset = SafetyDataset(list(texts), labels_list, tokenizer)

    val_size = int(len(full_dataset) * 0.2)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    # ---- Class-weighted loss for imbalance ----
    # Compute weights inversely proportional to class frequency
    class_weights = compute_class_weight(
        "balanced",
        classes=np.unique(labels_list),
        y=labels_list,
    )
    weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    logger.info(f"Class weights: {dict(zip(settings.SAFETY_LABELS, class_weights.round(2)))}")

    # ---- Model ----
    model = AutoModelForSequenceClassification.from_pretrained(
        settings.SAFETY_BASE_MODEL, num_labels=settings.SAFETY_NUM_LABELS,
        ignore_mismatched_sizes=True,
    )
    model = model.to(device)

    # ---- Optimizer + Scheduler ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.LEARNING_RATE, weight_decay=settings.WEIGHT_DECAY)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * 0.1), total_steps
    )

    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    early_stopper = EarlyStopping(patience=settings.EARLY_STOPPING_PATIENCE, mode="max")
    save_dir = settings.safety_model_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = 0.0

    for epoch in range(1, args.epochs + 1):
        logger.info(f"\nEpoch {epoch}/{args.epochs}")

        # Train
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings.GRADIENT_CLIP_NORM)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        logger.info(f"Train Loss: {total_loss/len(train_loader):.4f}")

        # Evaluate
        model.eval()
        all_preds, all_labels_eval = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels_batch = batch["labels"]
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
                all_preds.extend(preds.tolist())
                all_labels_eval.extend(labels_batch.numpy().tolist())

        f1 = f1_score(all_labels_eval, all_preds, average="macro", zero_division=0)
        logger.info(f"Val Macro-F1: {f1:.4f}")
        logger.info("\n" + classification_report(all_labels_eval, all_preds, target_names=settings.SAFETY_LABELS, zero_division=0))

        if f1 > best_f1:
            best_f1 = f1
            model.save_pretrained(str(save_dir))
            tokenizer.save_pretrained(str(save_dir))
            logger.info(f"✓ Saved best safety model (F1={best_f1:.4f})")

        if early_stopper(f1):
            logger.info("Early stopping triggered")
            break

    logger.info(f"\nTraining complete! Best Val F1: {best_f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DistilBERT safety classifier")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    main(args)
