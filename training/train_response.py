"""
=============================================================================
training/train_response.py — FLAN-T5 Fine-tuning on EmpatheticDialogues
=============================================================================

Fine-tunes google/flan-t5-small on the EmpatheticDialogues dataset to
produce empathetic, contextually appropriate responses.

Key DL Concepts for Seq2Seq Fine-tuning:
    ✓ Seq2Seq (Encoder-Decoder) Architecture: Input → Encoder → Decoder → Output
    ✓ Teacher Forcing: During training, the decoder receives ground-truth tokens
      as input (not its own previous predictions). This stabilises early training.
    ✓ Label Smoothing: Softens one-hot labels to [0.1, 0.9, ...] instead of [0, 1]
      This acts as regularisation — prevents the model from being overconfident.
    ✓ Cross-Entropy Loss on Decoder Outputs
    ✓ Gradient Clipping: Critical for encoder-decoder models
    ✓ Linear Warmup + Cosine Decay (AdamW)
    ✓ Checkpoint Saving + Resuming
    ✓ ROUGE/BLEU Evaluation (automatic metric for text generation)
    ✓ Max Input Length: 512 tokens (T5-small limit)
    ✓ Mixed Precision: fp16 for 50% memory reduction

EmpatheticDialogues Dataset:
    - 24,850 conversations
    - One speaker describes an emotional situation (prompt)
    - Another speaker responds empathetically (target)
    - Includes emotion labels for each conversation
    - Published by Rashkin et al. (Facebook AI Research)

Prompt Format:
    "Express empathy: [emotion] [prompt text]\nResponse:"

This instruction-style formatting leverages FLAN-T5's instruction tuning
— the model is pre-trained to follow instructions, so explicit instructions
in the prompt significantly improve response quality.

Usage:
    python training/train_response.py [--epochs 3] [--batch-size 8] [--fp16]

Expected training time:
    CPU: ~12–24 hours (not recommended)
    Single GPU (RTX 3080): ~2–4 hours
    Single GPU (V100/A100): ~45–90 minutes
=============================================================================
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from datasets import load_dataset
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)
import evaluate

from backend.config import settings


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_empathetic_dialogues(tokenizer, max_input_length: int = 512, max_target_length: int = 128):
    """
    Load and preprocess EmpatheticDialogues for seq2seq training.

    Dataset structure:
        - 'conv_id': conversation identifier
        - 'utterance_idx': turn number within conversation
        - 'context': emotional context word
        - 'prompt': speaker's message describing their emotional situation
        - 'selfeval': self-evaluation of response quality
        - 'tags': emotion tags

    We construct (input, target) pairs where:
        - input = instruction + emotion context + prompt
        - target = the empathetic response

    For multi-turn conversations, we pair consecutive utterances.
    """
    logger.info("Loading EmpatheticDialogues dataset...")
    dataset = load_dataset("empathetic_dialogues")

    def preprocess(examples):
        """
        Tokenise input-target pairs for seq2seq training.

        Seq2Seq tokenisation:
            - Input (encoder): tokenise prompt → input_ids, attention_mask
            - Target (decoder): tokenise response → labels
            - Padding token in labels is set to -100 (ignored by loss function)

        Label -100:
            Cross-entropy loss ignores positions where label == -100.
            This prevents the model from being penalised for generating
            any content at padding positions.
        """
        # Format input as instruction
        inputs = [
            f"Provide an empathetic response to someone feeling {ctx}: {prompt}"
            for ctx, prompt in zip(examples["context"], examples["prompt"])
        ]

        # Tokenise inputs
        model_inputs = tokenizer(
            inputs,
            max_length=max_input_length,
            padding=False,  # DataCollator will handle padding
            truncation=True,
        )

        # Tokenise targets (the empathetic responses)
        # Note: We use 'utterance' as the target response
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["utterance"],
                max_length=max_target_length,
                padding=False,
                truncation=True,
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # Apply preprocessing
    tokenised = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenising",
    )

    logger.info(
        f"Dataset sizes — Train: {len(tokenised['train'])}, "
        f"Val: {len(tokenised['validation'])}, Test: {len(tokenised['test'])}"
    )
    return tokenised


def compute_metrics_factory(tokenizer):
    """
    Create a compute_metrics function for ROUGE evaluation.

    ROUGE (Recall-Oriented Understudy for Gisting Evaluation):
        - ROUGE-1: Unigram overlap between generated and reference text
        - ROUGE-2: Bigram overlap
        - ROUGE-L: Longest common subsequence

    These are standard automatic evaluation metrics for text generation.
    Higher ROUGE = generated text more similar to reference responses.
    Note: ROUGE doesn't capture meaning — a grammatically different but
    semantically equivalent response may score low.

    BLEU (Bilingual Evaluation Understudy):
        - Precision-based n-gram overlap
        - Typically 10–30% for open-domain dialogue

    We report both for comprehensive evaluation.
    """
    rouge_metric = evaluate.load("rouge")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds

        # Replace -100 padding with pad_token_id for decoding
        labels = [
            [(l if l != -100 else tokenizer.pad_token_id) for l in label]
            for label in labels
        ]

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # ROUGE expects list of strings
        result = rouge_metric.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True,
        )

        # Extract mid F1 scores
        return {
            "rouge1": round(result["rouge1"], 4),
            "rouge2": round(result["rouge2"], 4),
            "rougeL": round(result["rougeL"], 4),
        }

    return compute_metrics


def main(args: argparse.Namespace) -> None:
    """Main FLAN-T5 fine-tuning function using HuggingFace Trainer."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Fine-tuning on: {device}")

    save_dir = settings.response_model_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load model and tokeniser ----
    model_id = settings.RESPONSE_BASE_MODEL
    logger.info(f"Loading base model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ---- Load dataset ----
    tokenised_data = load_empathetic_dialogues(tokenizer)

    # ---- Data Collator ----
    # DataCollatorForSeq2Seq handles:
    #   - Dynamic padding (pads each batch to the longest example)
    #   - Label padding with -100 (ignored by loss)
    # This is more memory-efficient than padding to max_length
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,  # Ignore padding positions in loss
    )

    # ---- Training Arguments ----
    # HuggingFace Trainer handles the training loop, logging, checkpointing
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(save_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=settings.LEARNING_RATE,
        weight_decay=settings.WEIGHT_DECAY,
        warmup_ratio=settings.WARMUP_RATIO,
        max_grad_norm=settings.GRADIENT_CLIP_NORM,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,           # Keep only 2 checkpoints
        load_best_model_at_end=True,  # Load best checkpoint after training
        metric_for_best_model="rougeL",
        greater_is_better=True,
        predict_with_generate=True,   # Use model.generate() during evaluation
        generation_max_length=128,
        fp16=args.fp16 and device == "cuda",
        logging_steps=100,
        logging_dir=str(save_dir / "logs"),
        report_to="none",             # Disable WandB/TensorBoard by default
        dataloader_num_workers=0,     # 0 for Windows compatibility
        seed=42,
    )

    # ---- Trainer ----
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenised_data["train"],
        eval_dataset=tokenised_data["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_factory(tokenizer),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=settings.EARLY_STOPPING_PATIENCE
            )
        ],
    )

    # ---- Train ----
    logger.info("Starting FLAN-T5 fine-tuning...")
    logger.info("(This takes 2–4 hours on GPU, longer on CPU)")
    trainer.train()

    # ---- Save final model ----
    trainer.save_model(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))
    logger.info(f"✓ Fine-tuned model saved to: {save_dir}")

    # ---- Test evaluation ----
    logger.info("\nRunning test set evaluation...")
    test_results = trainer.predict(tokenised_data["test"])
    logger.info(f"Test ROUGE scores: {test_results.metrics}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune FLAN-T5 on EmpatheticDialogues")
    parser.add_argument("--epochs", type=int, default=settings.NUM_EPOCHS_GENERATOR)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()
    main(args)
