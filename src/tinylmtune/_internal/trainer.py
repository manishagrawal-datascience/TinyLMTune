import logging

import numpy as np
from datasets import Dataset
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score, f1_score

from .model_builder import build_model

logger = logging.getLogger(__name__)


def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted", zero_division=0),
    }


def _compute_qna_metrics(eval_pred):
    
    (start_logits, end_logits), (start_true, end_true) = eval_pred
    start_pred = np.argmax(start_logits, axis=-1)
    end_pred = np.argmax(end_logits, axis=-1)
    start_acc = np.mean(start_pred == start_true)
    end_acc = np.mean(end_pred == end_true)
    exact_match = np.mean((start_pred == start_true) & (end_pred == end_true))
    return {
        "exact_match": exact_match,
        "start_acc": start_acc,
        "end_acc": end_acc,
    }


def _compute_ner_metrics(eval_pred):
    
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    # Flatten and filter out -100 (ignored tokens)
    mask = labels.flatten() != -100
    true_flat = labels.flatten()[mask]
    pred_flat = preds.flatten()[mask]
    return {
        "token_f1": f1_score(true_flat, pred_flat, average="weighted", zero_division=0),
        "token_accuracy": accuracy_score(true_flat, pred_flat),
    }


def train_and_evaluate(
    train_ds: Dataset,
    val_ds: Dataset,
    task: str,
    num_labels: int = 2,
    # ── 11 GA-optimised parameters ──
    learning_rate: float = 3e-5,
    batch_size: int = 16,
    epochs: int = 5,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    dropout: float = 0.1,
    attention_dropout: float = 0.1,
    gradient_accumulation_steps: int = 1,
    lr_scheduler_type: str = "linear",
    label_smoothing: float = 0.0,
    max_grad_norm: float = 1.0,
    # ── Fixed parameters ──
    max_len: int = 128,
    output_dir: str = "./tmp_trainer",
    label2id: dict | None = None,
    id2label: dict | None = None,
) -> dict:
    
    model = build_model(
        task=task, num_labels=num_labels,
        label2id=label2id, id2label=id2label,
        dropout=dropout,
        attention_dropout=attention_dropout,
    )

    
    effective_label_smoothing = 0.0 if task == "qna" else label_smoothing

    if task == "classification":
        metric_name = "f1"
        greater = True
    elif task == "qna":
        metric_name = "exact_match"
        greater = True
    elif task == "ner":
        metric_name = "token_f1"
        greater = True
    else:
        metric_name = "eval_loss"
        greater = False

    args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        gradient_accumulation_steps=gradient_accumulation_steps,
        lr_scheduler_type=lr_scheduler_type,
        label_smoothing_factor=effective_label_smoothing,
        max_grad_norm=max_grad_norm,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        greater_is_better=greater,
        logging_strategy="epoch",
        report_to="none",
        fp16=False,
        remove_unused_columns=False,
    )

    if task == "classification":
        compute_fn = _compute_metrics
    elif task == "qna":
        compute_fn = _compute_qna_metrics
    elif task == "ner":
        compute_fn = _compute_ner_metrics
    else:
        compute_fn = None

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    logger.info(
        "Training: lr=%s bs=%d epochs=%d dropout=%.2f attn_drop=%.2f "
        "grad_accum=%d scheduler=%s label_smooth=%.3f grad_norm=%.1f",
        learning_rate, batch_size, epochs, dropout, attention_dropout,
        gradient_accumulation_steps, lr_scheduler_type, label_smoothing, max_grad_norm,
    )
    trainer.train()

    metrics = trainer.evaluate()
    logger.info("Eval metrics: %s", metrics)
    return {"model": model, "trainer": trainer, "metrics": metrics}
