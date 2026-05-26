"""
Shared constants for tinyLMTune internals.
"""

from transformers import (
    AutoModelForMaskedLM,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
)

TINYBERT_MODEL = "huawei-noah/TinyBERT_General_4L_312D"

SUPPORTED_TASKS = ("classification", "summarization", "generation", "qna", "ner")

TASK_HEAD = {
    "classification": AutoModelForSequenceClassification,
    "summarization":  AutoModelForMaskedLM,
    "generation":     AutoModelForMaskedLM,
    "qna":            AutoModelForQuestionAnswering,
    "ner":            AutoModelForTokenClassification,
}

# GA hyperparameter search space boundaries
GA_SEARCH_SPACE = {
    "learning_rate":               (1e-5, 5e-4),
    "batch_size":                  [4, 8, 16, 32],
    "epochs":                      (2, 10),
    "warmup_ratio":                (0.0, 0.3),
    "weight_decay":                (0.0, 0.1),
    "dropout":                     (0.0, 0.3),
    "attention_dropout":           (0.0, 0.3),
    "gradient_accumulation_steps": [1, 2, 4, 8],
    "lr_scheduler_type":           ["linear", "cosine", "cosine_with_restarts", "constant_with_warmup"],
    "label_smoothing":             (0.0, 0.2),
    "max_grad_norm":               (0.5, 5.0),
}
