"""
Search space advisor: recommends GA hyperparameter ranges based on
task type and dataset size for all 11 parameters.
"""

import logging
from copy import deepcopy

from tinylmtune._internal.constants import GA_SEARCH_SPACE

logger = logging.getLogger(__name__)


def recommend_search_space(
    n_samples: int,
    task: str,
    max_len: int = 128,
) -> dict:
    """
    Recommend a search space for all 11 GA parameters.

    Adapts ranges based on dataset size — small datasets get more
    regularisation, larger LR ranges, more epochs.
    """
    rec = deepcopy(GA_SEARCH_SPACE)

    # ── Learning rate ─────────────────────────────────────────────
    if n_samples < 100:
        rec["learning_rate"] = (1e-5, 2e-4)
    elif n_samples < 500:
        rec["learning_rate"] = (1e-5, 5e-4)
    elif n_samples < 2000:
        rec["learning_rate"] = (5e-6, 5e-4)
    else:
        rec["learning_rate"] = (5e-6, 1e-3)

    # ── Batch size ────────────────────────────────────────────────
    max_sensible = max(4, n_samples // 3)
    batch_opts = [b for b in [4, 8, 16, 32] if b <= max_sensible]
    if len(batch_opts) < 2:
        batch_opts = [4, 8]
    rec["batch_size"] = batch_opts

    # ── Epochs ────────────────────────────────────────────────────
    if n_samples < 100:
        rec["epochs"] = (5, 20)
    elif n_samples < 500:
        rec["epochs"] = (3, 15)
    elif n_samples < 2000:
        rec["epochs"] = (2, 10)
    else:
        rec["epochs"] = (2, 8)

    # ── Warmup ────────────────────────────────────────────────────
    rec["warmup_ratio"] = (0.0, 0.2) if n_samples < 200 else (0.0, 0.3)

    # ── Weight decay ──────────────────────────────────────────────
    if n_samples < 200:
        rec["weight_decay"] = (0.0, 0.15)
    elif n_samples < 1000:
        rec["weight_decay"] = (0.0, 0.1)
    else:
        rec["weight_decay"] = (0.0, 0.05)

    # ── Dropout — more for small datasets ─────────────────────────
    if n_samples < 100:
        rec["dropout"] = (0.1, 0.4)
        rec["attention_dropout"] = (0.1, 0.4)
    elif n_samples < 500:
        rec["dropout"] = (0.05, 0.3)
        rec["attention_dropout"] = (0.05, 0.3)
    else:
        rec["dropout"] = (0.0, 0.2)
        rec["attention_dropout"] = (0.0, 0.2)

    # ── Gradient accumulation ─────────────────────────────────────
    # Useful when batch_size is small (simulates larger batches)
    if n_samples < 200:
        rec["gradient_accumulation_steps"] = [1, 2, 4]
    else:
        rec["gradient_accumulation_steps"] = [1, 2, 4, 8]

    # ── LR scheduler ─────────────────────────────────────────────
    rec["lr_scheduler_type"] = ["linear", "cosine", "cosine_with_restarts", "constant_with_warmup"]

    # ── Label smoothing — more for small datasets ─────────────────
    if n_samples < 200:
        rec["label_smoothing"] = (0.0, 0.2)
    else:
        rec["label_smoothing"] = (0.0, 0.1)

    # ── Max grad norm ─────────────────────────────────────────────
    rec["max_grad_norm"] = (0.5, 5.0)

    return rec


def print_recommendation(
    n_samples: int,
    task: str,
    max_len: int = 128,
    user_overrides: dict | None = None,
) -> dict:
    """
    Print a human-readable search space recommendation.
    Call before optimize_slm() to sanity-check your setup.
    """
    rec = recommend_search_space(n_samples, task, max_len)

    print("=" * 65)
    print(f"  tinyLMTune — Search Space Recommendation")
    print(f"  Task: {task}  |  Samples: {n_samples}  |  Max len: {max_len}")
    print(f"  Parameters: {len(rec)}")
    print("=" * 65)
    print("\n  Recommended search_space:")
    for key, val in rec.items():
        print(f"    {key:32s}: {val}")

    if user_overrides:
        print("\n  Your overrides:")
        for key, val in user_overrides.items():
            rec_val = rec.get(key, "N/A")
            marker = "✓" if val == rec_val else "⚠ differs"
            print(f"    {key:32s}: {val}  {marker}")
            if val != rec_val:
                print(f"      {'':32s}  rec: {rec_val}")

    print()
    return rec
