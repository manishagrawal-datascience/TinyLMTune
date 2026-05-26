"""
Token length analyzer: profiles user data through the TinyBERT tokenizer
to determine the optimal max_len value.

Instead of guessing max_len=128, this module tokenizes a sample of the
data WITHOUT truncation and picks max_len based on real percentiles:

    p50  — covers half the data (fast, some truncation)
    p75  — covers most data (good balance)
    p95  — covers nearly all (minimal truncation)
    p100 — covers everything (no truncation, slowest)
"""

import logging
import math
from collections import Counter

from transformers import AutoTokenizer

from tinylmtune._internal.constants import TINYBERT_MODEL

logger = logging.getLogger(__name__)

# TinyBERT absolute max
_MODEL_MAX_LEN = 512

# Snap to GPU-friendly sizes
_STANDARD_LENGTHS = [2, 4, 8, 16, 24, 32, 64, 96, 128, 192, 256, 384, 512]


def _snap_up(value: int) -> int:
    """Round up to the nearest standard length."""
    for std in _STANDARD_LENGTHS:
        if std >= value:
            return std
    return _MODEL_MAX_LEN


def _extract_texts(records: list[dict], task: str) -> list[str]:
    """Pull the text field(s) from records based on task."""
    texts = []
    for rec in records:
        if task == "qna":
            q = rec.get("question", "")
            a = rec.get("answer", "")
            texts.append(f"{q} {a}")
        elif task == "generation":
            p = rec.get("prompt", "")
            c = rec.get("completion", "")
            texts.append(f"{p} {c}")
        elif task == "summarization":
            texts.append(rec.get("text", ""))
        else:
            texts.append(rec.get("text", ""))
    return [t for t in texts if t.strip()]


def analyze_token_lengths(
    records: list[dict],
    task: str,
    model_name: str = TINYBERT_MODEL,
) -> dict:
    """
    Tokenize data and return a length profile with recommendations.

    Returns dict with:
        min, max, mean, median, p50, p75, p90, p95, p100,
        truncation_pct, recommended_max_len (int), recommended_ga_choices (list)
    """
    texts = _extract_texts(records, task)
    if not texts:
        logger.warning("No texts to analyze — using default max_len=128")
        return {"recommended_max_len": 128, "recommended_ga_choices": [64, 128, 256]}

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize WITHOUT truncation to get true lengths
    lengths = []
    for t in texts:
        tokens = tokenizer(t, truncation=False, padding=False)
        lengths.append(len(tokens["input_ids"]))

    lengths.sort()
    n = len(lengths)

    def _pct(p):
        idx = int(math.ceil(p / 100.0 * n)) - 1
        return lengths[max(0, min(idx, n - 1))]

    profile = {
        "n_samples": n,
        "min": lengths[0],
        "max": lengths[-1],
        "mean": int(sum(lengths) / n),
        "median": lengths[n // 2],
        "p50": _pct(50),
        "p75": _pct(75),
        "p90": _pct(90),
        "p95": _pct(95),
        "p100": lengths[-1],
    }

    # Truncation percentages at standard lengths
    truncation = {}
    for std in _STANDARD_LENGTHS:
        trunc_count = sum(1 for l in lengths if l > std)
        truncation[std] = round(trunc_count / n * 100, 1)
    profile["truncation_pct"] = truncation

    # ── Pick the best fixed max_len ─────────────────────────────────────
    # Use p95: covers 95% of data, good balance of speed vs coverage
    recommended = _snap_up(profile["p95"])
    recommended = min(recommended, _MODEL_MAX_LEN)
    profile["recommended_max_len"] = recommended

    # ── Pick GA search choices ──────────────────────────────────────────
    # 3-4 data-driven values spanning the useful range
    candidates = set()
    candidates.add(_snap_up(profile["p75"]))
    candidates.add(_snap_up(profile["p95"]))

    p50_snap = _snap_up(profile["p50"])
    if p50_snap < _snap_up(profile["p75"]):
        candidates.add(p50_snap)

    p100_snap = min(_snap_up(profile["p100"]), _MODEL_MAX_LEN)
    if p100_snap > _snap_up(profile["p95"]):
        candidates.add(p100_snap)

    if len(candidates) < 2:
        candidates.add(_snap_up(max(32, p50_snap // 2)))

    ga_choices = sorted(set(min(v, _MODEL_MAX_LEN) for v in candidates))
    profile["recommended_ga_choices"] = ga_choices

    logger.info(
        "Token analysis: n=%d min=%d mean=%d p95=%d max=%d → max_len=%d, ga_choices=%s",
        n, profile["min"], profile["mean"], profile["p95"],
        profile["max"], recommended, ga_choices,
    )

    return profile


def print_token_analysis(
    records: list[dict],
    task: str,
    model_name: str = TINYBERT_MODEL,
) -> dict:
    """
    Print a human-readable token length report. Call before optimize_slm().

    Usage:
        from tinylmtune import print_token_analysis
        print_token_analysis(my_data, task="classification")
    """
    p = analyze_token_lengths(records, task, model_name)

    print("=" * 60)
    print(f"  tinyLMTune — Token Length Analysis")
    print(f"  Task: {task}  |  Samples: {p['n_samples']}")
    print("=" * 60)
    print(f"\n  Token lengths:  min={p['min']}  mean={p['mean']}  "
          f"median={p['median']}  max={p['max']}")
    print(f"\n  Percentiles:")
    for pct in [50, 75, 90, 95, 100]:
        print(f"    p{pct:3d}: {p[f'p{pct}']:4d} tokens")
    print(f"\n  Truncation at standard lengths:")
    for length, pct in p["truncation_pct"].items():
        bar = "█" * int(pct / 2)
        tag = " ← recommended" if length == p["recommended_max_len"] else ""
        print(f"    max_len={length:3d}: {pct:5.1f}% truncated  {bar}{tag}")
    print(f"\n  Recommended max_len: {p['recommended_max_len']}")
    print(f"  GA search choices:   {p['recommended_ga_choices']}\n")
    return p
