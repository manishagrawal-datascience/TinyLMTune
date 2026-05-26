"""
Raw text converter: takes plain strings and produces structured records.

Uses a local HuggingFace model (Flan-T5) for labelling — no Ollama needed.
For generation/summarization, uses heuristics (no LLM needed at all).
"""

import json
import logging
import re
from pathlib import Path

from tinylmtune._internal.cleaner import clean_text
from tinylmtune._internal.llm_backend import generate_text, generate_batch, unload_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Splitting helpers
# ---------------------------------------------------------------------------

def _split_block(text: str, strategy: str = "sentence") -> list[str]:
    text = clean_text(text)
    if not text:
        return []
    if strategy == "paragraph":
        chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
    elif strategy == "line":
        chunks = [c.strip() for c in text.splitlines() if c.strip()]
    else:
        chunks = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    return [c for c in chunks if len(c) >= 10]


def _load_txt_file(path: str | Path) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return p.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Heuristic converters (no LLM needed)
# ---------------------------------------------------------------------------

def _simple_generation_split(texts: list[str]) -> list[dict]:
    """Split each text into prompt/completion pairs."""
    records = []
    for t in texts:
        words = t.split()
        if len(words) < 4:
            continue
        mid = max(len(words) // 2, 2)
        records.append({
            "prompt": " ".join(words[:mid]),
            "completion": " ".join(words[mid:]),
        })
    return records


def _simple_summarization_split(texts: list[str]) -> list[dict]:
    """Use first half as summary of the full text."""
    records = []
    for t in texts:
        words = t.split()
        if len(words) < 10:
            continue
        mid = max(len(words) // 3, 5)
        records.append({
            "text": t,
            "summary": " ".join(words[:mid]),
        })
    return records


# ---------------------------------------------------------------------------
# LLM-based labelling (using Flan-T5)
# ---------------------------------------------------------------------------

def _label_classification(
    texts: list[str],
    labels: str | None = None,
    llm_model: str = "google/flan-t5-small",
) -> list[dict]:
    """Classify each text using Flan-T5."""
    label_list = [l.strip() for l in (labels or "positive,negative").split(",")]
    label_str = ", ".join(label_list)

    prompts = [
        f"Classify the following text as one of [{label_str}]. "
        f"Reply with only the label.\nText: {t}"
        for t in texts
    ]

    results = generate_batch(prompts, model_name=llm_model, max_new_tokens=20)
    records = []
    for text, result in zip(texts, results):
        predicted = result.strip().lower()
        # Find best matching label
        matched = None
        for lbl in label_list:
            if lbl.lower() in predicted:
                matched = lbl
                break
        if matched is None:
            matched = label_list[0]  # fallback to first label
        records.append({"text": text, "label": matched})

    return records


def _label_qna(
    texts: list[str],
    llm_model: str = "google/flan-t5-small",
) -> list[dict]:
    """Generate Q&A pairs from texts using Flan-T5."""
    prompts = [
        f"Generate a question that this text answers. "
        f"Reply with only the question.\nText: {t}"
        for t in texts
    ]
    questions = generate_batch(prompts, model_name=llm_model, max_new_tokens=50)

    records = []
    for text, question in zip(texts, questions):
        q = question.strip()
        if len(q) >= 5:
            records.append({"question": q, "answer": text})
    return records


def _label_ner(
    texts: list[str],
    llm_model: str = "google/flan-t5-small",
) -> list[dict]:
    """Extract entities from text using Flan-T5."""
    records = []
    for text in texts:
        prompt = (
            f"Extract all named entities from this text. "
            f"List each entity with its type (PER, ORG, LOC, MISC).\n"
            f"Text: {text}\n"
            f"Entities:"
        )
        raw = generate_text(prompt, model_name=llm_model, max_new_tokens=100)

        # Parse simple entity mentions from the response
        entities = []
        for entity_type in ["PER", "ORG", "LOC", "MISC"]:
            if entity_type in raw:
                # Try to find entity names mentioned near the type label
                parts = raw.split(entity_type)
                for part in parts[:-1]:
                    # Look for the last word/phrase before the type label
                    words = part.strip().rstrip(":,-").strip().split()
                    if words:
                        entity_text = words[-1].strip("(),.:;")
                        if entity_text and entity_text in text:
                            start = text.find(entity_text)
                            entities.append({
                                "text": entity_text,
                                "label": entity_type,
                                "start": start,
                                "end": start + len(entity_text),
                            })

        records.append({"text": text, "entities": entities})

    return records


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def convert_raw_text(
    raw_input: str | list[str],
    task: str,
    labels: str | None = None,
    split_strategy: str = "sentence",
    use_llm: bool = True,
    llm_model: str = "google/flan-t5-small",
    # Legacy params (ignored, kept for backwards compatibility)
    use_ollama: bool = True,
    ollama_model: str = "mistral",
) -> list[dict]:
    """
    Convert raw text into structured records for tinyLMTune training.

    Uses a local HuggingFace model (Flan-T5) for labelling.
    No Ollama or external servers needed.

    Parameters
    ----------
    raw_input : str | list[str]
        Text block, list of strings, or path to .txt file.
    task : str
        classification | summarization | qna | generation | ner
    labels : str | None
        Comma-separated labels for classification.
    split_strategy : str
        "sentence" | "paragraph" | "line"
    use_llm : bool
        If True, use Flan-T5 for labelling. If False, use heuristics
        (works for generation and summarization only).
    llm_model : str
        HuggingFace model name (default: google/flan-t5-small).

    Returns
    -------
    list[dict] — structured records.
    """
    # ── Resolve input into list[str] ─────────────────────────────────
    if isinstance(raw_input, list):
        texts = [clean_text(t) for t in raw_input if isinstance(t, str)]
    elif isinstance(raw_input, str):
        if len(raw_input) < 500 and Path(raw_input).is_file():
            logger.info("Loading raw text from file: %s", raw_input)
            content = _load_txt_file(raw_input)
            texts = _split_block(content, strategy=split_strategy)
        else:
            texts = _split_block(raw_input, strategy=split_strategy)
    else:
        raise TypeError(
            f"raw_input must be str or list[str]. Got: {type(raw_input).__name__}"
        )

    texts = [t for t in texts if t and len(t) >= 10]
    if not texts:
        raise ValueError("No usable text found after cleaning/splitting.")

    logger.info("Converted raw input → %d text chunks", len(texts))

    # ── Convert to structured records ────────────────────────────────
    # Tasks that can use heuristics (no LLM)
    if task == "generation":
        records = _simple_generation_split(texts)
    elif task == "summarization" and not use_llm:
        records = _simple_summarization_split(texts)
    # Tasks that need LLM labelling
    elif not use_llm:
        raise ValueError(
            f"use_llm=False only supports 'generation' and 'summarization'. "
            f"For '{task}', set use_llm=True or provide pre-labelled data."
        )
    elif task == "classification":
        records = _label_classification(texts, labels, llm_model)
    elif task == "summarization":
        records = _simple_summarization_split(texts)  # heuristic is fine
    elif task == "qna":
        records = _label_qna(texts, llm_model)
    elif task == "ner":
        records = _label_ner(texts, llm_model)
    else:
        raise ValueError(f"Unknown task: {task}")

    # Free GPU memory for training
    unload_model(llm_model)

    if not records:
        raise RuntimeError(
            f"Labelling produced 0 valid records from {len(texts)} texts. "
            f"Try a larger model (google/flan-t5-base) or provide structured data."
        )

    logger.info("Produced %d structured records from raw text", len(records))
    return records
