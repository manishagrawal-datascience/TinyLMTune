"""
Synthetic corpus generation using a local HuggingFace model (Flan-T5).
No Ollama or external servers needed.

Strategy for small models (flan-t5-small/base):
  - Use simple, focused prompts — one instruction per call
  - For complex tasks (QnA, NER), use multi-step generation
  - Never ask the model to produce structured formats (JSON, multi-field)
"""

import json
import logging
import re
from pathlib import Path

from tinylmtune._internal.cleaner import clean_text
from tinylmtune._internal.llm_backend import generate_text, unload_model

logger = logging.getLogger(__name__)


# ── Simple single-shot prompts (classification, generation) ───────
_SINGLE_PROMPTS = {
    "classification": "Write a {label} review about {topic}:",
    "generation": "Write a fact about {topic}:",
    "summarization_text": "Write a paragraph about {topic}:",
    "summarization_summary": "Summarize this in one sentence: {text}",
    "qna_context": "Write a short factual paragraph about {topic}:",
    "qna_question": "Ask a question about this text: {context}",
    "qna_answer": "Answer this question based on the text. Text: {context} Question: {question} Answer:",
    "ner_sentence": (
        "Write a sentence about {topic} that mentions a person, "
        "a company, and a city:"
    ),
}


# ── Parsers ───────────────────────────────────────────────────────

def _parse_classification(text: str, label: str) -> dict | None:
    text = text.strip().strip('"').strip()
    if len(text) < 10:
        return None
    return {"text": text, "label": label}


def _parse_generation(text: str) -> dict | None:
    words = text.strip().split()
    if len(words) < 6:
        return None
    mid = max(len(words) // 3, 3)
    return {"prompt": " ".join(words[:mid]), "completion": " ".join(words[mid:])}


def _extract_entities_from_text(text: str) -> list[dict]:
    """Simple heuristic NER: find capitalized multi-word spans."""
    entities = []
    # Pattern: sequences of capitalized words (2+ chars each)
    pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
    for match in re.finditer(pattern, text):
        span = match.group(0)
        # Skip if it's the start of a sentence (check if preceded by '. ' or start)
        start = match.start()
        if start == 0 or text[start - 2:start] == '. ':
            # Could be sentence start, only keep if 2+ words
            if ' ' not in span:
                continue
        # Guess label based on common patterns
        if any(w in span.lower() for w in ['inc', 'corp', 'ltd', 'company', 'group', 'university']):
            label = "ORG"
        elif any(w in span.lower() for w in ['city', 'river', 'mountain', 'street', 'park']):
            label = "LOC"
        elif len(span.split()) <= 3:
            # Short capitalized spans — alternate between PER, ORG, LOC
            label = ["PER", "ORG", "LOC"][len(entities) % 3]
        else:
            label = "MISC"
        entities.append({
            "text": span,
            "label": label,
            "start": start,
            "end": start + len(span),
        })
    return entities


# ── Multi-step generators ─────────────────────────────────────────

def _generate_qna_example(topic: str, llm_model: str) -> dict | None:
    """Generate QnA in 3 steps: context → question → answer."""
    # Step 1: Generate a context paragraph
    context = generate_text(
        _SINGLE_PROMPTS["qna_context"].format(topic=topic),
        model_name=llm_model, max_new_tokens=150,
    )
    context = clean_text(context)
    if len(context) < 30:
        return None

    # Step 2: Generate a question about the context
    question = generate_text(
        _SINGLE_PROMPTS["qna_question"].format(context=context),
        model_name=llm_model, max_new_tokens=50,
    )
    question = clean_text(question)
    if len(question) < 5:
        return None

    # Step 3: Generate an answer grounded in the context
    answer = generate_text(
        _SINGLE_PROMPTS["qna_answer"].format(context=context, question=question),
        model_name=llm_model, max_new_tokens=50,
    )
    answer = clean_text(answer)
    if len(answer) < 1:
        return None

    # Verify the answer (or a close match) appears in the context
    if answer.lower() not in context.lower():
        # Try to find the best substring match
        words = answer.split()
        for length in range(len(words), 0, -1):
            for start in range(len(words) - length + 1):
                substr = " ".join(words[start:start + length])
                if substr.lower() in context.lower():
                    answer = substr
                    break
            else:
                continue
            break
        else:
            # Last resort: take first noun phrase from context as answer
            context_words = context.split()
            if len(context_words) >= 3:
                answer = " ".join(context_words[:3])
            else:
                return None

    return {"question": question, "context": context, "answer": answer}


def _generate_summarization_example(topic: str, llm_model: str) -> dict | None:
    """Generate summarization in 2 steps: text → summary."""
    # Step 1: Generate text
    text = generate_text(
        _SINGLE_PROMPTS["summarization_text"].format(topic=topic),
        model_name=llm_model, max_new_tokens=150,
    )
    text = clean_text(text)
    if len(text) < 30:
        return None

    # Step 2: Generate summary of that text
    summary = generate_text(
        _SINGLE_PROMPTS["summarization_summary"].format(text=text),
        model_name=llm_model, max_new_tokens=60,
    )
    summary = clean_text(summary)
    if len(summary) < 10:
        # Fallback: use first sentence
        sentences = text.split('.')
        summary = sentences[0].strip() + '.' if sentences else text[:50]

    return {"text": text, "summary": summary}


def _generate_ner_example(topic: str, llm_model: str) -> dict | None:
    """Generate NER: sentence → extract entities via heuristics."""
    text = generate_text(
        _SINGLE_PROMPTS["ner_sentence"].format(topic=topic),
        model_name=llm_model, max_new_tokens=80,
    )
    text = clean_text(text)
    if len(text) < 15:
        return None

    entities = _extract_entities_from_text(text)
    # Only keep if we found at least 1 entity
    if not entities:
        return None

    return {"text": text, "entities": entities}


# ── Main entry point ──────────────────────────────────────────────

def generate_corpus(
    task: str,
    topic: str = "general knowledge",
    n_examples: int = 200,
    labels: str | None = None,
    output_path: str = "corpus.jsonl",
    llm_model: str = "google/flan-t5-small",
) -> Path:
    """
    Generate a synthetic training corpus using a HuggingFace model.

    Uses multi-step generation for complex tasks (QnA, summarization, NER)
    to work reliably with small models like flan-t5-small.

    Parameters
    ----------
    task : str
        classification | summarization | qna | generation | ner
    topic : str
        Topic hint for generation.
    n_examples : int
        Number of examples to generate.
    labels : str | None
        Comma-separated labels for classification.
    output_path : str
        Where to save the JSONL file.
    llm_model : str
        HuggingFace model name (default: google/flan-t5-small).

    Returns
    -------
    Path to the generated JSONL file.
    """
    if task not in ("classification", "summarization", "qna", "generation", "ner"):
        raise ValueError(f"Unknown task '{task}'")

    label_list = [l.strip() for l in (labels or "positive,negative").split(",")]
    records = []

    logger.info("Generating %d %s examples about '%s' using %s ...",
                n_examples, task, topic, llm_model)

    for i in range(n_examples):
        try:
            rec = None

            if task == "classification":
                label = label_list[i % len(label_list)]
                prompt = _SINGLE_PROMPTS["classification"].format(topic=topic, label=label)
                raw = clean_text(generate_text(prompt, model_name=llm_model, max_new_tokens=100))
                rec = _parse_classification(raw, label)

            elif task == "generation":
                prompt = _SINGLE_PROMPTS["generation"].format(topic=topic)
                raw = clean_text(generate_text(prompt, model_name=llm_model, max_new_tokens=100))
                rec = _parse_generation(raw)

            elif task == "qna":
                rec = _generate_qna_example(topic, llm_model)

            elif task == "summarization":
                rec = _generate_summarization_example(topic, llm_model)

            elif task == "ner":
                rec = _generate_ner_example(topic, llm_model)

            if rec:
                records.append(rec)

        except Exception as exc:
            logger.warning("Example %d failed: %s", i + 1, exc)
            continue

        if (i + 1) % 50 == 0:
            logger.info("Generated %d / %d examples (%d valid so far)",
                        i + 1, n_examples, len(records))

    # Free GPU memory for training
    unload_model(llm_model)

    if not records:
        raise RuntimeError(
            f"Corpus generation produced 0 valid records from {n_examples} attempts. "
            f"Try a larger model (google/flan-t5-base) or provide data directly."
        )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    logger.info("Saved %d records → %s", len(records), out)
    return out
