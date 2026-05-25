"""
=============================================================================
tinyLMTune — Benchmark Testing
=============================================================================

Tests the framework against real HuggingFace datasets for each task.
No Ollama needed — pure structured data.

Usage:
    python benchmark_test.py                    # run all tasks
    python benchmark_test.py classification     # run one task
    python benchmark_test.py classification qna # run specific tasks
=============================================================================
"""

import logging
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("benchmark")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Benchmark dataset loaders                                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def load_classification_data(n=200):
    """Rotten Tomatoes — binary sentiment classification."""
    from datasets import load_dataset
    ds = load_dataset("rotten_tomatoes", split="train")
    ds = ds.shuffle(seed=42).select(range(min(n, len(ds))))

    label_map = {0: "negative", 1: "positive"}
    records = [{"text": r["text"], "label": label_map[r["label"]]} for r in ds]
    log.info("Loaded %d classification records (rotten_tomatoes)", len(records))
    return records


def load_summarization_data(n=200):
    """XSum — news article summarization."""
    from datasets import load_dataset
    ds = load_dataset("xsum", split="train")
    ds = ds.shuffle(seed=42).select(range(min(n, len(ds))))

    records = [{"text": r["document"][:500], "summary": r["summary"]} for r in ds]
    log.info("Loaded %d summarization records (xsum)", len(records))
    return records


def load_qna_data(n=200):
    """SQuAD — extractive question answering."""
    from datasets import load_dataset
    ds = load_dataset("squad", split="train")
    ds = ds.shuffle(seed=42).select(range(min(n, len(ds))))

    records = []
    for r in ds:
        answer = r["answers"]["text"][0] if r["answers"]["text"] else ""
        if not answer:
            continue
        records.append({
            "question": r["question"],
            "context": r["context"],
            "answer": answer,
        })
    log.info("Loaded %d QnA records (squad)", len(records))
    return records


def load_generation_data(n=200):
    """WikiText-2 — text generation (prompt/completion pairs)."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    records = []
    for r in ds:
        text = r["text"].strip()
        if len(text) < 30:
            continue
        words = text.split()
        mid = max(len(words) // 3, 3)
        records.append({
            "prompt": " ".join(words[:mid]),
            "completion": " ".join(words[mid:]),
        })
        if len(records) >= n:
            break
    log.info("Loaded %d generation records (wikitext-2)", len(records))
    return records


def load_ner_data(n=200):
    """CoNLL-2003 — named entity recognition."""
    from datasets import load_dataset
    ds = load_dataset("conll2003", split="train", trust_remote_code=True)
    ds = ds.shuffle(seed=42).select(range(min(n, len(ds))))

    # CoNLL NER tag mapping
    tag_names = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG",
                 "B-LOC", "I-LOC", "B-MISC", "I-MISC"]

    records = []
    for r in ds:
        tokens = r["tokens"]
        ner_tags = r["ner_tags"]
        text = " ".join(tokens)

        # Build entity spans from BIO tags
        entities = []
        current = None
        char_pos = 0

        for tok, tag_id in zip(tokens, ner_tags):
            tag = tag_names[tag_id] if tag_id < len(tag_names) else "O"
            tok_start = text.find(tok, char_pos)
            tok_end = tok_start + len(tok)

            if tag.startswith("B-"):
                if current:
                    entities.append(current)
                current = {
                    "text": tok,
                    "label": tag[2:],
                    "start": tok_start,
                    "end": tok_end,
                }
            elif tag.startswith("I-") and current and tag[2:] == current["label"]:
                current["text"] = text[current["start"]:tok_end]
                current["end"] = tok_end
            else:
                if current:
                    entities.append(current)
                    current = None

            char_pos = tok_end

        if current:
            entities.append(current)

        records.append({"text": text, "entities": entities})

    log.info("Loaded %d NER records (conll2003)", len(records))
    return records


BENCHMARK_LOADERS = {
    "classification": load_classification_data,
    "summarization":  load_summarization_data,
    "qna":            load_qna_data,
    "generation":     load_generation_data,
    "ner":            load_ner_data,
}


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Run benchmark                                                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def run_benchmark(
    task: str,
    n_samples: int = 200,
    pop_size: int = 4,
    generations: int = 2,
):
    """Run tinyLMTune on a benchmark dataset for the given task."""
    from tinylmtune import optimize_slm, TinyInference, print_token_analysis

    print("=" * 65)
    print(f"  BENCHMARK: {task}")
    print("=" * 65)

    # ── Load benchmark data ──────────────────────────────────────────
    loader = BENCHMARK_LOADERS.get(task)
    if loader is None:
        print(f"  Unknown task: {task}")
        return None

    data = loader(n_samples)
    print(f"\n  Loaded {len(data)} records")
    print(f"  Sample: {data[0]}\n")

    # ── Token analysis ───────────────────────────────────────────────
    print_token_analysis(data, task=task)

    # ── Train ────────────────────────────────────────────────────────
    output_dir = f"benchmark_{task}"
    start = time.time()

    best = optimize_slm(
        task=task,
        user_data=data,
        pop_size=pop_size,
        generations=generations,
        output_dir=output_dir,
    )

    elapsed = time.time() - start

    print(f"\n  ✓ Training completed in {elapsed:.1f}s")
    print(f"  ✓ Best fitness: {best.get('fitness', 'N/A')}")
    print(f"  ✓ Best max_len: {best.get('max_len', 'N/A')}")
    print(f"  ✓ Best config:")
    for k, v in best.items():
        if k not in ("fitness", "output_dir", "max_len"):
            print(f"      {k}: {v}")

    # ── Inference test ───────────────────────────────────────────────
    model = TinyInference(output_dir)

    test_inputs = {
        "classification": "This movie was absolutely brilliant and well directed.",
        "summarization":  "The government announced new policies for renewable energy.",
        "generation":     "Artificial intelligence is",
        "ner":            "Elon Musk founded SpaceX in Los Angeles.",
    }

    if task == "qna":
        prediction = model.predict(
            "What is machine learning?",
            context="Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        )
    else:
        test_text = test_inputs.get(task, "Test input")
        prediction = model.predict(test_text)
    print(f"\n  ✓ Inference test:")
    if task == "qna":
        print(f"      Input:  What is machine learning? (with context)")
    else:
        print(f"      Input:  {test_text}")
    print(f"      Output: {prediction}")
    print()

    return best


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Main                                                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    # Parse task names from command line
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    tasks = args if args else ["classification"]  # default to classification

    # Optional flags
    n_samples = 200
    pop_size = 4
    generations = 2

    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--samples" and i + 2 < len(sys.argv):
            n_samples = int(sys.argv[i + 2])
        if arg == "--pop" and i + 2 < len(sys.argv):
            pop_size = int(sys.argv[i + 2])
        if arg == "--gens" and i + 2 < len(sys.argv):
            generations = int(sys.argv[i + 2])

    results = {}
    for task in tasks:
        if task in BENCHMARK_LOADERS:
            best = run_benchmark(task, n_samples, pop_size, generations)
            if best:
                results[task] = best
        else:
            print(f"Unknown task: {task}. Choose from: {list(BENCHMARK_LOADERS)}")

    # Summary
    if len(results) > 1:
        print("=" * 65)
        print("  BENCHMARK SUMMARY")
        print("=" * 65)
        for task, best in results.items():
            print(f"  {task:20s}: fitness={best.get('fitness', 'N/A'):.4f}  "
                  f"max_len={best.get('max_len', 'N/A')}")
        print()
