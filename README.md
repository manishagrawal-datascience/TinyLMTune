# tinyLMTune

**Genetic-Algorithm-Optimised TinyBERT Fine-Tuning in one function call.**

tinyLMTune automates the full pipeline: 
(1) dataset building  
(2) GA hyperparameter search 
(3) TinyBERT fine-tuning 
(4) model export 

Bring your own data or let it generate synthetic training data automatically.

## Installation

```bash
pip install tinylmtune
```

## Quick Start

### Option 1 — Bring your own data (recommended)

```python
from tinylmtune import optimize_slm, TinyInference

my_data = [
    {"text": "Loved this product!", "label": "positive"},
    {"text": "Broke after one day.", "label": "negative"},
    {"text": "Works fine, nothing special.", "label": "neutral"},
    # ... at least 50+ records recommended
]

best = optimize_slm(
    task="classification",
    user_data=my_data,
    output_dir="my_model",
)

model = TinyInference("my_model")
print(model.predict("Absolutely amazing quality!"))
```

### Option 2 — Use a JSONL file

```python
best = optimize_slm(
    task="classification",
    corpus_path="my_data.jsonl",   # one JSON object per line
    output_dir="my_model",
)
```

### Option 3 — Auto-generate synthetic data (default fallback)

If neither `user_data` nor `corpus_path` is provided, tinyLMTune generates
synthetic training data via a local Ollama/Mistral model:

```python
best = optimize_slm(
    task="classification",
    corpus_prompt="Generate movie-review sentiment examples",
    n_examples=200,
    output_dir="my_model",
)
```

## Data Format

Each record is a dict (or a JSON line in a `.jsonl` file). The required keys depend on the task:

| Task | Required keys | Example |
|------|---------------|---------|
| `classification` | `text`, `label` | `{"text": "Great film!", "label": "positive"}` |
| `summarization` | `text`, `summary` | `{"text": "Long article...", "summary": "Short version."}` |
| `qna` | `question`, `answer` | `{"question": "What is X?", "answer": "X is..."}` |
| `generation` | `prompt`, `completion` | `{"prompt": "Once upon a", "completion": "time there was..."}` |
| `ner` | `text`, `entities` | `{"text": "John in NYC", "entities": [{"text": "John", "label": "PER", "start": 0, "end": 4}]}` |

Records missing required keys are skipped with a warning. If all records are invalid, an error is raised showing the expected format.

## Public API

Only two symbols are exported:

| Symbol | Purpose |
|--------|---------|
| `optimize_slm()` | Full train pipeline — returns best config |
| `TinyInference` | Load & predict with a saved model |

All internal modules live in `tinylmtune._internal` and are **not** part of the public API.

## Project Structure

```
tinylmtune/
├── __init__.py              # Public API: optimize_slm, TinyInference
├── _internal/               # Private — do not import directly
│   ├── __init__.py
│   ├── constants.py         # Model names, task mappings, GA search space
│   ├── cleaner.py           # Text cleaning utilities
│   ├── corpus_gen.py        # Synthetic data generation (Ollama/Mistral)
│   ├── dataset.py           # list[dict] / JSONL → HuggingFace Dataset
│   ├── model_builder.py     # TinyBERT model instantiation
│   ├── trainer.py           # HuggingFace Trainer wrapper
│   ├── ga_optimizer.py      # Genetic algorithm hyperparameter search
│   ├── inference.py         # TinyInference class + save_best_model
│   └── pipeline.py          # optimize_slm() orchestrator
├── setup.py
└── README.md
```
