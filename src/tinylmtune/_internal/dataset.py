import json
import logging
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer

from tinylmtune._internal.cleaner import clean_text
from tinylmtune._internal.constants import TINYBERT_MODEL

logger = logging.getLogger(__name__)


def _load_jsonl(path: str | Path) -> list[dict]:
    """Load and clean records from a JSONL file."""
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                obj = {k: clean_text(v) if isinstance(v, str) else v
                       for k, v in obj.items()}
                records.append(obj)
            except json.JSONDecodeError:
                continue
    return records


def _tokenise_classification(records, tokenizer, max_len, label2id):
    texts = [r["text"] for r in records]
    labels = [label2id.get(r.get("label", ""), 0) for r in records]
    enc = tokenizer(texts, truncation=True, padding="max_length", max_length=max_len)
    enc["labels"] = labels
    return Dataset.from_dict(enc)


def _tokenise_summarization(records, tokenizer, max_len):
    """Encode (text, summary) as sentence pair; labels = summary tokens only."""
    all_input_ids, all_attention, all_labels = [], [], []
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id

    for r in records:
        text = r.get("text", "")
        summary = r.get("summary", "")
        enc = tokenizer(text, summary, truncation=True, padding="max_length",
                        max_length=max_len)
        input_ids = enc["input_ids"]

        # Labels = copy of input_ids, but mask everything before first [SEP]
        # so loss only applies to the summary portion
        labels = list(input_ids)
        first_sep = None
        for idx, tid in enumerate(input_ids):
            if tid == sep_id:
                first_sep = idx
                break
        # Mask source text + [CLS] + first [SEP]
        if first_sep is not None:
            for idx in range(first_sep + 1):
                labels[idx] = -100
        # Mask padding
        for idx in range(len(labels)):
            if input_ids[idx] == pad_id:
                labels[idx] = -100

        all_input_ids.append(input_ids)
        all_attention.append(enc["attention_mask"])
        all_labels.append(labels)

    return Dataset.from_dict({
        "input_ids": all_input_ids,
        "attention_mask": all_attention,
        "labels": all_labels,
    })


def _tokenise_qna(records, tokenizer, max_len):
    """Encode (question, context) with character-offset-based answer spans."""
    all_input_ids, all_attention = [], []
    all_start, all_end = [], []
    skipped = 0

    for r in records:
        question = r.get("question", "")
        context = r.get("context", "")
        answer = r.get("answer", "")

        # Handle SQuAD-style answers dict
        if isinstance(answer, dict) and "text" in answer:
            ans_texts = answer["text"]
            ans_text = ans_texts[0] if isinstance(ans_texts, list) and ans_texts else str(ans_texts)
            ans_starts = answer.get("answer_start", [None])
            ans_start_char = ans_starts[0] if isinstance(ans_starts, list) and ans_starts else ans_starts
        else:
            ans_text = str(answer)
            ans_start_char = None

        if not (question and context and ans_text):
            skipped += 1
            continue

        
        if ans_start_char is None:
            ans_start_char = context.lower().find(ans_text.lower())
        if ans_start_char is None or ans_start_char == -1:
            # Try partial match
            short = ans_text[:min(30, len(ans_text))]
            ans_start_char = context.lower().find(short.lower())
        if ans_start_char is None or ans_start_char == -1:
            skipped += 1
            continue

        ans_end_char = ans_start_char + len(ans_text)

        
        enc = tokenizer(question, context, truncation=True, padding="max_length",
                        max_length=max_len, return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping")
        input_ids = enc["input_ids"]

       
        sep_id = tokenizer.sep_token_id
        ctx_start_tok = 1  # after [CLS]
        for idx, tid in enumerate(input_ids):
            if tid == sep_id:
                ctx_start_tok = idx + 1
                break

        
        start_tok, end_tok = 0, 0
        for idx in range(ctx_start_tok, len(offsets)):
            tok_s, tok_e = offsets[idx]
            if tok_s == 0 and tok_e == 0:
                continue
            if tok_s <= ans_start_char < tok_e:
                start_tok = idx
            if tok_s < ans_end_char <= tok_e:
                end_tok = idx
                break

        if end_tok == 0 and start_tok > 0:
            end_tok = start_tok

        all_input_ids.append(input_ids)
        all_attention.append(enc["attention_mask"])
        all_start.append(start_tok)
        all_end.append(end_tok)

    if skipped:
        logger.warning("QnA: skipped %d records (answer not in context)", skipped)

    return Dataset.from_dict({
        "input_ids": all_input_ids,
        "attention_mask": all_attention,
        "start_positions": all_start,
        "end_positions": all_end,
    })


def _tokenise_ner(records, tokenizer, max_len):
    
    NER_LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG",
                  "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
    label2id = {l: i for i, l in enumerate(NER_LABELS)}

    all_input_ids, all_attention, all_labels = [], [], []
    for r in records:
        text = r.get("text", "")
        entities = r.get("entities", [])
        enc = tokenizer(text, truncation=True, padding="max_length",
                        max_length=max_len, return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping")
        input_ids = enc["input_ids"]
        label_ids = [-100] * len(input_ids)  # ignore special tokens by default

        
        for idx, (os_, oe_) in enumerate(offsets):
            if os_ == 0 and oe_ == 0:
                continue  # [CLS], [SEP], [PAD]
            label_ids[idx] = label2id["O"]

        
        for ent in entities:
            ent_start = ent.get("start", -1)
            ent_end = ent.get("end", -1)
            ent_label = ent.get("label", "MISC")
            if ent_start < 0 or ent_end < 0:
                continue

            b_tag = label2id.get(f"B-{ent_label}", label2id.get("B-MISC", 0))
            i_tag = label2id.get(f"I-{ent_label}", label2id.get("I-MISC", 0))

            first_token = True
            for idx, (os_, oe_) in enumerate(offsets):
                if os_ == 0 and oe_ == 0:
                    continue
                if oe_ <= ent_start or os_ >= ent_end:
                    continue
                # This token overlaps with the entity span
                if first_token:
                    label_ids[idx] = b_tag
                    first_token = False
                else:
                    label_ids[idx] = i_tag

        all_input_ids.append(input_ids)
        all_attention.append(enc["attention_mask"])
        all_labels.append(label_ids)

    return Dataset.from_dict({
        "input_ids": all_input_ids,
        "attention_mask": all_attention,
        "labels": all_labels,
    })


def _tokenise_generation(records, tokenizer, max_len):
    texts = [f"{r.get('prompt', '')} {r.get('completion', '')}" for r in records]
    enc = tokenizer(texts, truncation=True, padding="max_length", max_length=max_len)
    enc["labels"] = [list(ids) for ids in enc["input_ids"]]
    return Dataset.from_dict(enc)


_TOKENISERS = {
    "classification": _tokenise_classification,
    "summarization":  _tokenise_summarization,
    "qna":            _tokenise_qna,
    "ner":            _tokenise_ner,
    "generation":     _tokenise_generation,
}


def _validate_records(records: list[dict], task: str) -> list[dict]:
    required_keys = {
        "classification": {"text", "label"},
        "summarization":  {"text", "summary"},
        "qna":            {"question", "answer", "context"},
        "generation":     {"prompt", "completion"},
        "ner":            {"text", "entities"},
    }
    needed = required_keys.get(task)
    if needed is None:
        raise ValueError(f"Unknown task '{task}'. Supported: {list(required_keys)}")

    valid, skipped = [], 0
    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            skipped += 1
            continue
        missing = needed - rec.keys()
        if missing:
            logger.warning("Record %d missing keys %s — skipping", i, missing)
            skipped += 1
            continue
        valid.append(rec)

    if skipped:
        logger.warning("Skipped %d / %d records due to missing keys", skipped, len(records))
    if not valid:
        raise ValueError(
            f"No valid records found. Each record must have keys: {needed}. "
            f"Example: {_format_example(task)}"
        )
    return valid


def _format_example(task: str) -> str:
    examples = {
        "classification": '{"text": "Great movie!", "label": "positive"}',
        "summarization":  '{"text": "Long article...", "summary": "Short summary."}',
        "qna":            '{"question": "What is X?", "context": "X is a framework for...", "answer": "a framework"}',
        "generation":     '{"prompt": "Once upon a", "completion": "time there was..."}',
        "ner":            '{"text": "John lives in NYC", "entities": [{"text": "John", "label": "PER", "start": 0, "end": 4}]}',
    }
    return examples.get(task, "{}")


def build_dataset(
    task: str,
    user_data: list[dict] | None = None,
    corpus_path: str | Path | None = None,
    max_len: int = 128,
    model_name: str = TINYBERT_MODEL,
    labels: str | None = None,
) -> tuple[Dataset, Dataset, AutoTokenizer, dict]:
    
    if user_data is not None:
        if not isinstance(user_data, list):
            raise TypeError(
                f"user_data must be a list[dict], got {type(user_data).__name__}. "
                f"Example: [{_format_example(task)}]"
            )
        logger.info("Using %d user-provided records", len(user_data))
        records = _validate_records(user_data, task)
    elif corpus_path is not None:
        records = _load_jsonl(corpus_path)
        if not records:
            raise ValueError(f"No valid records in {corpus_path}")
        records = _validate_records(records, task)
    else:
        raise ValueError(
            "Provide either user_data (list[dict]) or corpus_path (JSONL file). "
            f"Expected record format: {_format_example(task)}"
        )

    records = [
        {k: clean_text(v) if isinstance(v, str) else v for k, v in r.items()}
        for r in records
    ]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    meta = {}

    if task == "classification":
        unique_labels = sorted({r.get("label", "") for r in records})
        label2id = {lbl: i for i, lbl in enumerate(unique_labels)}
        id2label = {i: lbl for lbl, i in label2id.items()}
        ds = _tokenise_classification(records, tokenizer, max_len, label2id)
        meta = {"label2id": label2id, "id2label": id2label, "num_labels": len(unique_labels)}
    elif task == "ner":
        NER_LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG",
                      "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
        label2id = {l: i for i, l in enumerate(NER_LABELS)}
        id2label = {i: l for l, i in label2id.items()}
        ds = _tokenise_ner(records, tokenizer, max_len)
        meta = {"label2id": label2id, "id2label": id2label, "num_labels": len(NER_LABELS)}
    elif task in _TOKENISERS:
        ds = _TOKENISERS[task](records, tokenizer, max_len)
    else:
        raise ValueError(f"Unknown task '{task}'")

    split = ds.train_test_split(test_size=0.2, seed=42)
    logger.info("Dataset: %d train, %d val", len(split["train"]), len(split["test"]))
    return split["train"], split["test"], tokenizer, meta
