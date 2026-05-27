import logging
import os
from pathlib import Path
from tinylmtune._internal.corpus_gen import generate_corpus
from tinylmtune._internal.dataset import build_dataset, _validate_records, _format_example, _load_jsonl
from tinylmtune._internal.ga_optimizer import TinyOptimizer
from tinylmtune._internal.inference import save_best_model
from tinylmtune._internal.raw_converter import convert_raw_text
from tinylmtune._internal.search_space_advisor import recommend_search_space
from tinylmtune._internal.token_analyzer import analyze_token_lengths
from tinylmtune._internal.trainer import train_and_evaluate

logger = logging.getLogger(__name__)

def _detect_and_resolve_data(
    user_data,
    task: str,
    labels: str | None,
    split_strategy: str,
    llm_model: str,
) -> list[dict]:
   
    
    if isinstance(user_data, str):
        logger.info("user_data is a string — treating as raw text")
        return convert_raw_text(
            raw_input=user_data, task=task, labels=labels,
            split_strategy=split_strategy, llm_model=llm_model,
        )

    if not isinstance(user_data, list) or len(user_data) == 0:
        raise TypeError(
            "user_data must be a non-empty list[dict], list[str], or str. "
            f"Got: {type(user_data).__name__}"
        )

    first = user_data[0]

    
    if isinstance(first, str):
        logger.info("user_data is list[str] (%d items) — converting via Flan-T5", len(user_data))
        return convert_raw_text(
            raw_input=user_data, task=task, labels=labels,
            split_strategy=split_strategy, llm_model=llm_model,
            
        )

    
    if isinstance(first, dict):
        try:
            valid = _validate_records(user_data, task)
            logger.info("user_data validated: %d / %d structured records", len(valid), len(user_data))
            return valid
        except ValueError:
            _TEXT_KEYS = ("text", "content", "sentence", "input", "body", "raw")
            raw_texts = []
            for rec in user_data:
                if not isinstance(rec, dict):
                    continue
                for key in _TEXT_KEYS:
                    val = rec.get(key, "")
                    if isinstance(val, str) and len(val.strip()) >= 10:
                        raw_texts.append(val.strip())
                        break
            if raw_texts:
                logger.warning(
                    "Structured validation failed — found %d raw text fields. "
                    "Auto-converting via Flan-T5.", len(raw_texts),
                )
                return convert_raw_text(
                    raw_input=raw_texts, task=task, labels=labels,
                    split_strategy=split_strategy, llm_model=llm_model,
                    
                )
            raise ValueError(
                f"user_data records don't match task '{task}' and contain "
                f"no extractable text fields (tried keys: {_TEXT_KEYS}). "
                f"Expected format: {_format_example(task)}"
            )

    raise TypeError(f"user_data items must be dict or str. Got: {type(first).__name__}")


def optimize_slm(
    task: str,
    user_data: list[dict] | list[str] | str | None = None,
    corpus_prompt: str = "general knowledge",
    n_examples: int = 200,
    labels: str | None = None,
    split_strategy: str = "sentence",
    llm_model: str = "google/flan-t5-small",
    search_space: dict | None = None,
    pop_size: int = 6,
    generations: int = 3,
    crossover_rate: float = 0.7,
    mutation_rate: float = 0.25,
    max_len: int = 128,
    output_dir: str | None = None,
    corpus_path: str | None = None,
) -> dict:
    
   
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), "tiny_model")
    elif not os.path.isabs(output_dir):
        output_dir = os.path.join(os.getcwd(), output_dir)
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Model will be saved to: %s", output_dir)

    
    _user_data = None
    _corpus_path = None

    if user_data is not None:
        _user_data = _detect_and_resolve_data(
            user_data=user_data, task=task, labels=labels,
            split_strategy=split_strategy, llm_model=llm_model,
            
        )
        logger.info("Resolved %d structured records", len(_user_data))
    elif corpus_path and Path(corpus_path).exists():
        logger.info("Using existing corpus: %s", corpus_path)
        _corpus_path = Path(corpus_path)
    else:
        logger.info("No user data — generating synthetic corpus")
        _corpus_path = generate_corpus(
            task=task, topic=corpus_prompt, n_examples=n_examples,
            labels=labels, output_path=os.path.join(output_dir, "corpus.jsonl"),
            llm_model=llm_model,
        )

    
    if _user_data is not None:
        raw_records = _user_data
    elif _corpus_path is not None:
        raw_records = _load_jsonl(_corpus_path)
    else:
        raise RuntimeError("No data resolved")

    
    token_profile = analyze_token_lengths(raw_records, task)
    fixed_max_len = token_profile["recommended_max_len"]
    logger.info(
        "Token analysis: p50=%d p95=%d max=%d → fixed max_len=%d",
        token_profile["p50"], token_profile["p95"],
        token_profile["max"], fixed_max_len,
    )

    train_ds, val_ds, tokenizer, meta = build_dataset(
        task=task, user_data=raw_records, max_len=fixed_max_len,
    )
    num_labels = meta.get("num_labels", 2)
    n_train = len(train_ds)
    logger.info("Task=%s, num_labels=%d, n_train=%d", task, num_labels, n_train)

    
    recommended = recommend_search_space(
        n_samples=n_train, task=task, max_len=fixed_max_len,
    )
    
    recommended.pop("max_len", None)

    
    if search_space:
        search_space.pop("max_len", None)  # ignore if user passed it
        recommended.update(search_space)

    logger.info("Fixed max_len=%d | GA search space (%d params): %s",
                fixed_max_len, len(recommended), list(recommended.keys()))

    
    optimizer = TinyOptimizer(
        train_ds=train_ds,
        val_ds=val_ds,
        task=task,
        num_labels=num_labels,
        max_len=fixed_max_len,
        search_space=recommended,
        meta=meta,
    )
    best_config, best_fitness = optimizer.run(
        pop_size=pop_size,
        generations=generations,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
    )
    logger.info("Best config (fitness=%.4f): %s", best_fitness, best_config)

    
    ga_history = optimizer.history
    ga_generation_stats = optimizer.generation_stats

    
    result = train_and_evaluate(
        train_ds=train_ds, val_ds=val_ds,
        task=task, num_labels=num_labels,
        learning_rate=best_config["learning_rate"],
        batch_size=best_config["batch_size"],
        epochs=best_config["epochs"],
        warmup_ratio=best_config["warmup_ratio"],
        weight_decay=best_config["weight_decay"],
        dropout=best_config.get("dropout", 0.1),
        attention_dropout=best_config.get("attention_dropout", 0.1),
        gradient_accumulation_steps=best_config.get("gradient_accumulation_steps", 1),
        lr_scheduler_type=best_config.get("lr_scheduler_type", "linear"),
        label_smoothing=best_config.get("label_smoothing", 0.0),
        max_grad_norm=best_config.get("max_grad_norm", 1.0),
        max_len=fixed_max_len,
        output_dir=output_dir,
        label2id=meta.get("label2id"),
        id2label=meta.get("id2label"),
    )

    save_best_model(
        trainer=result["trainer"],
        tokenizer=tokenizer,
        task=task,
        output_dir=output_dir,
        best_config=best_config,
    )

    best_config["max_len"] = fixed_max_len
    best_config["task"] = task
    best_config["output_dir"] = output_dir
    best_config["ga_history"] = ga_history
    best_config["ga_generation_stats"] = ga_generation_stats
    return best_config
