"""
LLM backend for tinyLMTune — uses Flan-T5 from HuggingFace.

No Ollama, no external servers. Runs locally on GPU or CPU.
The model is loaded once and cached for the session.

Supports any HuggingFace text2text model — default is
google/flan-t5-small (77M params, fits on any GPU).
For better quality, use google/flan-t5-base (248M) or
google/flan-t5-large (783M).
"""

import logging

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

logger = logging.getLogger(__name__)

# Default model — small enough to run alongside TinyBERT training
DEFAULT_MODEL = "google/flan-t5-base"

# Singleton cache: model_name → (model, tokenizer)
_MODEL_CACHE: dict[str, tuple] = {}


def _get_model(model_name: str = DEFAULT_MODEL):
    """Load and cache the LLM. Returns (model, tokenizer)."""
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    logger.info("Loading LLM: %s ...", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    logger.info("LLM loaded on %s (%d params)",
                device, sum(p.numel() for p in model.parameters()))

    _MODEL_CACHE[model_name] = (model, tokenizer)
    return model, tokenizer


def unload_model(model_name: str = DEFAULT_MODEL):
    """Free the cached LLM from memory (call before TinyBERT training)."""
    if model_name in _MODEL_CACHE:
        del _MODEL_CACHE[model_name]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Unloaded LLM: %s", model_name)


def generate_text(
    prompt: str,
    model_name: str = DEFAULT_MODEL,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    num_return_sequences: int = 1,
) -> str:
    """
    Generate text from a prompt using the cached LLM.

    Returns the generated text as a string.
    """
    model, tokenizer = _get_model(model_name)
    device = next(model.parameters()).device

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            num_return_sequences=num_return_sequences,
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text


def generate_batch(
    prompts: list[str],
    model_name: str = DEFAULT_MODEL,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
) -> list[str]:
    """
    Generate text for a batch of prompts.

    Returns a list of generated texts, one per prompt.
    """
    model, tokenizer = _get_model(model_name)
    device = next(model.parameters()).device

    results = []
    # Process in small batches to avoid OOM
    batch_size = 8
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", truncation=True,
            max_length=512, padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
            )

        for out in outputs:
            text = tokenizer.decode(out, skip_special_tokens=True)
            results.append(text)

    return results
