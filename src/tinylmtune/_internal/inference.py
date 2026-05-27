import json
import logging
from pathlib import Path

import torch
from transformers import AutoTokenizer

from tinylmtune._internal.constants import TASK_HEAD, TINYBERT_MODEL

logger = logging.getLogger(__name__)


def save_best_model(
    trainer,
    tokenizer: AutoTokenizer,
    task: str,
    output_dir: str,
    best_config: dict,
):
    
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))

    meta = {"task": task, "best_config": best_config}
    (out / "tinylmtune_meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("Model saved → %s", out)


class TinyInference:

    def __init__(self, model_dir: str):
        self._dir = Path(model_dir)
        meta_path = self._dir / "tinylmtune_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"No tinylmtune_meta.json in {model_dir}")

        meta = json.loads(meta_path.read_text())
        self.task = meta["task"]
        self.config = meta.get("best_config", {})

        self._tokenizer = AutoTokenizer.from_pretrained(str(self._dir))

        model_cls = TASK_HEAD[self.task]
        self._model = model_cls.from_pretrained(str(self._dir))
        self._model.eval()
        logger.info("Loaded %s model from %s", self.task, self._dir)

    def predict(self, text: str, **kwargs) -> dict:
        
        dispatch = {
            "classification": self._predict_classification,
            "summarization":  self._predict_masked,
            "generation":     self._predict_masked,
            "qna":            self._predict_qna,
            "ner":            self._predict_ner,
        }
        fn = dispatch.get(self.task)
        if fn is None:
            raise ValueError(f"Inference not implemented for task '{self.task}'")
        return fn(text, **kwargs)

    def _predict_classification(self, text: str, **kwargs) -> dict:
        enc = self._tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = self._model(**enc).logits
        probs = torch.softmax(logits, dim=-1).squeeze()
        predicted = int(torch.argmax(probs))
        raw_map = self._model.config.id2label or {}
        id2label = {int(k): v for k, v in raw_map.items()}
        return {
            "label": id2label.get(predicted, str(predicted)),
            "confidence": float(probs[predicted]),
            "probabilities": {id2label.get(i, str(i)): float(p) for i, p in enumerate(probs)},
        }

    def _predict_masked(self, text: str, **kwargs) -> dict:
        enc = self._tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = self._model(**enc).logits
        predicted_ids = torch.argmax(logits, dim=-1).squeeze()
        decoded = self._tokenizer.decode(predicted_ids, skip_special_tokens=True)
        return {"output": decoded}

    def _predict_qna(self, text: str, context: str = "", **kwargs) -> dict:
        if not context:
            return {"answer": "", "start": 0, "end": 0,
                    "error": "QnA requires a 'context' parameter"}
        enc = self._tokenizer(text, context, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            out = self._model(**enc)
        start = int(torch.argmax(out.start_logits, dim=-1))
        end = int(torch.argmax(out.end_logits, dim=-1))
        
        if end < start:
            start_logits = out.start_logits.squeeze()
            end_logits = out.end_logits.squeeze()
            best_score = float("-inf")
            for s in torch.topk(start_logits, 10).indices.tolist():
                for e in torch.topk(end_logits, 10).indices.tolist():
                    if e >= s and (e - s) < 30:  # valid span, max 30 tokens
                        score = start_logits[s].item() + end_logits[e].item()
                        if score > best_score:
                            best_score = score
                            start, end = s, e

        tokens = enc["input_ids"][0][start:end + 1]
        answer = self._tokenizer.decode(tokens, skip_special_tokens=True)
        return {"answer": answer, "start": start, "end": end}

    def _predict_ner(self, text: str, **kwargs) -> dict:
        enc = self._tokenizer(text, return_tensors="pt", truncation=True,
                              padding=True, return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping").squeeze().tolist()
        with torch.no_grad():
            logits = self._model(**enc).logits
        preds = torch.argmax(logits, dim=-1).squeeze().tolist()
        raw_map = self._model.config.id2label or {}
        id2label = {int(k): v for k, v in raw_map.items()}

        entities, current = [], None
        for idx, (pred, (os_, oe_)) in enumerate(zip(preds, offsets)):
            if os_ == 0 and oe_ == 0:
                continue
            tag = id2label.get(pred, "O")
            if tag.startswith("B-"):
                if current:
                    entities.append(current)
                current = {"text": text[os_:oe_], "label": tag[2:], "start": os_, "end": oe_}
            elif tag.startswith("I-") and current and tag[2:] == current["label"]:
                current["text"] = text[current["start"]:oe_]
                current["end"] = oe_
            else:
                if current:
                    entities.append(current)
                    current = None
        if current:
            entities.append(current)

        return {"text": text, "entities": entities}
