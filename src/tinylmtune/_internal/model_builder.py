from transformers import AutoConfig

from tinylmtune._internal.constants import TASK_HEAD, TINYBERT_MODEL


def build_model(
    task: str,
    model_name: str = TINYBERT_MODEL,
    num_labels: int = 2,
    label2id: dict | None = None,
    id2label: dict | None = None,
    dropout: float = 0.1,
    attention_dropout: float = 0.1,
):
   
    model_cls = TASK_HEAD.get(task)
    if model_cls is None:
        raise ValueError(f"Unknown task '{task}'. Supported: {list(TASK_HEAD)}")

    config = AutoConfig.from_pretrained(model_name)

    if task in ("classification", "ner"):
        config.num_labels = num_labels

    
    config.hidden_dropout_prob = dropout
    config.attention_probs_dropout_prob = attention_dropout

    if label2id:
        config.label2id = label2id
    if id2label:
        config.id2label = {int(k): v for k, v in id2label.items()}

    model = model_cls.from_pretrained(model_name, config=config, ignore_mismatched_sizes=True)
    return model
