import logging
import random
from copy import deepcopy

from datasets import Dataset

from tinylmtune._internal.constants import GA_SEARCH_SPACE
from tinylmtune._internal.trainer import train_and_evaluate

logger = logging.getLogger(__name__)

_INT_PARAMS = {"epochs"}
_ROUND_PARAMS = {
    "warmup_ratio": 3, "weight_decay": 4, "dropout": 3,
    "attention_dropout": 3, "label_smoothing": 3, "max_grad_norm": 2,
}


def _sample_param(key: str, spec):
    if isinstance(spec, list):
        return random.choice(spec)
    lo, hi = spec
    if key in _INT_PARAMS:
        return random.randint(int(lo), int(hi))
    val = random.uniform(lo, hi)
    if key in _ROUND_PARAMS:
        return round(val, _ROUND_PARAMS[key])
    return val


def _random_individual(search_space: dict) -> dict:
    return {key: _sample_param(key, spec) for key, spec in search_space.items()}


def _crossover(parent_a: dict, parent_b: dict) -> dict:
    child = {}
    keys = list(parent_a.keys())
    cx_point = random.randint(1, len(keys) - 1)
    for i, k in enumerate(keys):
        child[k] = parent_a[k] if i < cx_point else parent_b[k]
    return child


def _mutate(individual: dict, rate: float = 0.25, search_space: dict | None = None) -> dict:
    ind = deepcopy(individual)
    sp = search_space or GA_SEARCH_SPACE
    for key in ind:
        if key not in sp:
            continue
        if random.random() > rate:
            continue
        ind[key] = _sample_param(key, sp[key])
    return ind


def _fitness(
    individual: dict,
    train_ds: Dataset,
    val_ds: Dataset,
    task: str,
    num_labels: int,
    max_len: int,
    meta: dict | None = None,
) -> float:
    meta = meta or {}
    try:
        result = train_and_evaluate(
            train_ds=train_ds, val_ds=val_ds,
            task=task, num_labels=num_labels,
            learning_rate=individual["learning_rate"],
            batch_size=individual["batch_size"],
            epochs=individual["epochs"],
            warmup_ratio=individual["warmup_ratio"],
            weight_decay=individual["weight_decay"],
            dropout=individual.get("dropout", 0.1),
            attention_dropout=individual.get("attention_dropout", 0.1),
            gradient_accumulation_steps=individual.get("gradient_accumulation_steps", 1),
            lr_scheduler_type=individual.get("lr_scheduler_type", "linear"),
            label_smoothing=individual.get("label_smoothing", 0.0),
            max_grad_norm=individual.get("max_grad_norm", 1.0),
            max_len=max_len,
            label2id=meta.get("label2id"),
            id2label=meta.get("id2label"),
        )
        metrics = result["metrics"]
        if task == "classification":
            return metrics.get("eval_f1", metrics.get("eval_accuracy", 0.0))
        elif task == "qna":
            return metrics.get("eval_exact_match", 0.0)
        elif task == "ner":
            return metrics.get("eval_token_f1", 0.0)
        else:
            loss = metrics.get("eval_loss", 999.0)
            return 1.0 / (1.0 + loss)
    except Exception as exc:
        logger.warning("Individual failed: %s — %s", individual, exc)
        return 0.0


class TinyOptimizer:

    def __init__(
        self,
        train_ds: Dataset,
        val_ds: Dataset,
        task: str,
        num_labels: int = 2,
        max_len: int = 128,
        search_space: dict | None = None,
        meta: dict | None = None,
    ):
        self._train_ds = train_ds
        self._val_ds = val_ds
        self._task = task
        self._num_labels = num_labels
        self._max_len = max_len
        self._search_space = search_space or GA_SEARCH_SPACE
        self._meta = meta or {}

        # Full history — populated during run()
        self.history: list[dict] = []
        self.generation_stats: list[dict] = []

    def run(
        self,
        pop_size: int = 6,
        generations: int = 3,
        crossover_rate: float = 0.7,
        mutation_rate: float = 0.25,
        elitism: int = 1,
    ) -> tuple[dict, float]:
        
        pop_size = max(pop_size, 4)
        population = [_random_individual(self._search_space) for _ in range(pop_size)]

        best_ever, best_fit = None, -1.0
        self.history = []
        self.generation_stats = []

        for gen in range(generations):
            scores = []
            for idx, ind in enumerate(population):
                f = _fitness(ind, self._train_ds, self._val_ds,
                             self._task, self._num_labels, self._max_len,
                             self._meta)
                scores.append((ind, f))

                
                record = {
                    "generation": gen + 1,
                    "individual": idx,
                    "fitness": f,
                    **{k: v for k, v in ind.items()},
                }
                self.history.append(record)

                if f > best_fit:
                    best_fit = f
                    best_ever = deepcopy(ind)

            scores.sort(key=lambda x: x[1], reverse=True)
            fitnesses = [s for _, s in scores]

            gen_stat = {
                "generation": gen + 1,
                "best_fitness": fitnesses[0],
                "worst_fitness": fitnesses[-1],
                "avg_fitness": sum(fitnesses) / len(fitnesses),
                "std_fitness": (sum((f - sum(fitnesses)/len(fitnesses))**2 for f in fitnesses) / len(fitnesses)) ** 0.5,
                "best_config": deepcopy(scores[0][0]),
            }
            self.generation_stats.append(gen_stat)

            logger.info(
                "Gen %d/%d — best=%.4f  avg=%.4f  worst=%.4f",
                gen + 1, generations, gen_stat["best_fitness"],
                gen_stat["avg_fitness"], gen_stat["worst_fitness"],
            )

            
            next_pop = [deepcopy(scores[i][0]) for i in range(elitism)]
            while len(next_pop) < pop_size:
                a = min(random.sample(range(len(scores)), 3), key=lambda i: -scores[i][1])
                b = min(random.sample(range(len(scores)), 3), key=lambda i: -scores[i][1])
                pa, pb = scores[a][0], scores[b][0]
                child = _crossover(pa, pb) if random.random() < crossover_rate else deepcopy(pa)
                child = _mutate(child, rate=mutation_rate, search_space=self._search_space)
                next_pop.append(child)
            population = next_pop[:pop_size]

        best_ever["fitness"] = best_fit
        return best_ever, best_fit
