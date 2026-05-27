import logging
from pathlib import Path

logger = logging.getLogger(__name__)


_METRIC_NAMES = {
    "classification": "F1 Score",
    "summarization":  "Inverse Loss",
    "generation":     "Inverse Perplexity",
    "qna":            "Inverse Loss",
    "ner":            "Token F1",
}


_NUMERIC_PARAMS = [
    "learning_rate", "batch_size", "epochs", "warmup_ratio", "weight_decay",
    "dropout", "attention_dropout", "gradient_accumulation_steps",
    "label_smoothing", "max_grad_norm",
]


_CATEGORICAL_PARAMS = ["lr_scheduler_type"]


def _ensure_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install it with: pip install matplotlib"
        )


def plot_fitness_over_generations(generation_stats, task="classification", save_path=None):
    
    plt = _ensure_matplotlib()

    gens = [s["generation"] for s in generation_stats]
    bests = [s["best_fitness"] for s in generation_stats]
    avgs = [s["avg_fitness"] for s in generation_stats]
    worsts = [s["worst_fitness"] for s in generation_stats]

    metric_name = _METRIC_NAMES.get(task, "Fitness")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(gens, bests, "g-o", linewidth=2, markersize=8, label="Best", zorder=3)
    ax.plot(gens, avgs, "b--s", linewidth=1.5, markersize=6, label="Average")
    ax.plot(gens, worsts, "r:^", linewidth=1, markersize=5, label="Worst", alpha=0.7)

    ax.fill_between(gens, worsts, bests, alpha=0.1, color="blue")

    
    max_fit = max(bests)
    max_gen = gens[bests.index(max_fit)]
    ax.annotate(
        f"Best: {max_fit:.4f}",
        xy=(max_gen, max_fit), xytext=(max_gen + 0.3, max_fit + 0.02),
        fontsize=11, fontweight="bold", color="green",
        arrowprops=dict(arrowstyle="->", color="green"),
    )

    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(f"GA Optimisation Progress — {metric_name}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(gens)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved: %s", save_path)
    return fig


def plot_parameter_scatter(history, task="classification", save_path=None):
   
    plt = _ensure_matplotlib()
    import numpy as np

    params = [p for p in _NUMERIC_PARAMS if p in history[0]]
    if not params:
        logger.warning("No numeric parameters found in history")
        return None

    n_params = len(params)
    cols = min(3, n_params)
    rows = (n_params + cols - 1) // cols

    metric_name = _METRIC_NAMES.get(task, "Fitness")

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4.5 * rows))
    if n_params == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    fitnesses = [h["fitness"] for h in history]
    best_idx = fitnesses.index(max(fitnesses))

    
    _MARKERS = ["o", "s", "D", "^", "v", "P", "X", "h", "<", ">"]
    _COLORS = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]

    unique_gens = sorted(set(h["generation"] for h in history))
    gen_marker = {g: _MARKERS[i % len(_MARKERS)] for i, g in enumerate(unique_gens)}
    gen_color = {g: _COLORS[i % len(_COLORS)] for i, g in enumerate(unique_gens)}

    for i, param in enumerate(params):
        if i >= len(axes):
            break
        ax = axes[i]
        vals = [h[param] for h in history]

        # Plot each generation with its own marker shape and color
        for gen in unique_gens:
            gen_idx = [j for j, h in enumerate(history) if h["generation"] == gen]
            gen_vals = [vals[j] for j in gen_idx]
            gen_fits = [fitnesses[j] for j in gen_idx]

            ax.scatter(
                gen_vals, gen_fits,
                marker=gen_marker[gen], c=gen_color[gen],
                s=60, alpha=0.7, edgecolors="white", linewidth=0.5,
                label=f"Gen {gen}",
            )

        
        ax.scatter(
            [vals[best_idx]], [fitnesses[best_idx]],
            c="red", s=250, marker="*", zorder=5,
            edgecolors="darkred", linewidth=1.5,
            label=f"Best ({vals[best_idx]:.4g})",
        )

        ax.set_xlabel(param, fontsize=10)
        ax.set_ylabel(metric_name, fontsize=10)
        ax.set_title(f"{param}", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.2)

        if param == "learning_rate":
            ax.set_xscale("log")

        
        if i == 0:
            ax.legend(fontsize=7, loc="lower right", ncol=2)

    
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Parameter vs {metric_name} — All Individuals", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved: %s", save_path)
    return fig


def plot_scheduler_comparison(history, task="classification", save_path=None):
    
    plt = _ensure_matplotlib()

    if "lr_scheduler_type" not in history[0]:
        return None

    metric_name = _METRIC_NAMES.get(task, "Fitness")

    
    scheduler_fitness = {}
    for h in history:
        sched = h["lr_scheduler_type"]
        scheduler_fitness.setdefault(sched, []).append(h["fitness"])

    labels = sorted(scheduler_fitness.keys())
    data = [scheduler_fitness[l] for l in labels]

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, labels=labels, patch_artist=True)

    colors = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0"]
    for patch, color in zip(bp["boxes"], colors[:len(labels)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    
    means = [sum(d) / len(d) for d in data]
    best_idx = means.index(max(means))
    bp["boxes"][best_idx].set_edgecolor("red")
    bp["boxes"][best_idx].set_linewidth(3)

    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(f"LR Scheduler Comparison — {metric_name}", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved: %s", save_path)
    return fig


def plot_best_config_evolution(generation_stats, history=None, task="classification", save_path=None):
   
    plt = _ensure_matplotlib()
    import numpy as np

    metric_name = _METRIC_NAMES.get(task, "Fitness")

    
    if not history:
        logger.warning("plot_best_config_evolution needs 'history' for per-generation lines. "
                        "Pass results['ga_history'].")
        return None

    params = [p for p in _NUMERIC_PARAMS if p in history[0]]
    if not params:
        return None

    unique_gens = sorted(set(h["generation"] for h in history))

    _COLORS = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]
    _MARKERS = ["o", "s", "D", "^", "v", "P", "X", "h", "<", ">"]

    n_params = len(params)
    cols = min(3, n_params)
    rows = (n_params + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4.5 * rows))
    if n_params == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    
    all_fitnesses = [h["fitness"] for h in history]
    best_idx = all_fitnesses.index(max(all_fitnesses))
    best_record = history[best_idx]

    for i, param in enumerate(params):
        if i >= len(axes):
            break
        ax = axes[i]

        for gen in unique_gens:
            gen_data = [h for h in history if h["generation"] == gen]
            # Sort by param value for clean line
            gen_data.sort(key=lambda h: h[param])

            vals = [h[param] for h in gen_data]
            fits = [h["fitness"] for h in gen_data]

            color = _COLORS[(gen - 1) % len(_COLORS)]
            marker = _MARKERS[(gen - 1) % len(_MARKERS)]

            ax.plot(
                vals, fits,
                marker=marker, color=color,
                linewidth=1.2, markersize=6, alpha=0.8,
                label=f"Gen {gen}",
            )

        
        ax.scatter(
            [best_record[param]], [best_record["fitness"]],
            c="red", s=250, marker="*", zorder=10,
            edgecolors="darkred", linewidth=1.5,
            label=f"Best ({best_record[param]:.4g})",
        )

        ax.set_xlabel(param, fontsize=10)
        ax.set_ylabel(metric_name, fontsize=10)
        ax.set_title(f"{param} vs {metric_name}", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.2)

        if param == "learning_rate":
            ax.set_xscale("log")

        if i == 0:
            ax.legend(fontsize=7, loc="best", ncol=2)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"Parameter Range vs {metric_name} — Per Generation",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved: %s", save_path)
    return fig


def plot_population_heatmap(history, generation=None, task="classification", save_path=None):
    
    plt = _ensure_matplotlib()
    import numpy as np

    if generation is None:
        generation = max(h["generation"] for h in history)

    gen_data = [h for h in history if h["generation"] == generation]
    if not gen_data:
        return None

    metric_name = _METRIC_NAMES.get(task, "Fitness")
    params = [p for p in _NUMERIC_PARAMS if p in gen_data[0]]

    
    matrix = []
    fitnesses = []
    for h in gen_data:
        row = [h[p] for p in params]
        matrix.append(row)
        fitnesses.append(h["fitness"])

    matrix = np.array(matrix, dtype=float)
    
    col_min = matrix.min(axis=0)
    col_max = matrix.max(axis=0)
    col_range = col_max - col_min
    col_range[col_range == 0] = 1
    norm_matrix = (matrix - col_min) / col_range

    
    sort_idx = np.argsort(fitnesses)[::-1]
    norm_matrix = norm_matrix[sort_idx]
    sorted_fit = [fitnesses[i] for i in sort_idx]

    fig, ax = plt.subplots(figsize=(max(10, len(params) * 0.8), max(4, len(gen_data) * 0.5)))
    im = ax.imshow(norm_matrix, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(params)))
    ax.set_xticklabels(params, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(gen_data)))
    ax.set_yticklabels([f"#{i+1} (fit={f:.4f})" for i, f in enumerate(sorted_fit)], fontsize=9)

   
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.get_yticklabels()[0].set_color("green")

    
    for i in range(norm_matrix.shape[0]):
        for j in range(norm_matrix.shape[1]):
            orig_idx = sort_idx[i]
            val = matrix[orig_idx, j]
            text = f"{val:.3g}" if isinstance(val, float) else str(int(val))
            ax.text(j, i, text, ha="center", va="center", fontsize=7,
                    color="white" if norm_matrix[i, j] > 0.6 else "black")

    ax.set_title(
        f"Generation {generation} — Population Heatmap (sorted by {metric_name})",
        fontsize=13, fontweight="bold",
    )
    plt.colorbar(im, ax=ax, label="Normalized value")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved: %s", save_path)
    return fig


def plot_results(results: dict, task: str = None, save_dir: str = None, show: bool = True):
    
    plt = _ensure_matplotlib()

    history = results.get("ga_history", [])
    gen_stats = results.get("ga_generation_stats", [])
    task = task or results.get("task", "classification")

    if not history:
        logger.warning("No GA history found in results. Did you run optimize_slm()?")
        return {}

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    def _save(name):
        return str(Path(save_dir) / f"{name}.png") if save_dir else None

    figs = {}

    
    if gen_stats:
        figs["fitness_progress"] = plot_fitness_over_generations(
            gen_stats, task, save_path=_save("01_fitness_progress"),
        )

    
    if history:
        figs["parameter_scatter"] = plot_parameter_scatter(
            history, task, save_path=_save("02_parameter_scatter"),
        )

    
    if history and "lr_scheduler_type" in history[0]:
        fig = plot_scheduler_comparison(
            history, task, save_path=_save("03_scheduler_comparison"),
        )
        if fig:
            figs["scheduler_comparison"] = fig

    
    if history:
        figs["config_evolution"] = plot_best_config_evolution(
            gen_stats, history=history, task=task,
            save_path=_save("04_config_evolution"),
        )

    
    if history:
        figs["population_heatmap"] = plot_population_heatmap(
            history, task=task, save_path=_save("05_population_heatmap"),
        )

    if show:
        plt.show()

    logger.info("Generated %d plots", len(figs))
    return figs


def print_best_config_table(results: dict):
   
    task = results.get("task", "classification")
    metric_name = _METRIC_NAMES.get(task, "Fitness")

    print("=" * 55)
    print(f"  Best Configuration — {metric_name}: {results.get('fitness', 'N/A'):.4f}")
    print("=" * 55)

    skip = {"fitness", "output_dir", "max_len", "task", "ga_history", "ga_generation_stats"}
    for key, val in results.items():
        if key in skip:
            continue
        if isinstance(val, float):
            print(f"  {key:35s}: {val:.6g}")
        else:
            print(f"  {key:35s}: {val}")

    print(f"  {'max_len (fixed from data)':35s}: {results.get('max_len', 'N/A')}")
    print(f"  {'output_dir':35s}: {results.get('output_dir', 'N/A')}")
    print()
