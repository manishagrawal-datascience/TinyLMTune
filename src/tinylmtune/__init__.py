
"""tinyLMTune — Genetic-Algorithm-Optimised TinyBERT Fine-Tuning
==============================================================

Public API:
    from tinylmtune import optimize_slm, TinyInference
    from tinylmtune import print_token_analysis, print_recommendation

Three upgrades:
    1. max_len auto-determined from data via token analysis (GA optimises it)
    2. Search space recommended based on task + dataset size
    3. user_data accepts structured dicts, raw text, or file paths

Usage:
    # ── Structured dicts ──
    best = optimize_slm(task="classification", user_data=[
        {"text": "Great!", "label": "positive"},
        {"text": "Terrible.", "label": "negative"},
    ])

    # ── Raw strings (auto-labelled via Flan-T5) ──
    best = optimize_slm(task="classification", user_data=[
        "This movie was wonderful!", "Worst film ever.",
    ], labels="positive,negative")

    # ── Text file ──
    best = optimize_slm(task="classification", user_data="reviews.txt")

    # ── Pre-flight analysis ──
    from tinylmtune import print_token_analysis, print_recommendation
    print_token_analysis(my_data, task="classification")
    print_recommendation(n_samples=200, task="classification")

    # ── Inference ──
    model = TinyInference("tiny_model")
    print(model.predict("Amazing product!"))
"""

__version__ = "2.0.0"

from tinylmtune._internal.pipeline import optimize_slm
from tinylmtune._internal.inference import TinyInference
from tinylmtune._internal.token_analyzer import print_token_analysis, analyze_token_lengths
from tinylmtune._internal.search_space_advisor import print_recommendation, recommend_search_space
from tinylmtune._internal.visualizer import (
    plot_results,
    plot_fitness_over_generations,
    plot_parameter_scatter,
    plot_scheduler_comparison,
    plot_best_config_evolution,
    plot_population_heatmap,
    print_best_config_table,
)

__all__ = [
    "optimize_slm",
    "TinyInference",
    "print_token_analysis",
    "analyze_token_lengths",
    "print_recommendation",
    "recommend_search_space",
    "plot_results",
    "plot_fitness_over_generations",
    "plot_parameter_scatter",
    "plot_scheduler_comparison",
    "plot_best_config_evolution",
    "plot_population_heatmap",
    "print_best_config_table",
]
