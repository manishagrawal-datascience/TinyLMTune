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
