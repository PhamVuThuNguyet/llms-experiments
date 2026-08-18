"""Plot generation package for MLLM analysis figures."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from analysis.plotting.accuracy import (
    plot_completion_rate,
    plot_dual_accuracy,
    plot_sens_spec,
    plot_triage_class_sens_spec,
)
from analysis.plotting.diagnostics import (
    plot_confusion_matrices_with_invalid,
    plot_failure_breakdown,
    plot_paired_heatmaps,
    plot_self_consistency,
)
from analysis.plotting.model_metadata import (
    MODEL_DISPLAY_NAMES,
    MODEL_ORDER,
    PRICING,
    compute_model_costs,
    display_name,
    ordered_models,
    sort_by_model_order,
    with_display_names,
)
from analysis.plotting.operational import (
    plot_cost_vs_accuracy,
    plot_latency_and_cost,
    plot_success_fail_rates,
)
from analysis.plotting.style import (
    COLORS,
    ERROR_TYPE_COLORS,
    FIGURE_SIZE,
    FIGURE_SIZE_HEATMAP,
    FIGURE_SIZE_WIDE,
    FONT,
    TASK_TITLES,
    apply_style,
)


def generate_all_plots(
    task: str,
    df: pd.DataFrame,
    metrics: pd.DataFrame,
    paired: pd.DataFrame,
    consistency: pd.DataFrame,
    failures: pd.DataFrame,
    out_dir: Path,
    class_rates: pd.DataFrame | None = None,
) -> list[Path]:
    """Generate all analysis plots for one task."""
    apply_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    paths.append(plot_dual_accuracy(metrics, task, out_dir))
    paths.append(plot_completion_rate(metrics, task, out_dir))
    sens = plot_sens_spec(metrics, task, out_dir)
    if sens:
        paths.append(sens)
    if task == "triage" and class_rates is not None:
        triage_plot = plot_triage_class_sens_spec(class_rates, out_dir)
        if triage_plot:
            paths.append(triage_plot)
    paths.append(plot_self_consistency(consistency, task, out_dir))
    paths.extend(plot_paired_heatmaps(paired, task, out_dir))
    fail_plot = plot_failure_breakdown(failures, task, out_dir)
    if fail_plot:
        paths.append(fail_plot)
    paths.extend(plot_confusion_matrices_with_invalid(df, task, out_dir))
    paths.extend(plot_latency_and_cost(df, task, out_dir))
    cost_acc = plot_cost_vs_accuracy(df, metrics, task, out_dir)
    if cost_acc:
        paths.append(cost_acc)
    paths.append(plot_success_fail_rates(metrics, task, out_dir))
    return paths


__all__ = [
    "COLORS",
    "ERROR_TYPE_COLORS",
    "FIGURE_SIZE",
    "FIGURE_SIZE_HEATMAP",
    "FIGURE_SIZE_WIDE",
    "FONT",
    "MODEL_DISPLAY_NAMES",
    "MODEL_ORDER",
    "PRICING",
    "TASK_TITLES",
    "apply_style",
    "compute_model_costs",
    "display_name",
    "generate_all_plots",
    "ordered_models",
    "sort_by_model_order",
    "with_display_names",
]
