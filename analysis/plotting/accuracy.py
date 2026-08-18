"""Accuracy, completion, and sensitivity/specificity plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.constants import TRIAGE_CLASSES
from analysis.plotting.model_metadata import (
    ordered_models,
    sort_by_model_order,
    with_display_names,
)
from analysis.plotting.style import (
    COLORS,
    FIGURE_SIZE,
    FIGURE_SIZE_WIDE,
    FONT,
    TASK_TITLES,
    err_bars,
    save_figure,
)


def plot_dual_accuracy(metrics: pd.DataFrame, task: str, out_dir: Path) -> Path:
    df = sort_by_model_order(metrics)
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    y = np.arange(len(df))
    height = 0.35

    e2e_err = err_bars(
        df["accuracy_end_to_end"],
        df["accuracy_end_to_end_ci_low"],
        df["accuracy_end_to_end_ci_high"],
    )
    valid_err = err_bars(
        df["accuracy_among_valid"],
        df["accuracy_among_valid_ci_low"],
        df["accuracy_among_valid_ci_high"],
    )

    ax.barh(
        y - height / 2,
        df["accuracy_end_to_end"].fillna(0),
        height,
        xerr=e2e_err,
        label="End-to-end (invalid = incorrect)",
        color=COLORS["end_to_end"],
        alpha=0.9,
        capsize=3,
    )
    ax.barh(
        y + height / 2,
        df["accuracy_among_valid"].fillna(0),
        height,
        xerr=valid_err,
        label="Among valid outputs",
        color=COLORS["among_valid"],
        alpha=0.9,
        capsize=3,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(df["model_name"])
    ax.set_xlabel("Accuracy")
    ax.set_xlim(0, 1.15)
    ax.set_title(f"{TASK_TITLES[task]}: Dual Accuracy with 95% Bootstrap CIs")
    ax.legend(loc="lower right")
    return save_figure(fig, out_dir / f"{task}_dual_accuracy_with_ci.png")


def plot_completion_rate(metrics: pd.DataFrame, task: str, out_dir: Path) -> Path:
    df = sort_by_model_order(metrics)
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    err = err_bars(df["completion_rate"], df["completion_ci_low"], df["completion_ci_high"])
    bars = ax.barh(
        df["model_name"].astype(str),
        df["completion_rate"],
        xerr=err,
        color=COLORS["completion"],
        alpha=0.9,
        capsize=3,
    )
    ax.set_xlim(0, 1.18)
    ax.set_xlabel("Task Completion Rate")
    ax.set_title(f"{TASK_TITLES[task]}: Structured-Output Completion Rate (95% CI)")
    for bar, value in zip(bars, df["completion_rate"]):
        ax.text(
            value + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1%}",
            va="center",
            fontsize=FONT["annot"],
        )
    return save_figure(fig, out_dir / f"{task}_completion_rate_with_ci.png")


def plot_sens_spec(metrics: pd.DataFrame, task: str, out_dir: Path) -> Path | None:
    if "sensitivity_end_to_end" not in metrics.columns:
        return None
    df = sort_by_model_order(metrics)
    fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE_WIDE)
    x = np.arange(len(df))
    width = 0.35

    for ax, sens_col, spec_col, title in (
        (
            axes[0],
            "sensitivity_end_to_end",
            "specificity_end_to_end",
            "End-to-end (invalid treated as incorrect)",
        ),
        (
            axes[1],
            "sensitivity_among_valid",
            "specificity_among_valid",
            "Among valid outputs",
        ),
    ):
        sens_err = None
        spec_err = None
        if f"{sens_col}_ci_low" in df.columns and f"{sens_col}_ci_high" in df.columns:
            sens_err = err_bars(df[sens_col], df[f"{sens_col}_ci_low"], df[f"{sens_col}_ci_high"])
        if f"{spec_col}_ci_low" in df.columns and f"{spec_col}_ci_high" in df.columns:
            spec_err = err_bars(df[spec_col], df[f"{spec_col}_ci_low"], df[f"{spec_col}_ci_high"])

        ax.bar(
            x - width / 2,
            df[sens_col].fillna(0),
            width,
            yerr=sens_err,
            label="Sensitivity",
            color=COLORS["sensitivity"],
            alpha=0.9,
            capsize=3,
            error_kw={"elinewidth": 1.2},
        )
        ax.bar(
            x + width / 2,
            df[spec_col].fillna(0),
            width,
            yerr=spec_err,
            label="Specificity",
            color=COLORS["specificity"],
            alpha=0.9,
            capsize=3,
            error_kw={"elinewidth": 1.2},
        )
        ax.set_xticks(x)
        ax.set_xticklabels(df["model_name"].astype(str), rotation=40, ha="right")
        ax.set_ylim(0, 1.25)
        ax.set_title(title, fontsize=FONT["subtitle"])
        ax.legend()

    has_ci = "sensitivity_end_to_end_ci_low" in df.columns
    fig.suptitle(
        f"{TASK_TITLES[task]}: Sensitivity and Specificity"
        + (" with 95% Bootstrap CIs" if has_ci else "")
    )
    fig.tight_layout()
    return save_figure(fig, out_dir / f"{task}_sensitivity_specificity_dual.png")


def plot_triage_class_sens_spec(class_rates: pd.DataFrame, out_dir: Path) -> Path | None:
    """
    Dual end-to-end / among-valid sensitivity & specificity by triage class.

    Layout: 2 rows (definitions) x 3 columns (High Risk, Low Risk, Unclear).
    """
    if class_rates is None or class_rates.empty:
        return None
    required = {"model_name", "definition", "class", "tpr", "tnr"}
    if not required.issubset(class_rates.columns):
        return None

    definitions = [
        ("end_to_end", "End-to-end (invalid treated as incorrect)"),
        ("among_valid", "Among valid outputs"),
    ]
    models = ordered_models(class_rates["model_name"])

    fig, axes = plt.subplots(2, 3, figsize=(18, 11), sharey=True)
    x = np.arange(len(models))
    width = 0.35
    labeled_rates = with_display_names(class_rates)

    for row_idx, (definition, row_title) in enumerate(definitions):
        for col_idx, cls in enumerate(TRIAGE_CLASSES):
            ax = axes[row_idx, col_idx]
            subset = labeled_rates[
                (labeled_rates["definition"] == definition) & (labeled_rates["class"] == cls)
            ].set_index("model_name")
            tpr = pd.Series([subset["tpr"].get(m, np.nan) for m in models], dtype=float)
            tnr = pd.Series([subset["tnr"].get(m, np.nan) for m in models], dtype=float)

            tpr_err = None
            tnr_err = None
            if "tpr_ci_low" in subset.columns and "tpr_ci_high" in subset.columns:
                tpr_low = pd.Series(
                    [subset["tpr_ci_low"].get(m, np.nan) for m in models], dtype=float
                )
                tpr_high = pd.Series(
                    [subset["tpr_ci_high"].get(m, np.nan) for m in models], dtype=float
                )
                tpr_err = err_bars(tpr, tpr_low, tpr_high)
            if "tnr_ci_low" in subset.columns and "tnr_ci_high" in subset.columns:
                tnr_low = pd.Series(
                    [subset["tnr_ci_low"].get(m, np.nan) for m in models], dtype=float
                )
                tnr_high = pd.Series(
                    [subset["tnr_ci_high"].get(m, np.nan) for m in models], dtype=float
                )
                tnr_err = err_bars(tnr, tnr_low, tnr_high)

            ax.bar(
                x - width / 2,
                tpr.fillna(0),
                width,
                yerr=tpr_err,
                label="Sensitivity (TPR)",
                color=COLORS["sensitivity"],
                alpha=0.9,
                capsize=2,
                error_kw={"elinewidth": 1.0},
            )
            ax.bar(
                x + width / 2,
                tnr.fillna(0),
                width,
                yerr=tnr_err,
                label="Specificity (TNR)",
                color=COLORS["specificity"],
                alpha=0.9,
                capsize=2,
                error_kw={"elinewidth": 1.0},
            )
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=40, ha="right", fontsize=FONT["heatmap"])
            ax.set_ylim(0, 1.25)
            if row_idx == 0:
                ax.set_title(cls, fontsize=FONT["subtitle"])
            if col_idx == 0:
                ax.set_ylabel(row_title, fontsize=FONT["annot"])
            if row_idx == 0 and col_idx == 2:
                ax.legend(loc="upper right", fontsize=FONT["annot"])

    fig.suptitle("TBI Triage: Class-wise Sensitivity and Specificity with 95% Bootstrap CIs")
    fig.tight_layout()
    return save_figure(fig, out_dir / "triage_sensitivity_specificity_by_class.png")
