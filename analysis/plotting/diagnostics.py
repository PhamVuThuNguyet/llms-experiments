"""Consistency, failure, heatmap, and confusion-matrix plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from analysis.constants import INVALID_PREDICTION_SENTINEL
from analysis.loaders import TASK_CONFIGS
from analysis.plotting.model_metadata import (
    display_name,
    ordered_models,
    sort_by_model_order,
    with_display_names,
)
from analysis.plotting.style import (
    COLORS,
    ERROR_TYPE_COLORS,
    FIGURE_SIZE,
    FIGURE_SIZE_HEATMAP,
    FONT,
    TASK_TITLES,
    save_figure,
)


def plot_self_consistency(consistency: pd.DataFrame, task: str, out_dir: Path) -> Path:
    df = sort_by_model_order(consistency)
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    y = np.arange(len(df))
    height = 0.35
    ax.barh(
        y - height / 2,
        df["exact_consistency"].fillna(0),
        height,
        label="Exact string match",
        color=COLORS["exact"],
        alpha=0.9,
    )
    ax.barh(
        y + height / 2,
        df["semantic_consistency"].fillna(0),
        height,
        label="Semantic JSON equality",
        color=COLORS["semantic"],
        alpha=0.9,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(df["model_name"].astype(str))
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Consistency Rate")
    ax.set_title(f"{TASK_TITLES[task]}: Exact vs Semantic Self-Consistency")
    ax.legend(loc="lower right")
    return save_figure(fig, out_dir / f"{task}_self_consistency_exact_vs_semantic.png")


def plot_paired_heatmaps(paired: pd.DataFrame, task: str, out_dir: Path) -> list[Path]:
    if paired.empty:
        return []
    paired = paired.copy()
    paired["model_a"] = paired["model_a"].map(display_name)
    paired["model_b"] = paired["model_b"].map(display_name)
    models = ordered_models(set(paired["model_a"]) | set(paired["model_b"]))
    index = {name: i for i, name in enumerate(models)}
    diff = np.full((len(models), len(models)), np.nan)
    pvals = np.full((len(models), len(models)), np.nan)
    for row in paired.itertuples(index=False):
        i, j = index[row.model_a], index[row.model_b]
        diff[i, j] = row.diff_b_minus_a
        diff[j, i] = -row.diff_b_minus_a
        pvals[i, j] = row.mcnemar_p_value
        pvals[j, i] = row.mcnemar_p_value
    np.fill_diagonal(diff, 0.0)
    np.fill_diagonal(pvals, 1.0)

    paths = []
    for matrix, title, fname, cmap in (
        (
            diff,
            "Paired Accuracy Difference (row → column)",
            f"{task}_paired_accuracy_diff_heatmap.png",
            COLORS["heatmap_diff"],
        ),
        (
            pvals,
            "McNemar Exact P-values",
            f"{task}_mcnemar_pvalue_heatmap.png",
            COLORS["heatmap_p"],
        ),
    ):
        fig, ax = plt.subplots(figsize=FIGURE_SIZE_HEATMAP)
        im = ax.imshow(matrix, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(models)))
        ax.set_yticks(range(len(models)))
        ax.set_xticklabels(models, rotation=40, ha="right", fontsize=FONT["annot"])
        ax.set_yticklabels(models, fontsize=FONT["annot"])
        ax.set_title(f"{TASK_TITLES[task]}: {title}")
        ax.grid(False)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=FONT["tick"])
        for i in range(len(models)):
            for j in range(len(models)):
                value = matrix[i, j]
                if np.isnan(value):
                    continue
                text = (
                    f"{value:.2f}"
                    if fname.endswith("diff_heatmap.png")
                    else ("1" if value >= 0.999 else f"{value:.2g}")
                )
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    fontsize=FONT["heatmap"],
                    color="black",
                )
        paths.append(save_figure(fig, out_dir / fname))
    return paths


def plot_failure_breakdown(failures: pd.DataFrame, task: str, out_dir: Path) -> Path | None:
    if failures.empty or "error_type" not in failures.columns:
        return None
    labeled = with_display_names(failures)
    counts = labeled.groupby(["model_name", "error_type"]).size().unstack(fill_value=0)
    counts = counts.reindex(ordered_models(counts.index))
    preferred = [
        c
        for c in ["invalid_schema", "false_negative", "false_positive", "misclassification"]
        if c in counts.columns
    ]
    other = [c for c in counts.columns if c not in preferred]
    counts = counts[preferred + other]
    colors = [ERROR_TYPE_COLORS.get(col, "#5D6D7E") for col in counts.columns]
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    counts.plot(kind="bar", stacked=True, ax=ax, color=colors, width=0.75, legend=True)
    ax.set_xlabel("Model")
    ax.set_ylabel("Number of Failure Cases")
    ax.set_title(f"{TASK_TITLES[task]}: Qualitative Failure-Type Breakdown")
    ax.legend(title="Error type")
    ax.tick_params(axis="x", rotation=40)
    fig.tight_layout()
    return save_figure(fig, out_dir / f"{task}_failure_type_breakdown.png")


def plot_confusion_matrices_with_invalid(
    df: pd.DataFrame, task: str, out_dir: Path
) -> list[Path]:
    config = TASK_CONFIGS[task]
    pred_col = config["prediction_col"]
    labeled = df[df["ground_truth"].notna()].copy()
    if labeled.empty:
        return []

    if config["binary"]:
        labels = [0, 1, INVALID_PREDICTION_SENTINEL]
        label_names = ["Negative (0)", "Positive (1)", "Invalid (-1)"]
        class_to_id = None
        classes = None
    else:
        classes = ["High Risk", "Low Risk", "Unclear", "Invalid"]
        labels = list(range(len(classes)))
        label_names = classes
        class_to_id = {name: i for i, name in enumerate(classes[:-1])}

    models = ordered_models(labeled["model_name"].unique())
    n_cols = 5
    n_rows = int(np.ceil(len(models) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.8 * n_cols, 3.8 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, model_name in zip(axes, models):
        raw_names = [
            raw for raw in labeled["model_name"].unique() if display_name(raw) == model_name
        ]
        model_df = labeled[labeled["model_name"].isin(raw_names)]
        if config["binary"]:
            y_true = model_df["ground_truth"].astype(int)
            y_pred = model_df[pred_col].fillna(INVALID_PREDICTION_SENTINEL).astype(int)
        else:
            y_true = model_df["ground_truth"].map(class_to_id).astype(int)
            y_pred = (
                model_df[pred_col]
                .map(class_to_id)
                .fillna(len(classes) - 1)
                .astype(int)
            )
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        ax.imshow(cm, cmap=COLORS["confusion"])
        ax.set_title(model_name, fontsize=FONT["annot"])
        ax.set_xticks(range(len(label_names)))
        ax.set_yticks(range(len(label_names)))
        ax.set_xticklabels(label_names, rotation=40, ha="right", fontsize=FONT["heatmap"])
        ax.set_yticklabels(label_names, fontsize=FONT["heatmap"])
        ax.grid(False)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    fontsize=FONT["confusion"],
                )
    for ax in axes[len(models) :]:
        ax.axis("off")
    fig.suptitle(f"{TASK_TITLES[task]}: Confusion Matrices (including Invalid)")
    fig.tight_layout()
    return [save_figure(fig, out_dir / f"{task}_confusion_matrix_all_models_with_invalid.png")]
