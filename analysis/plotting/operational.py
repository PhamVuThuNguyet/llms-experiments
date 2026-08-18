"""Latency, cost, and success/fail rate plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.plotting.model_metadata import (
    compute_model_costs,
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
    save_figure,
)


def plot_latency_and_cost(df: pd.DataFrame, task: str, out_dir: Path) -> list[Path]:
    paths = []
    labeled = with_display_names(df)
    latency = labeled.groupby("model_name")["total_latency_ms"].mean()
    latency = latency.reindex(ordered_models(latency.index))
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    bars = ax.barh(latency.index.astype(str), latency.values, color=COLORS["latency"], alpha=0.9)
    ax.set_xlabel("Average Latency (ms)")
    ax.set_title(f"{TASK_TITLES[task]}: Average Latency by Model")
    for bar, value in zip(bars, latency.values):
        ax.text(
            value,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.0f}",
            va="center",
            fontsize=FONT["annot"],
        )
    paths.append(save_figure(fig, out_dir / f"{task}_average_latency.png"))

    cost_df = compute_model_costs(df)
    if not cost_df.empty:
        cost_df["model_name"] = pd.Categorical(
            cost_df["model_name"],
            categories=ordered_models(cost_df["model_name"]),
            ordered=True,
        )
        cost_df = cost_df.sort_values("model_name")
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        bars = ax.barh(
            cost_df["model_name"].astype(str),
            cost_df["avg_total_cost"],
            color=COLORS["cost"],
            alpha=0.9,
        )
        ax.set_xlabel("Average Cost per Call (USD)")
        ax.set_title(f"{TASK_TITLES[task]}: Average Cost per Call")
        for bar, value in zip(bars, cost_df["avg_total_cost"]):
            ax.text(
                value,
                bar.get_y() + bar.get_height() / 2,
                f"${value:.4f}",
                va="center",
                fontsize=FONT["annot"],
            )
        paths.append(save_figure(fig, out_dir / f"{task}_average_cost_per_call.png"))
    return paths


def plot_cost_vs_accuracy(
    df: pd.DataFrame, metrics: pd.DataFrame, task: str, out_dir: Path
) -> Path | None:
    cost_df = compute_model_costs(df)
    if cost_df.empty:
        return None
    metrics_named = with_display_names(metrics)
    merged = cost_df.merge(
        metrics_named[["model_name", "accuracy_end_to_end"]], on="model_name"
    )
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.scatter(
        merged["avg_total_cost"],
        merged["accuracy_end_to_end"],
        s=110,
        color=COLORS["scatter"],
        alpha=0.9,
        zorder=3,
    )
    for row in merged.itertuples(index=False):
        ax.annotate(
            row.model_name,
            (row.avg_total_cost, row.accuracy_end_to_end),
            fontsize=FONT["annot"],
            xytext=(5, 5),
            textcoords="offset points",
        )
    ax.set_xlabel("Average Cost per Call (USD)")
    ax.set_ylabel("End-to-end Accuracy")
    ax.set_title(f"{TASK_TITLES[task]}: Cost vs End-to-end Accuracy")
    return save_figure(fig, out_dir / f"{task}_cost_vs_accuracy_scatter.png")


def plot_success_fail_rates(metrics: pd.DataFrame, task: str, out_dir: Path) -> Path:
    df = sort_by_model_order(metrics)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGURE_SIZE_WIDE)
    x = np.arange(len(df))
    ax1.bar(x, df["completion_rate"], color=COLORS["success"], alpha=0.9)
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["model_name"].astype(str), rotation=40, ha="right")
    ax1.set_ylim(0, 1.2)
    ax1.set_title("Task Completion Rate\n(Valid JSON / Total)", fontsize=FONT["subtitle"])
    ax1.set_ylabel("Rate")
    for i, value in enumerate(df["completion_rate"]):
        ax1.text(i, value + 0.02, f"{value:.1%}", ha="center", fontsize=FONT["annot"])

    fail = 1.0 - df["completion_rate"]
    ax2.bar(x, fail, color=COLORS["fail"], alpha=0.9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(df["model_name"].astype(str), rotation=40, ha="right")
    ax2.set_ylim(0, 1.2)
    ax2.set_title(
        "Schema / Parse Fail Rate\n(Invalid Outputs / Total)", fontsize=FONT["subtitle"]
    )
    ax2.set_ylabel("Rate")
    for i, value in enumerate(fail):
        ax2.text(i, value + 0.02, f"{value:.1%}", ha="center", fontsize=FONT["annot"])

    fig.suptitle(f"{TASK_TITLES[task]}: Success / Fail Rates")
    fig.tight_layout()
    return save_figure(fig, out_dir / f"{task}_model_success_fail_rates.png")
