"""Shared matplotlib style constants for manuscript figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {
    "end_to_end": "#2C5F8A",
    "among_valid": "#3D8B66",
    "sensitivity": "#C0392B",
    "specificity": "#2471A3",
    "completion": "#D35400",
    "fail": "#922B21",
    "exact": "#2C5F8A",
    "semantic": "#6C3483",
    "latency": "#117A65",
    "cost": "#B7950B",
    "success": "#1E8449",
    "scatter": "#2C5F8A",
    "heatmap_diff": "RdBu_r",
    "heatmap_p": "YlGnBu_r",
    "confusion": "Blues",
}

ERROR_TYPE_COLORS = {
    "invalid_schema": "#7F8C8D",
    "false_negative": "#C0392B",
    "false_positive": "#E67E22",
    "misclassification": "#8E44AD",
}

FONT = {
    "title": 15,
    "subtitle": 12,
    "axis": 12,
    "tick": 10,
    "legend": 10,
    "annot": 9,
    "heatmap": 7,
    "confusion": 8,
}

FIGURE_SIZE = (14, 8)
FIGURE_SIZE_WIDE = (16, 8)
FIGURE_SIZE_HEATMAP = (12, 10)

TASK_TITLES = {
    "mls": "Midline Shift Detection",
    "bf": "Burst Fracture Detection",
    "triage": "TBI Triage",
}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica", "sans-serif"],
            "font.size": FONT["tick"],
            "axes.titlesize": FONT["title"],
            "axes.titleweight": "bold",
            "axes.labelsize": FONT["axis"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": FONT["tick"],
            "ytick.labelsize": FONT["tick"],
            "legend.fontsize": FONT["legend"],
            "legend.frameon": False,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linestyle": "--",
            "axes.axisbelow": True,
        }
    )


def save_figure(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def err_bars(point: pd.Series, low: pd.Series, high: pd.Series) -> np.ndarray:
    low = low.fillna(point)
    high = high.fillna(point)
    err = np.vstack([(point - low).clip(lower=0), (high - point).clip(lower=0)])
    return np.nan_to_num(err, nan=0.0)
