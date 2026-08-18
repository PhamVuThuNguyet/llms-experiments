"""Model display names, ordering, pricing, and cost helpers."""

from __future__ import annotations

import pandas as pd

from analysis.constants import TOKENS_PER_MILLION

MODEL_DISPLAY_NAMES = {
    "gpt-4.1": "gpt-4.1",
    "gpt-4o": "gpt-4o",
    "o1": "o1",
    "gpt-5": "gpt-5",
    "gpt-5-chat-latest": "gpt-5-chat-latest",
    "claude-opus-4-20250514": "claude-opus-4-20250514",
    "claude-opus-4-1-20250805": "claude-opus-4-1-20250805",
    "claude-sonnet-4-20250514": "claude-sonnet-4-20250514",
    "claude-3-7-sonnet-20250219": "claude-3-7-sonnet-20250219",
    "claude-3-7-sonnet-latest": "claude-3-7-sonnet-20250219",
    "grok-4-fast-reasoning": "grok-4-fast-reasoning",
}

MODEL_ORDER = [
    "gpt-4.1",
    "gpt-4o",
    "o1",
    "gpt-5",
    "gpt-5-chat-latest",
    "claude-opus-4-20250514",
    "claude-opus-4-1-20250805",
    "claude-sonnet-4-20250514",
    "claude-3-7-sonnet-20250219",
    "grok-4-fast-reasoning",
]

# USD per 1M tokens (same table as eda-and-report notebooks)
PRICING = {
    "gpt-4o": {"input": 2.5, "output": 10.00},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-5": {"input": 1.25, "output": 10.00},
    "gpt-5-chat-latest": {"input": 1.25, "output": 10.00},
    "o1": {"input": 15.00, "output": 60.00},
    "claude-opus-4-20250514": {"input": 15.00, "output": 75.00},
    "claude-opus-4-1-20250805": {"input": 15.00, "output": 75.00},
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "claude-3-7-sonnet-latest": {"input": 3.00, "output": 15.00},
    "claude-3-7-sonnet-20250219": {"input": 3.00, "output": 15.00},
    "grok-4-fast-reasoning": {"input": 0.2, "output": 0.5},
}


def display_name(name: str) -> str:
    return MODEL_DISPLAY_NAMES.get(str(name), str(name))


def with_display_names(df: pd.DataFrame, column: str = "model_name") -> pd.DataFrame:
    out = df.copy()
    out[column] = out[column].map(display_name)
    return out


def ordered_models(names) -> list[str]:
    displayed = [display_name(n) for n in names]
    unique = list(dict.fromkeys(displayed))
    ordered = [m for m in MODEL_ORDER if m in unique]
    extras = [m for m in unique if m not in ordered]
    return ordered + extras


def sort_by_model_order(df: pd.DataFrame, column: str = "model_name") -> pd.DataFrame:
    out = with_display_names(df, column)
    order = ordered_models(out[column])
    out[column] = pd.Categorical(out[column], categories=order, ordered=True)
    return out.sort_values(column)


def compute_model_costs(df: pd.DataFrame) -> pd.DataFrame:
    """Average USD cost per call for models with known pricing."""
    cost_rows = []
    for model_name, model_df in df.groupby("model_name"):
        prices = PRICING.get(model_name)
        if not prices:
            continue
        input_cost = (model_df["input_tokens"].fillna(0) / TOKENS_PER_MILLION) * prices[
            "input"
        ]
        output_cost = (model_df["output_tokens"].fillna(0) / TOKENS_PER_MILLION) * prices[
            "output"
        ]
        cost_rows.append(
            {
                "model_name": display_name(model_name),
                "avg_total_cost": float((input_cost + output_cost).mean()),
            }
        )
    return pd.DataFrame(cost_rows)
