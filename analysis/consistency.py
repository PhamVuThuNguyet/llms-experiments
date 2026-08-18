"""Exact and semantic self-consistency metrics across runs."""

from __future__ import annotations

import pandas as pd

from analysis.metrics import _safe_rate
from analysis.parsing import normalize_response_text, responses_semantically_equal


def _consistency_by_model(
    df: pd.DataFrame,
    metric_name: str,
    agrees,
) -> pd.DataFrame:
    rows = []
    for model_name, model_df in df.groupby("model_name"):
        case_ok = 0
        case_total = 0
        for _, case_df in model_df.groupby("experiment_id"):
            texts = case_df["response_text"].tolist()
            if len(texts) < 2:
                continue
            case_total += 1
            if agrees(texts):
                case_ok += 1
        rows.append(
            {
                "model_name": model_name,
                "n_multi_run_cases": case_total,
                metric_name: _safe_rate(case_ok, case_total),
                "n_agree": case_ok,
            }
        )
    return pd.DataFrame(rows)


def semantic_self_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per model: fraction of cases where all available runs agree semantically.

    Expects columns: model_name, experiment_id, response_text, and optionally version.
    """

    def agrees(texts: list) -> bool:
        norms = [normalize_response_text(text) for text in texts]
        if any(item is None for item in norms):
            return False
        first = texts[0]
        return all(responses_semantically_equal(first, other) for other in texts[1:])

    return _consistency_by_model(df, "semantic_consistency", agrees)


def exact_self_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """Exact character-level response_text identity across runs."""

    def agrees(texts: list) -> bool:
        normalized = [str(t) if pd.notna(t) else "" for t in texts]
        return len(set(normalized)) == 1

    return _consistency_by_model(df, "exact_consistency", agrees)
