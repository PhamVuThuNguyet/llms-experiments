"""Appendix table formatting helpers."""

from __future__ import annotations

import pandas as pd

from analysis.constants import (
    APPENDIX_INTEGER_COLUMNS,
    APPENDIX_P_VALUE_COLUMNS,
    P_VALUE_DISPLAY_FLOOR,
)


def format_appendix_p_value(
    value: float,
    *,
    floor: float = P_VALUE_DISPLAY_FLOOR,
    decimals: int = 2,
) -> str:
    """Format a p-value for appendix tables; very small values show as <0.01."""
    if pd.isna(value):
        return ""
    if value < floor:
        return f"<{floor:.{decimals}f}"
    return f"{value:.{decimals}f}"


def format_appendix_p_values(df: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    """Convert McNemar p-value columns to manuscript-friendly strings."""
    out = df.copy()
    for col in APPENDIX_P_VALUE_COLUMNS:
        if col not in out.columns:
            continue
        out[col] = out[col].map(
            lambda value: format_appendix_p_value(value, decimals=decimals)
        )
    return out


def format_mcnemar_pvalue_matrix(matrix: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    """Format symmetric McNemar p-matrix; diagonal entries are not applicable."""
    out = matrix.astype(object).copy()
    for row_label in out.index:
        for col_label in out.columns:
            if row_label == col_label:
                out.loc[row_label, col_label] = "NA"
            else:
                out.loc[row_label, col_label] = format_appendix_p_value(
                    out.loc[row_label, col_label],
                    decimals=decimals,
                )
    return out


def round_appendix_table(df: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    """Round numeric appendix columns for manuscript-ready CSV export."""
    out = df.copy()
    for col in out.columns:
        if col in APPENDIX_INTEGER_COLUMNS:
            if pd.api.types.is_numeric_dtype(out[col]):
                out[col] = out[col].round(0).astype("Int64")
            continue
        if col in APPENDIX_P_VALUE_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].round(decimals)
    return out
