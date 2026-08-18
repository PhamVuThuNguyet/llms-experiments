"""Shared analysis constants and path roots."""

from __future__ import annotations

from pathlib import Path

ANALYSIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ANALYSIS_DIR.parent
REPO_ROOT = PROJECT_ROOT.parent

DEFAULT_BOOTSTRAP_SAMPLES = 2000
DEFAULT_BOOTSTRAP_ALPHA = 0.05
DEFAULT_BOOTSTRAP_SEED = 0
P_VALUE_DISPLAY_FLOOR = 0.01
TOKENS_PER_MILLION = 1_000_000
INVALID_PREDICTION_SENTINEL = -1

APPENDIX_INTEGER_COLUMNS = frozenset(
    {
        "n",
        "n_total",
        "n_valid",
        "n_eval",
        "n_calls",
        "n_multi_run_cases",
        "n_exact_agree",
        "n_semantic_agree",
        "a_only_correct",
        "b_only_correct",
        "tp",
        "fp",
        "fn",
        "tn",
        "tp_valid",
        "tn_valid",
        "fp_valid",
        "fn_valid",
    }
)

APPENDIX_P_VALUE_COLUMNS = frozenset({"mcnemar_p_value"})

TRIAGE_CLASSES = ("High Risk", "Low Risk", "Unclear")
