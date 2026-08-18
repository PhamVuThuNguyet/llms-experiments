"""Failure exports, rate tables, and McNemar matrices."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from analysis.constants import DEFAULT_BOOTSTRAP_SAMPLES, TRIAGE_CLASSES
from analysis.loaders import TASK_CONFIGS
from analysis.metrics import _safe_rate, bootstrap_ci


def export_failures(task: str, df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    config = TASK_CONFIGS[task]
    pred_col = config["prediction_col"]
    labeled = df[df["ground_truth"].notna()].copy()
    labeled["parse_valid"] = labeled[pred_col].notna()
    if config["binary"]:
        labeled["correct"] = (labeled[pred_col] == labeled["ground_truth"]) & labeled[
            "parse_valid"
        ]
        labeled["error_type"] = np.where(
            ~labeled["parse_valid"],
            "invalid_schema",
            np.where(
                labeled["correct"],
                "none",
                np.where(labeled["ground_truth"] == 1, "false_negative", "false_positive"),
            ),
        )
    else:
        labeled["correct"] = (labeled[pred_col] == labeled["ground_truth"]) & labeled[
            "parse_valid"
        ]
        labeled["error_type"] = np.where(
            ~labeled["parse_valid"],
            "invalid_schema",
            np.where(labeled["correct"], "none", "misclassification"),
        )

    failures = labeled[labeled["error_type"] != "none"].copy()
    cols = [
        "task",
        "model_name",
        "experiment_id",
        "version",
        "ground_truth",
        pred_col,
        "error_type",
        "response_text",
        "input_image_path",
        "total_latency_ms",
    ]
    failures.insert(0, "task", task)
    keep = [col for col in cols if col in failures.columns]
    failures = failures[keep]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    failures.to_csv(output_path, index=False)
    return failures


def export_binary_rates_table(task: str, metrics: pd.DataFrame) -> pd.DataFrame:
    """Long-format FPR/FNR and related rates for appendix tables."""
    rate_cols = [
        c
        for c in metrics.columns
        if any(
            c.startswith(prefix)
            for prefix in (
                "sensitivity_",
                "specificity_",
                "fpr_",
                "fnr_",
                "precision_",
                "f1_",
            )
        )
        and not c.endswith(("_ci_low", "_ci_high"))
    ]
    rows = []
    for _, row in metrics.iterrows():
        for col in rate_cols:
            rows.append(
                {
                    "task": task,
                    "model_name": row["model_name"],
                    "metric": col,
                    "point": row[col],
                    "ci_low": row.get(f"{col}_ci_low"),
                    "ci_high": row.get(f"{col}_ci_high"),
                }
            )
    return pd.DataFrame(rows)


def _empty_class_rate_row(
    task: str,
    model_name: str,
    definition: str,
    cls: str,
) -> dict:
    return {
        "task": task,
        "model_name": model_name,
        "definition": definition,
        "class": cls,
        "n_eval": 0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "tn": 0,
        "tpr": float("nan"),
        "tpr_ci_low": float("nan"),
        "tpr_ci_high": float("nan"),
        "fpr": float("nan"),
        "fpr_ci_low": float("nan"),
        "fpr_ci_high": float("nan"),
        "tnr": float("nan"),
        "tnr_ci_low": float("nan"),
        "tnr_ci_high": float("nan"),
        "fnr": float("nan"),
        "fnr_ci_low": float("nan"),
        "fnr_ci_high": float("nan"),
    }


def _bootstrap_rate_ci(
    vals: np.ndarray,
    point: float,
    n_boot: int,
    seed: int,
) -> tuple[float, float, float]:
    if vals.size:
        return bootstrap_ci(vals, n_boot=n_boot, seed=seed)
    return point, float("nan"), float("nan")


def _class_confusion(
    eval_true: np.ndarray,
    eval_pred: np.ndarray,
    cls: str,
) -> tuple[int, int, int, int, np.ndarray, np.ndarray]:
    is_true = eval_true == cls
    is_pred = np.array(
        [(p == cls) if p is not None else False for p in eval_pred],
        dtype=bool,
    )
    tp = int(np.sum(is_true & is_pred))
    fn = int(np.sum(is_true & ~is_pred))
    fp = int(np.sum(~is_true & is_pred))
    tn = int(np.sum(~is_true & ~is_pred))
    return tp, fp, fn, tn, is_true, is_pred


def export_triage_class_rates(
    task: str,
    df: pd.DataFrame,
    n_boot: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Per-class TPR/FPR/TNR/FNR for triage under dual definitions.

    End-to-end: invalid predictions count as incorrect (FN for the true class).
    Among-valid: only successfully parsed predictions.
    Bootstrap 95% CIs are reported for TPR and TNR.
    """
    pred_col = TASK_CONFIGS[task]["prediction_col"]
    rows = []
    labeled = df[df["ground_truth"].notna()].copy()

    for model_name, model_df in labeled.groupby("model_name"):
        true = model_df["ground_truth"].astype(str).to_numpy()
        pred_raw = model_df[pred_col]
        valid_mask = pred_raw.notna().to_numpy()
        pred = pred_raw.astype(object).where(pred_raw.notna(), other=None).to_numpy()

        for definition, use_invalid_as_incorrect in (
            ("end_to_end", True),
            ("among_valid", False),
        ):
            if use_invalid_as_incorrect:
                eval_true = true
                eval_pred = pred
            else:
                eval_true = true[valid_mask]
                eval_pred = pred[valid_mask]
                if eval_true.size == 0:
                    for cls in TRIAGE_CLASSES:
                        rows.append(
                            _empty_class_rate_row(task, model_name, definition, cls)
                        )
                    continue

            for cls in TRIAGE_CLASSES:
                tp, fp, fn, tn, is_true, is_pred = _class_confusion(
                    eval_true, eval_pred, cls
                )
                tpr = _safe_rate(tp, tp + fn)
                fpr = _safe_rate(fp, fp + tn)
                tnr = _safe_rate(tn, tn + fp)
                fnr = _safe_rate(fn, fn + tp)

                pos_idx = np.where(is_true)[0]
                neg_idx = np.where(~is_true)[0]
                sens_vals = is_pred[pos_idx].astype(float) if pos_idx.size else np.asarray([])
                spec_vals = (~is_pred[neg_idx]).astype(float) if neg_idx.size else np.asarray([])
                fpr_vals = is_pred[neg_idx].astype(float) if neg_idx.size else np.asarray([])
                fnr_vals = (~is_pred[pos_idx]).astype(float) if pos_idx.size else np.asarray([])

                cls_i = TRIAGE_CLASSES.index(cls) + 1
                base = 0 if definition == "end_to_end" else 100
                tpr_point, tpr_low, tpr_high = _bootstrap_rate_ci(
                    sens_vals, tpr, n_boot, seed + 17 * cls_i + base
                )
                tnr_point, tnr_low, tnr_high = _bootstrap_rate_ci(
                    spec_vals, tnr, n_boot, seed + 31 * cls_i + base
                )
                fpr_point, fpr_low, fpr_high = _bootstrap_rate_ci(
                    fpr_vals, fpr, n_boot, seed + 41 * cls_i + base
                )
                fnr_point, fnr_low, fnr_high = _bootstrap_rate_ci(
                    fnr_vals, fnr, n_boot, seed + 53 * cls_i + base
                )

                rows.append(
                    {
                        "task": task,
                        "model_name": model_name,
                        "definition": definition,
                        "class": cls,
                        "n_eval": int(len(eval_true)),
                        "tp": tp,
                        "fp": fp,
                        "fn": fn,
                        "tn": tn,
                        "tpr": tpr_point,
                        "tpr_ci_low": tpr_low,
                        "tpr_ci_high": tpr_high,
                        "fpr": fpr_point,
                        "fpr_ci_low": fpr_low,
                        "fpr_ci_high": fpr_high,
                        "tnr": tnr_point,
                        "tnr_ci_low": tnr_low,
                        "tnr_ci_high": tnr_high,
                        "fnr": fnr_point,
                        "fnr_ci_low": fnr_low,
                        "fnr_ci_high": fnr_high,
                    }
                )

    return pd.DataFrame(rows)


def build_mcnemar_matrix(paired: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Symmetric McNemar p-value and accuracy-difference matrices from paired comparisons."""
    models = sorted(set(paired["model_a"]).union(set(paired["model_b"])))
    p_matrix = pd.DataFrame(np.nan, index=models, columns=models)
    diff_matrix = pd.DataFrame(np.nan, index=models, columns=models)
    for model in models:
        p_matrix.loc[model, model] = 0.0
        diff_matrix.loc[model, model] = 0.0
    for _, row in paired.iterrows():
        a, b = row["model_a"], row["model_b"]
        p_matrix.loc[a, b] = row["mcnemar_p_value"]
        p_matrix.loc[b, a] = row["mcnemar_p_value"]
        diff_matrix.loc[a, b] = row["diff_b_minus_a"]
        diff_matrix.loc[b, a] = -row["diff_b_minus_a"]
    return p_matrix, diff_matrix
