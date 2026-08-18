"""Diagnostic and operational metrics with bootstrap uncertainty."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import comb
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score

from analysis.constants import (
    DEFAULT_BOOTSTRAP_ALPHA,
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
)


@dataclass(frozen=True)
class MetricEstimate:
    name: str
    point: float
    ci_low: float | None = None
    ci_high: float | None = None
    n: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _safe_rate(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return float("nan")
    return float(numerator) / float(denominator)


def bootstrap_ci(
    values: Iterable[float],
    statistic=np.mean,
    n_boot: int = DEFAULT_BOOTSTRAP_SAMPLES,
    alpha: float = DEFAULT_BOOTSTRAP_ALPHA,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    """Return (point, ci_low, ci_high) for a 1-D array of case-level values."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(statistic(arr))
    if arr.size == 1:
        return point, point, point
    rng = np.random.default_rng(seed)
    samples = np.empty(n_boot, dtype=float)
    for index in range(n_boot):
        draw = rng.choice(arr, size=arr.size, replace=True)
        samples[index] = float(statistic(draw))
    low = float(np.quantile(samples, alpha / 2.0))
    high = float(np.quantile(samples, 1.0 - alpha / 2.0))
    return point, low, high


def binary_confusion(y_true: Iterable, y_pred: Iterable) -> dict[str, int]:
    true = np.asarray(list(y_true), dtype=int)
    pred = np.asarray(list(y_pred), dtype=int)
    if true.size == 0:
        return {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
    tn, fp, fn, tp = confusion_matrix(true, pred, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def _empty_binary_metrics(n_total: int, n_valid: int, completion: float) -> dict:
    return {
        "n_total": n_total,
        "n_valid": n_valid,
        "completion_rate": MetricEstimate("completion_rate", completion, n=n_total),
        "accuracy": MetricEstimate("accuracy", float("nan"), n=0),
        "sensitivity": MetricEstimate("sensitivity", float("nan"), n=0),
        "specificity": MetricEstimate("specificity", float("nan"), n=0),
        "precision": MetricEstimate("precision", float("nan"), n=0),
        "f1": MetricEstimate("f1", float("nan"), n=0),
        "fpr": MetricEstimate("fpr", float("nan"), n=0),
        "fnr": MetricEstimate("fnr", float("nan"), n=0),
        "auc": None,
        "tp": 0,
        "tn": 0,
        "fp": 0,
        "fn": 0,
    }


def _prepare_binary_evaluation(
    true: pd.Series,
    pred: pd.Series,
    invalid_as_incorrect: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, int], float | None]:
    if invalid_as_incorrect:
        eval_true = true.to_numpy(dtype=float)
        eval_pred = pred.to_numpy(dtype=float)
        scored_pred = np.where(np.isnan(eval_pred), 1 - eval_true, eval_pred).astype(int)
        scored_true = eval_true.astype(int)
        return scored_true, scored_pred, binary_confusion(scored_true, scored_pred), None

    eval_true = true[pred.notna()].to_numpy(dtype=int)
    eval_pred = pred[pred.notna()].to_numpy(dtype=int)
    auc = None
    if eval_true.size and len(np.unique(eval_true)) > 1:
        try:
            auc = float(roc_auc_score(eval_true, eval_pred))
        except ValueError:
            auc = None
    return eval_true, eval_pred, binary_confusion(eval_true, eval_pred), auc


def _case_rate_indicators(
    scored_true: np.ndarray,
    scored_pred: np.ndarray,
) -> dict[str, np.ndarray]:
    pos = scored_true == 1
    neg = scored_true == 0
    pred_pos = scored_pred == 1
    return {
        "sens": (scored_pred[pos] == 1).astype(float) if np.any(pos) else np.asarray([]),
        "spec": (scored_pred[neg] == 0).astype(float) if np.any(neg) else np.asarray([]),
        "fpr": (scored_pred[neg] == 1).astype(float) if np.any(neg) else np.asarray([]),
        "fnr": (scored_pred[pos] == 0).astype(float) if np.any(pos) else np.asarray([]),
        "prec": (scored_true[pred_pos] == 1).astype(float) if np.any(pred_pos) else np.asarray([]),
    }


def _bootstrap_or_point(
    values: np.ndarray,
    point: float,
    n_boot: int,
    seed: int,
) -> tuple[float, float | None, float | None]:
    if values.size:
        return bootstrap_ci(values, n_boot=n_boot, seed=seed)
    return point, None, None


def _bootstrap_f1(
    true_arr: np.ndarray,
    pred_arr: np.ndarray,
    cm: dict[str, int],
    n_boot: int,
    seed: int,
) -> tuple[float, float, float]:
    point = _safe_rate(2 * cm["tp"], 2 * cm["tp"] + cm["fp"] + cm["fn"])
    if true_arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = np.empty(n_boot, dtype=float)
    for index in range(n_boot):
        idx = rng.choice(true_arr.size, size=true_arr.size, replace=True)
        draw_cm = binary_confusion(true_arr[idx], pred_arr[idx])
        samples[index] = _safe_rate(
            2 * draw_cm["tp"],
            2 * draw_cm["tp"] + draw_cm["fp"] + draw_cm["fn"],
        )
    low = float(np.quantile(samples, DEFAULT_BOOTSTRAP_ALPHA / 2.0))
    high = float(np.quantile(samples, 1.0 - DEFAULT_BOOTSTRAP_ALPHA / 2.0))
    return point, low, high


def compute_binary_metrics(
    y_true: Iterable,
    y_pred: Iterable,
    invalid_as_incorrect: bool = True,
    n_boot: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, MetricEstimate | float | int | None]:
    """
    Compute end-to-end or valid-only binary metrics.

    y_pred may contain NaN for invalid outputs. If invalid_as_incorrect,
    invalid predictions are scored as incorrect for accuracy / completion.
    """
    true = pd.Series(list(y_true), dtype="float")
    pred = pd.Series(list(y_pred), dtype="float")
    mask_labeled = true.notna()
    true = true[mask_labeled]
    pred = pred[mask_labeled]

    n_total = int(len(true))
    valid_mask = pred.notna()
    n_valid = int(valid_mask.sum())
    completion = _safe_rate(n_valid, n_total)

    if not invalid_as_incorrect and n_valid == 0:
        return _empty_binary_metrics(n_total, n_valid, completion)

    scored_true, scored_pred, cm, auc = _prepare_binary_evaluation(
        true, pred, invalid_as_incorrect
    )

    if invalid_as_incorrect:
        case_correct = ((pred.to_numpy(dtype=float) == true.to_numpy(dtype=float)) & ~np.isnan(pred.to_numpy(dtype=float))).astype(float)
    else:
        case_correct = (scored_pred == scored_true).astype(float)

    acc_point, acc_low, acc_high = bootstrap_ci(case_correct, n_boot=n_boot, seed=seed)

    sens = _safe_rate(cm["tp"], cm["tp"] + cm["fn"])
    spec = _safe_rate(cm["tn"], cm["tn"] + cm["fp"])
    fpr = _safe_rate(cm["fp"], cm["fp"] + cm["tn"])
    fnr = _safe_rate(cm["fn"], cm["fn"] + cm["tp"])
    prec = _safe_rate(cm["tp"], cm["tp"] + cm["fp"])

    indicators = _case_rate_indicators(scored_true, scored_pred)
    sens_point, sens_low, sens_high = _bootstrap_or_point(
        indicators["sens"], sens, n_boot, seed + 1
    )
    spec_point, spec_low, spec_high = _bootstrap_or_point(
        indicators["spec"], spec, n_boot, seed + 2
    )
    prec_point, prec_low, prec_high = _bootstrap_or_point(
        indicators["prec"], prec, n_boot, seed + 3
    )
    fpr_point, fpr_low, fpr_high = _bootstrap_or_point(
        indicators["fpr"], fpr, n_boot, seed + 6
    )
    fnr_point, fnr_low, fnr_high = _bootstrap_or_point(
        indicators["fnr"], fnr, n_boot, seed + 7
    )
    f1_point, f1_low, f1_high = _bootstrap_f1(
        scored_true, scored_pred, cm, n_boot, seed + 4
    )
    completion_point, completion_low, completion_high = bootstrap_ci(
        valid_mask.astype(float).to_numpy(),
        n_boot=n_boot,
        seed=seed + 5,
    )

    return {
        "n_total": n_total,
        "n_valid": n_valid,
        "completion_rate": MetricEstimate(
            "completion_rate", completion_point, completion_low, completion_high, n_total
        ),
        "accuracy": MetricEstimate(
            "accuracy",
            acc_point,
            acc_low,
            acc_high,
            n_total if invalid_as_incorrect else n_valid,
        ),
        "sensitivity": MetricEstimate("sensitivity", sens_point, sens_low, sens_high),
        "specificity": MetricEstimate("specificity", spec_point, spec_low, spec_high),
        "fpr": MetricEstimate("fpr", fpr_point, fpr_low, fpr_high),
        "fnr": MetricEstimate("fnr", fnr_point, fnr_low, fnr_high),
        "precision": MetricEstimate("precision", prec_point, prec_low, prec_high),
        "f1": MetricEstimate("f1", f1_point, f1_low, f1_high),
        "auc": auc,
        "tp": cm["tp"],
        "tn": cm["tn"],
        "fp": cm["fp"],
        "fn": cm["fn"],
    }


def _is_correct(pred, labels) -> np.ndarray:
    pred_series = pd.Series(pred)
    return pred_series.notna().to_numpy() & (pred_series.to_numpy() == labels)


def mcnemar_exact(
    y_true: Iterable,
    pred_a: Iterable,
    pred_b: Iterable,
) -> dict[str, float | int]:
    """
    Exact two-sided McNemar test on paired end-to-end correctness
    (invalid predictions count as incorrect).
    """
    true = pd.Series(list(y_true))
    a = pd.Series(list(pred_a))
    b = pd.Series(list(pred_b))
    mask = true.notna()
    true = true[mask].to_numpy()
    a = a[mask].to_numpy()
    b = b[mask].to_numpy()

    correct_a = _is_correct(a, true)
    correct_b = _is_correct(b, true)
    b_only = int(np.sum(~correct_a & correct_b))
    a_only = int(np.sum(correct_a & ~correct_b))

    n_disc = a_only + b_only
    if n_disc == 0:
        p_value = 1.0
    else:
        k = min(a_only, b_only)
        tail = sum(comb(n_disc, i) for i in range(0, k + 1)) / (2**n_disc)
        p_value = min(1.0, 2.0 * tail)

    return {
        "n": int(len(true)),
        "a_only_correct": a_only,
        "b_only_correct": b_only,
        "both_correct": int(np.sum(correct_a & correct_b)),
        "both_incorrect": int(np.sum(~correct_a & ~correct_b)),
        "p_value": float(p_value),
    }


def paired_bootstrap_accuracy_diff(
    y_true: Iterable,
    pred_a: Iterable,
    pred_b: Iterable,
    n_boot: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, float | int]:
    """Paired bootstrap CI for accuracy_B - accuracy_A (invalid = incorrect)."""
    true = pd.Series(list(y_true))
    a = pd.Series(list(pred_a))
    b = pd.Series(list(pred_b))
    mask = true.notna()
    true = true[mask].to_numpy()
    a = a[mask]
    b = b[mask]

    correct_a = (a.notna().to_numpy() & (a.to_numpy() == true)).astype(float)
    correct_b = (b.notna().to_numpy() & (b.to_numpy() == true)).astype(float)
    diff = correct_b - correct_a
    point, low, high = bootstrap_ci(diff, n_boot=n_boot, seed=seed)
    return {
        "n": int(len(true)),
        "accuracy_a": float(np.mean(correct_a)) if len(correct_a) else float("nan"),
        "accuracy_b": float(np.mean(correct_b)) if len(correct_b) else float("nan"),
        "diff_b_minus_a": point,
        "ci_low": low,
        "ci_high": high,
    }
