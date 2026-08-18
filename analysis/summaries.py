"""Per-model metric summarization and paired comparisons."""

from __future__ import annotations

import pandas as pd

from analysis.constants import DEFAULT_BOOTSTRAP_SAMPLES
from analysis.loaders import TASK_CONFIGS
from analysis.metrics import (
    MetricEstimate,
    _safe_rate,
    bootstrap_ci,
    compute_binary_metrics,
    mcnemar_exact,
    paired_bootstrap_accuracy_diff,
)


def compute_multiclass_accuracy(
    y_true: pd.Series,
    y_pred: pd.Series,
    invalid_as_incorrect: bool = True,
    n_boot: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = 0,
) -> dict:
    mask = y_true.notna()
    true = y_true[mask]
    pred = y_pred[mask]
    n_total = int(len(true))
    valid_mask = pred.notna()
    n_valid = int(valid_mask.sum())
    completion = _safe_rate(n_valid, n_total)

    if invalid_as_incorrect:
        correct = ((pred == true) & valid_mask).astype(float).to_numpy()
        point, low, high = bootstrap_ci(correct, n_boot=n_boot, seed=seed)
        n_eval = n_total
    else:
        if n_valid == 0:
            return {
                "n_total": n_total,
                "n_valid": 0,
                "completion_rate": MetricEstimate("completion_rate", completion, n=n_total),
                "accuracy": MetricEstimate("accuracy", float("nan"), n=0),
            }
        subset_true = true[valid_mask]
        subset_pred = pred[valid_mask]
        correct = (subset_pred == subset_true).astype(float).to_numpy()
        point, low, high = bootstrap_ci(correct, n_boot=n_boot, seed=seed)
        n_eval = n_valid

    completion_point, completion_low, completion_high = bootstrap_ci(
        valid_mask.astype(float).to_numpy(),
        n_boot=n_boot,
        seed=seed + 1,
    )
    return {
        "n_total": n_total,
        "n_valid": n_valid,
        "completion_rate": MetricEstimate(
            "completion_rate", completion_point, completion_low, completion_high, n_total
        ),
        "accuracy": MetricEstimate("accuracy", point, low, high, n_eval),
    }


def _binary_metric_columns(prefix: str, metrics: dict) -> dict:
    """Flatten end-to-end or among-valid binary metric dict into CSV columns."""
    return {
        f"sensitivity_{prefix}": metrics["sensitivity"].point,
        f"sensitivity_{prefix}_ci_low": metrics["sensitivity"].ci_low,
        f"sensitivity_{prefix}_ci_high": metrics["sensitivity"].ci_high,
        f"specificity_{prefix}": metrics["specificity"].point,
        f"specificity_{prefix}_ci_low": metrics["specificity"].ci_low,
        f"specificity_{prefix}_ci_high": metrics["specificity"].ci_high,
        f"fpr_{prefix}": metrics["fpr"].point,
        f"fpr_{prefix}_ci_low": metrics["fpr"].ci_low,
        f"fpr_{prefix}_ci_high": metrics["fpr"].ci_high,
        f"fnr_{prefix}": metrics["fnr"].point,
        f"fnr_{prefix}_ci_low": metrics["fnr"].ci_low,
        f"fnr_{prefix}_ci_high": metrics["fnr"].ci_high,
        f"precision_{prefix}": metrics["precision"].point,
        f"precision_{prefix}_ci_low": metrics["precision"].ci_low,
        f"precision_{prefix}_ci_high": metrics["precision"].ci_high,
        f"f1_{prefix}": metrics["f1"].point,
        f"f1_{prefix}_ci_low": metrics["f1"].ci_low,
        f"f1_{prefix}_ci_high": metrics["f1"].ci_high,
    }


def _accuracy_summary_row(
    task: str,
    model_name: str,
    end_to_end: dict,
    among_valid: dict,
    extra: dict | None = None,
) -> dict:
    row = {
        "task": task,
        "model_name": model_name,
        "n_total": end_to_end["n_total"],
        "n_valid": end_to_end["n_valid"],
        "completion_rate": end_to_end["completion_rate"].point,
        "completion_ci_low": end_to_end["completion_rate"].ci_low,
        "completion_ci_high": end_to_end["completion_rate"].ci_high,
        "accuracy_end_to_end": end_to_end["accuracy"].point,
        "accuracy_end_to_end_ci_low": end_to_end["accuracy"].ci_low,
        "accuracy_end_to_end_ci_high": end_to_end["accuracy"].ci_high,
        "accuracy_among_valid": among_valid["accuracy"].point,
        "accuracy_among_valid_ci_low": among_valid["accuracy"].ci_low,
        "accuracy_among_valid_ci_high": among_valid["accuracy"].ci_high,
    }
    if extra:
        row.update(extra)
    return row


def summarize_model_metrics(
    task: str,
    df: pd.DataFrame,
    n_boot: int = DEFAULT_BOOTSTRAP_SAMPLES,
) -> pd.DataFrame:
    config = TASK_CONFIGS[task]
    pred_col = config["prediction_col"]
    rows = []
    for model_name, model_df in df.groupby("model_name"):
        labeled = model_df[model_df["ground_truth"].notna()].copy()
        if labeled.empty:
            continue
        if config["binary"]:
            end_to_end = compute_binary_metrics(
                labeled["ground_truth"],
                labeled[pred_col],
                invalid_as_incorrect=True,
                n_boot=n_boot,
            )
            among_valid = compute_binary_metrics(
                labeled["ground_truth"],
                labeled[pred_col],
                invalid_as_incorrect=False,
                n_boot=n_boot,
            )
            rows.append(
                _accuracy_summary_row(
                    task,
                    model_name,
                    end_to_end,
                    among_valid,
                    {
                        **_binary_metric_columns("end_to_end", end_to_end),
                        **_binary_metric_columns("among_valid", among_valid),
                        "auc_among_valid": among_valid["auc"],
                        "tp_valid": among_valid["tp"],
                        "tn_valid": among_valid["tn"],
                        "fp_valid": among_valid["fp"],
                        "fn_valid": among_valid["fn"],
                    },
                )
            )
        else:
            end_to_end = compute_multiclass_accuracy(
                labeled["ground_truth"],
                labeled[pred_col],
                invalid_as_incorrect=True,
                n_boot=n_boot,
            )
            among_valid = compute_multiclass_accuracy(
                labeled["ground_truth"],
                labeled[pred_col],
                invalid_as_incorrect=False,
                n_boot=n_boot,
            )
            rows.append(_accuracy_summary_row(task, model_name, end_to_end, among_valid))
    return pd.DataFrame(rows).sort_values("accuracy_end_to_end", ascending=False)


def _primary_predictions(df: pd.DataFrame) -> pd.DataFrame:
    if "version" in df.columns:
        return (
            df.sort_values("version")
            .groupby(["model_name", "experiment_id"], as_index=False)
            .first()
        )
    return df.groupby(["model_name", "experiment_id"], as_index=False).first()


def paired_model_comparisons(
    task: str,
    df: pd.DataFrame,
    n_boot: int = DEFAULT_BOOTSTRAP_SAMPLES,
) -> pd.DataFrame:
    config = TASK_CONFIGS[task]
    pred_col = config["prediction_col"]
    models = sorted(df["model_name"].dropna().unique())
    primary = _primary_predictions(df)

    rows = []
    for i, model_a in enumerate(models):
        for model_b in models[i + 1 :]:
            left = primary[primary["model_name"] == model_a][
                ["experiment_id", "ground_truth", pred_col]
            ].rename(columns={pred_col: "pred_a"})
            right = primary[primary["model_name"] == model_b][
                ["experiment_id", pred_col]
            ].rename(columns={pred_col: "pred_b"})
            merged = left.merge(right, on="experiment_id", how="inner")
            merged = merged[merged["ground_truth"].notna()]
            if merged.empty:
                continue
            mcnemar = mcnemar_exact(merged["ground_truth"], merged["pred_a"], merged["pred_b"])
            boot = paired_bootstrap_accuracy_diff(
                merged["ground_truth"],
                merged["pred_a"],
                merged["pred_b"],
                n_boot=n_boot,
            )
            rows.append(
                {
                    "task": task,
                    "model_a": model_a,
                    "model_b": model_b,
                    "n": boot["n"],
                    "accuracy_a": boot["accuracy_a"],
                    "accuracy_b": boot["accuracy_b"],
                    "diff_b_minus_a": boot["diff_b_minus_a"],
                    "diff_ci_low": boot["ci_low"],
                    "diff_ci_high": boot["ci_high"],
                    "mcnemar_p_value": mcnemar["p_value"],
                    "a_only_correct": mcnemar["a_only_correct"],
                    "b_only_correct": mcnemar["b_only_correct"],
                }
            )
    return pd.DataFrame(rows)


def summarize_operational_metrics(task: str, df: pd.DataFrame) -> pd.DataFrame:
    """Mean TTFT, total latency, and token usage per model."""
    rows = []
    for model_name, model_df in df.groupby("model_name"):
        labeled = model_df[model_df["ground_truth"].notna()].copy()
        if labeled.empty:
            continue
        row = {"task": task, "model_name": model_name, "n_calls": int(len(labeled))}
        if "ttft_ms" in labeled.columns:
            ttft = pd.to_numeric(labeled["ttft_ms"], errors="coerce")
            row["mean_ttft_ms"] = float(ttft.mean())
            row["median_ttft_ms"] = float(ttft.median())
        if "total_latency_ms" in labeled.columns:
            latency = pd.to_numeric(labeled["total_latency_ms"], errors="coerce")
            row["mean_total_latency_ms"] = float(latency.mean())
            row["median_total_latency_ms"] = float(latency.median())
        for col, out in [
            ("input_tokens", "mean_input_tokens"),
            ("output_tokens", "mean_output_tokens"),
        ]:
            if col in labeled.columns:
                tokens = pd.to_numeric(labeled[col], errors="coerce")
                row[out] = float(tokens.mean())
        if "mean_input_tokens" in row and "mean_output_tokens" in row:
            row["mean_total_tokens"] = row["mean_input_tokens"] + row["mean_output_tokens"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("model_name")
