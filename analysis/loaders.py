"""Task configuration and result/ground-truth loaders."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from analysis.constants import PROJECT_ROOT, REPO_ROOT
from analysis.parsing import parse_binary_field, parse_triage_risk

TASK_CONFIGS = {
    "mls": {
        "result_glob": "mls_binary_*_results.csv",
        "prediction_col": "midline_shift_prediction",
        "json_field": "midline-shift-present",
        "ground_truth_csv": REPO_ROOT / "eda" / "ground_truth.csv",
        "binary": True,
    },
    "bf": {
        "result_glob": "bf_*_results.csv",
        "prediction_col": "burst_fracture_prediction",
        "json_field": "burst-fracture-present",
        "ground_truth_csv": REPO_ROOT / "eda" / "bf_ground_truth.csv",
        "binary": True,
    },
    "triage": {
        "result_glob": "triage_*_results.csv",
        "prediction_col": "risk_prediction",
        "json_field": "risk",
        "ground_truth_csv": REPO_ROOT / "eda" / "triage_ground_truth.csv",
        "binary": False,
    },
}

DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports" / "csv_reports"


def _load_mls_ground_truth(path: Path) -> dict:
    gt = pd.read_csv(path)
    gt["experiment_id"] = gt["name"].str.extract(r"CQ500-CT-(\d+)").astype(int)
    return dict(zip(gt["experiment_id"], gt["ground_truth"].astype(int)))


def _load_bf_ground_truth(path: Path) -> dict:
    gt = pd.read_csv(path)
    gt["experiment_id"] = (
        gt["StudyInstanceUID"].astype(str) + "/slice_" + gt["slice_number"].astype(str)
    )
    return dict(zip(gt["experiment_id"], gt["Burst_Fracture"].astype(int)))


def _load_triage_ground_truth(path: Path) -> dict:
    gt = pd.read_csv(path)
    gt["experiment_id"] = gt["Study_ID"].astype(int)
    return dict(zip(gt["experiment_id"], gt["Triaging"].astype(str)))


_GROUND_TRUTH_LOADERS = {
    "mls": _load_mls_ground_truth,
    "bf": _load_bf_ground_truth,
    "triage": _load_triage_ground_truth,
}


def load_ground_truth(task: str) -> dict:
    config = TASK_CONFIGS[task]
    loader = _GROUND_TRUTH_LOADERS.get(task)
    if loader is None:
        raise ValueError(f"Unknown task: {task}")
    return loader(Path(config["ground_truth_csv"]))


def _is_results_csv(path: Path) -> bool:
    name = path.name
    return (
        name.endswith("_results.csv")
        and "model_" not in name
        and "metrics" not in name
    )


def load_task_results(task: str, reports_dir: Path | None = None) -> pd.DataFrame:
    """Load per-model result CSVs for a task and attach ground truth."""
    config = TASK_CONFIGS[task]
    reports_dir = reports_dir or DEFAULT_REPORTS_DIR
    paths = [path for path in sorted(reports_dir.glob(config["result_glob"])) if _is_results_csv(path)]
    if not paths:
        raise FileNotFoundError(f"No result CSVs for task={task} in {reports_dir}")

    df = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)

    gt_map = load_ground_truth(task)
    if task in {"mls", "triage"}:
        df["experiment_id_key"] = pd.to_numeric(df["experiment_id"], errors="coerce")
    else:
        df["experiment_id_key"] = df["experiment_id"].astype(str)

    df["ground_truth"] = df["experiment_id_key"].map(gt_map)

    pred_col = config["prediction_col"]
    if pred_col not in df.columns or df[pred_col].isna().all():
        if config["binary"]:
            df[pred_col] = df["response_text"].apply(
                lambda text: parse_binary_field(text, config["json_field"])
            )
        else:
            df[pred_col] = df["response_text"].apply(parse_triage_risk)

    return df
