#!/usr/bin/env python3
"""
Analyze existing MLLM result CSVs (no new API calls).

Produces:
  - dual accuracy (end-to-end vs among-valid) with bootstrap CIs
  - paired McNemar + bootstrap accuracy differences
  - exact vs semantic self-consistency
  - qualitative failure exports
  - operational metrics (TTFT, latency, token usage)
  - appendix tables (FPR/FNR, McNemar matrix, full paired comparisons)
  - plots under reports/analysis/plots/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from analysis.appendix import (
    format_appendix_p_values,
    format_mcnemar_pvalue_matrix,
    round_appendix_table,
)
from analysis.consistency import exact_self_consistency, semantic_self_consistency
from analysis.constants import DEFAULT_BOOTSTRAP_SAMPLES
from analysis.failures import (
    build_mcnemar_matrix,
    export_binary_rates_table,
    export_failures,
    export_triage_class_rates,
)
from analysis.loaders import TASK_CONFIGS, load_task_results
from analysis.plotting import generate_all_plots
from analysis.summaries import (
    paired_model_comparisons,
    summarize_model_metrics,
    summarize_operational_metrics,
)


def _safe_to_csv(
    df: pd.DataFrame,
    path: Path,
    *,
    appendix: bool = False,
    format_pvalues: bool = False,
) -> Path:
    """Write CSV; if the target is locked, fall back to *_new.csv."""
    export_df = df
    if appendix:
        export_df = round_appendix_table(export_df)
    if format_pvalues:
        export_df = format_appendix_p_values(export_df)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        export_df.to_csv(path, index=False)
        return path
    except PermissionError:
        alt = path.with_name(f"{path.stem}_new{path.suffix}")
        export_df.to_csv(alt, index=False)
        print(f"WARNING: {path} is locked; wrote {alt} instead")
        return alt


def analyze_task(
    task: str,
    reports_dir: Path,
    out_dir: Path,
    plots_dir: Path,
    n_boot: int,
    skip_plots: bool,
) -> None:
    print(f"\n=== Analyzing task: {task} ===")
    df = load_task_results(task, reports_dir=reports_dir)
    print(f"Loaded {len(df)} rows, {df['model_name'].nunique()} models")

    metrics = summarize_model_metrics(task, df, n_boot=n_boot)
    metrics_path = _safe_to_csv(
        metrics, out_dir / f"{task}_dual_accuracy_bootstrap.csv", appendix=True
    )
    print(f"Wrote {metrics_path}")

    paired = paired_model_comparisons(task, df, n_boot=n_boot)
    paired_path = _safe_to_csv(
        paired, out_dir / f"{task}_paired_model_comparisons.csv", appendix=True
    )
    print(f"Wrote {paired_path}")

    exact = exact_self_consistency(df).rename(columns={"n_agree": "n_exact_agree"})
    semantic = semantic_self_consistency(df).rename(
        columns={"n_agree": "n_semantic_agree"}
    )
    consistency = exact.merge(
        semantic,
        on=["model_name", "n_multi_run_cases"],
        how="outer",
    )
    consistency_path = _safe_to_csv(
        consistency, out_dir / f"{task}_self_consistency_exact_vs_semantic.csv"
    )
    print(f"Wrote {consistency_path}")

    failures_path = out_dir / f"{task}_failure_cases.csv"
    failures = export_failures(task, df, failures_path)
    print(f"Wrote {failures_path} ({len(failures)} failure rows)")

    operational = summarize_operational_metrics(task, df)
    operational_path = _safe_to_csv(
        operational, out_dir / f"{task}_operational_metrics.csv", appendix=True
    )
    print(f"Wrote {operational_path}")

    appendix_dir = out_dir / "appendix"
    appendix_dir.mkdir(parents=True, exist_ok=True)
    _safe_to_csv(
        paired,
        appendix_dir / f"{task}_paired_comparisons_full.csv",
        appendix=True,
        format_pvalues=True,
    )

    mcnemar_p, mcnemar_diff = build_mcnemar_matrix(paired)
    mcnemar_p_export = format_mcnemar_pvalue_matrix(mcnemar_p)
    mcnemar_diff_export = round_appendix_table(mcnemar_diff)
    try:
        mcnemar_p_export.to_csv(appendix_dir / f"{task}_mcnemar_pvalue_matrix.csv")
        mcnemar_diff_export.to_csv(
            appendix_dir / f"{task}_paired_accuracy_diff_matrix.csv"
        )
    except PermissionError:
        mcnemar_p_export.to_csv(appendix_dir / f"{task}_mcnemar_pvalue_matrix_new.csv")
        mcnemar_diff_export.to_csv(
            appendix_dir / f"{task}_paired_accuracy_diff_matrix_new.csv"
        )
        print("WARNING: McNemar matrices locked; wrote *_new.csv")
    print(f"Wrote McNemar matrices to {appendix_dir}")

    if TASK_CONFIGS[task]["binary"]:
        rates = export_binary_rates_table(task, metrics)
        rates_path = _safe_to_csv(
            rates, appendix_dir / f"{task}_binary_rates_fpr_fnr.csv", appendix=True
        )
        print(f"Wrote {rates_path}")
        class_rates = None
    else:
        class_rates = export_triage_class_rates(task, df, n_boot=n_boot)
        rates_path = _safe_to_csv(
            class_rates, appendix_dir / f"{task}_class_rates_fpr_fnr.csv", appendix=True
        )
        print(f"Wrote {rates_path}")
    if not skip_plots:
        task_plots_dir = plots_dir / task
        paths = generate_all_plots(
            task=task,
            df=df,
            metrics=metrics,
            paired=paired,
            consistency=consistency,
            failures=failures,
            out_dir=task_plots_dir,
            class_rates=class_rates,
        )
        print(f"Wrote {len(paths)} plots to {task_plots_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(TASK_CONFIGS),
        choices=list(TASK_CONFIGS),
        help="Tasks to analyze",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports/csv_reports"),
        help="Directory with per-model *_results.csv files",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports/analysis"),
        help="Output directory for analysis CSVs",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=Path("reports/analysis/plots"),
        help="Output directory for analysis plots",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
        help="Bootstrap resamples",
    )
    parser.add_argument(
        "--skip-plots", action="store_true", help="Skip plot generation"
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Regenerate plots from existing analysis CSVs (no metric recomputation)",
    )
    return parser


def run_plots_only(
    task: str, reports_dir: Path, out_dir: Path, plots_dir: Path
) -> None:
    print(f"\n=== Regenerating plots: {task} ===")
    df = load_task_results(task, reports_dir=reports_dir)
    metrics = pd.read_csv(out_dir / f"{task}_dual_accuracy_bootstrap.csv")
    paired = pd.read_csv(out_dir / f"{task}_paired_model_comparisons.csv")
    consistency = pd.read_csv(
        out_dir / f"{task}_self_consistency_exact_vs_semantic.csv"
    )
    failures = pd.read_csv(out_dir / f"{task}_failure_cases.csv")
    class_rates = None
    if not TASK_CONFIGS[task]["binary"]:
        preferred = out_dir / "appendix" / f"{task}_class_rates_fpr_fnr.csv"
        fallback = out_dir / "appendix" / f"{task}_class_rates_fpr_fnr_new.csv"
        class_rates = pd.read_csv(preferred if preferred.exists() else fallback)

    task_plots_dir = plots_dir / task
    paths = generate_all_plots(
        task=task,
        df=df,
        metrics=metrics,
        paired=paired,
        consistency=consistency,
        failures=failures,
        out_dir=task_plots_dir,
        class_rates=class_rates,
    )
    print(f"Wrote {len(paths)} plots to {task_plots_dir}")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for task in args.tasks:
        if args.plots_only:
            run_plots_only(task, args.reports_dir, args.out_dir, args.plots_dir)
            continue
        analyze_task(
            task,
            args.reports_dir,
            args.out_dir,
            args.plots_dir,
            args.n_boot,
            args.skip_plots,
        )
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
