#!/usr/bin/env python3
"""
Convert burst-fracture CSV result files into per-record JSON files.

Output path format:
    output/bf-{version}/{experiment_id}/{model_name}_{timestamp}.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_CSV_FILES = [
    "bf_claude-opus-4-1-20250805_results.csv",
    "bf_claude-opus-4-20250514_results.csv",
    "bf_claude-sonnet-4-20250514_results.csv",
    "bf_gpt-4.1_results.csv",
    "bf_gpt-4o_results.csv",
    "bf_gpt-5_results.csv",
    "bf_gpt-5-chat-latest_results.csv",
    "bf_grok-4-fast-reasoning_results.csv",
    "bf_o1_results.csv",
]

FLOAT_FIELDS = {
    "ttft_ms",
    "total_latency_ms",
    "temperature",
    "top_p",
    "latency_seconds",
    "ttft_seconds",
}
INT_FIELDS = {"input_chars", "input_tokens", "output_chars", "output_tokens", "http_status"}


def configure_csv_field_size_limit() -> None:
    """Increase csv parser field limit for long model responses."""
    max_size = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_size)
            return
        except OverflowError:
            max_size //= 10


def parse_value(field: str, value: str) -> Any:
    if value == "":
        return None
    if field in FLOAT_FIELDS:
        try:
            return float(value)
        except ValueError:
            return value
    if field in INT_FIELDS:
        try:
            return int(float(value))
        except ValueError:
            return value
    if field == "has_error":
        return value.lower() == "true"
    return value


def model_safe_name(model_name: str | None) -> str:
    if not model_name:
        return "unknown-model"
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in model_name)


def convert_record(record: dict[str, str], now_stamp: str, output_root: Path) -> Path:
    cleaned: dict[str, Any] = {}
    for key, value in record.items():
        cleaned[key] = parse_value(key, value)

    version = str(cleaned.get("version") or "unknown")
    experiment_id = str(cleaned.get("experiment_id") or "").strip()
    if not experiment_id:
        raise ValueError("Missing experiment_id in row")

    model_name = model_safe_name(str(cleaned.get("model_name") or "unknown-model"))
    output_dir = output_root / f"bf-{version}" / Path(experiment_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{model_name}_{now_stamp}.json"
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return output_file


def convert_csv_file(csv_path: Path, output_root: Path) -> tuple[int, int]:
    now_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    written = 0
    skipped = 0

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                skipped += 1
                continue
            try:
                convert_record(row, now_stamp, output_root)
                written += 1
            except Exception:
                skipped += 1

    return written, skipped


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert burst-fracture CSV results to JSON files.")
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=Path("reports") / "csv_reports",
        help="Directory containing input CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Root output directory (default: llms-performance/output).",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=DEFAULT_CSV_FILES,
        help="CSV filenames to convert.",
    )
    return parser


def main() -> None:
    configure_csv_field_size_limit()

    args = build_arg_parser().parse_args()
    csv_dir: Path = args.csv_dir
    output_dir: Path = args.output_dir
    files: list[str] = args.files

    total_written = 0
    total_skipped = 0

    for filename in files:
        csv_path = csv_dir / filename
        if not csv_path.exists():
            print(f"[WARN] Missing CSV file: {csv_path}")
            continue

        written, skipped = convert_csv_file(csv_path, output_dir)
        total_written += written
        total_skipped += skipped
        print(f"[OK] {filename}: wrote={written}, skipped={skipped}")

    print(f"[DONE] total wrote={total_written}, total skipped={total_skipped}")


if __name__ == "__main__":
    main()
