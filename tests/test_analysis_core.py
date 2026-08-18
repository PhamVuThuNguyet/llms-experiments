"""Tests for analysis parsing, metrics, and appendix formatting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.appendix import format_appendix_p_value, round_appendix_table
from analysis.consistency import exact_self_consistency, semantic_self_consistency
from analysis.metrics import (
    binary_confusion,
    compute_binary_metrics,
    mcnemar_exact,
)
from analysis.parsing import (
    normalize_response_text,
    parse_binary_field,
    parse_triage_risk,
    responses_semantically_equal,
)
from analysis.plotting.model_metadata import (
    compute_model_costs,
    display_name,
    ordered_models,
)
from experiment.folder_discovery import (
    find_image_in_folder,
    get_numeric_experiment_folders,
)


def test_parse_binary_field_accepts_json_and_embedded_object():
    assert parse_binary_field('{"midline-shift-present": 1}', "midline-shift-present") == 1
    assert (
        parse_binary_field(
            'Answer: {"burst-fracture-present": 0}', "burst-fracture-present"
        )
        == 0
    )
    assert parse_binary_field("not json", "midline-shift-present") is None


def test_parse_triage_risk_normalizes_variants():
    assert parse_triage_risk('{"risk": "high_risk"}') == "High Risk"
    assert parse_triage_risk('{"risk": "Low-Risk"}') == "Low Risk"
    assert parse_triage_risk('{"risk": "unclear"}') == "Unclear"
    assert parse_triage_risk('{"risk": "maybe"}') is None


def test_semantic_equality_is_order_insensitive():
    left = '{"a": 1, "b": true}'
    right = '{"b": true, "a": 1}'
    assert responses_semantically_equal(left, right)
    assert normalize_response_text(left) == normalize_response_text(right)


def test_compute_binary_metrics_end_to_end_counts_invalid_as_incorrect():
    y_true = [1, 1, 0, 0]
    y_pred = [1, np.nan, 0, 1]
    metrics = compute_binary_metrics(
        y_true, y_pred, invalid_as_incorrect=True, n_boot=50, seed=1
    )
    assert metrics["n_total"] == 4
    assert metrics["n_valid"] == 3
    assert metrics["accuracy"].point == pytest.approx(0.5)
    assert metrics["fn"] == 1
    assert metrics["fp"] == 1


def test_compute_binary_metrics_among_valid_ignores_invalid():
    y_true = [1, 1, 0]
    y_pred = [1, np.nan, 0]
    metrics = compute_binary_metrics(
        y_true, y_pred, invalid_as_incorrect=False, n_boot=50, seed=1
    )
    assert metrics["n_valid"] == 2
    assert metrics["accuracy"].point == pytest.approx(1.0)
    assert metrics["tp"] == 1
    assert metrics["tn"] == 1


def test_binary_confusion_and_mcnemar():
    assert binary_confusion([0, 1, 1, 0], [0, 1, 0, 1]) == {
        "tn": 1,
        "fp": 1,
        "fn": 1,
        "tp": 1,
    }
    result = mcnemar_exact([1, 1, 0, 0], [1, 0, 0, 1], [1, 1, 0, 0])
    assert result["a_only_correct"] == 0
    assert result["b_only_correct"] == 2
    assert 0.0 <= result["p_value"] <= 1.0


def test_self_consistency_metrics():
    df = pd.DataFrame(
        {
            "model_name": ["m1", "m1", "m1", "m1"],
            "experiment_id": [1, 1, 2, 2],
            "response_text": [
                '{"a":1}',
                '{"a": 1}',
                '{"a":1}',
                '{"a":0}',
            ],
        }
    )
    exact = exact_self_consistency(df)
    semantic = semantic_self_consistency(df)
    assert exact.loc[0, "exact_consistency"] == pytest.approx(0.0)
    assert semantic.loc[0, "semantic_consistency"] == pytest.approx(0.5)


def test_appendix_formatting():
    assert format_appendix_p_value(0.001) == "<0.01"
    assert format_appendix_p_value(0.04) == "0.04"
    table = round_appendix_table(pd.DataFrame({"n_total": [1.2], "accuracy": [0.987]}))
    assert int(table.loc[0, "n_total"]) == 1
    assert table.loc[0, "accuracy"] == pytest.approx(0.99)


def test_model_display_and_cost_helpers():
    assert display_name("claude-3-7-sonnet-latest") == "claude-3-7-sonnet-20250219"
    assert ordered_models(["grok-4-fast-reasoning", "gpt-4o"]) == [
        "gpt-4o",
        "grok-4-fast-reasoning",
    ]
    df = pd.DataFrame(
        {
            "model_name": ["gpt-4o", "gpt-4o"],
            "input_tokens": [1_000_000, 1_000_000],
            "output_tokens": [1_000_000, 1_000_000],
        }
    )
    costs = compute_model_costs(df)
    assert len(costs) == 1
    assert costs.loc[0, "avg_total_cost"] == pytest.approx(12.5)


def test_batch_image_discovery(tmp_path):
    folder = tmp_path / "3"
    folder.mkdir()
    (folder / "slice.png").write_bytes(b"fake")
    (tmp_path / "notes.txt").write_text("x")
    assert find_image_in_folder(folder).endswith("slice.png")
    folders = get_numeric_experiment_folders(tmp_path, start_from=0)
    assert folders == [folder]
