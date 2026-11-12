"""Contratos offline de um experimento treinado com dados reais; não retreina."""

from pathlib import Path

import pandas as pd
import pytest

from earthfire.expansion_evaluate import summarize
from earthfire.io import read_json, sha256_file

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "artifacts/geography_annual_v1"


def test_real_geography_package_and_metrics_are_reproducible():
    manifest = read_json(MODEL / "manifest.json")
    for name, digest in manifest["files"].items():
        assert sha256_file(MODEL / name) == digest
    examples = pd.read_parquet(MODEL / "examples.parquet")
    predictions = pd.read_parquet(MODEL / "predictions.parquet")
    metrics = read_json(MODEL / "metrics.json")
    assert len(examples) == 2597 and examples.region_id.nunique() == 11
    assert len(predictions) == 844 and predictions.event.sum() == 209
    for name in ("with_geography", "without_geography"):
        measured = summarize(predictions.event, predictions[name])
        assert measured["brier"] == pytest.approx(metrics["test"][name]["brier"])
        assert measured["accuracy"] == metrics["test"][name]["accuracy"]
    # A acurácia maior não pode esconder piora probabilística ou transferência fraca.
    assert (
        metrics["test"]["with_geography"]["brier"]
        > metrics["test"]["without_geography"]["brier"]
    )
    assert metrics["spatial_test"]["with_geography"]["recall"] == 0


def test_real_geography_extrapolation_preserves_primary_forecast(monkeypatch):
    from earthfire import geographic_predict as module

    context = read_json(ROOT / "artifacts/geography_demo/geography.json")
    monkeypatch.setattr(module, "describe_geography", lambda *a, **kw: context)
    result = module.forecast_with_geography(
        -21.229167,
        -54.341825,
        ROOT / "artifacts/maxcoverage_v6",
        "2026-11-12T12:00:00Z",
        geography_model_dir=MODEL,
        fetch=False,
    )
    assert result["probability"] == pytest.approx(0.24017471119408398)
    assert result["primary_prediction_uses_geography"] is False
    candidate = result["geography_candidate"]
    assert candidate["deployment_status"] == "experimental_only"
    assert candidate["status"] == "out_of_distribution"
    assert candidate["probability"] is None
    assert candidate["probability_interval_95"] is None
    assert candidate["predictive_event_interval_95"] is None
    assert candidate["global_accuracy_established"] is False
    assert candidate["evaluation_lead_days"] == 0
    assert candidate["additional_lead_not_validated"] is True
