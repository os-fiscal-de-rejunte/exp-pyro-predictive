from datetime import date

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from earthfire.io import sha256_file
from earthfire.mask_location import _mask_for_location, _validation_support, modis_period_start, next_period


def test_modis_periods_respect_year_boundary():
    assert modis_period_start(date(2025, 12, 31)) == date(2025, 12, 27)
    assert next_period(date(2025, 12, 31)) == date(2026, 1, 1)
    assert next_period(date(2026, 1, 1)) == date(2026, 1, 9)


def test_location_requires_validation_support():
    metrics = {"validation": {"weather": {"spatial_groups": {
        "holdout": {"model": {"n": 40, "positive": 4, "brier": .1}, "seasonal": {"brier": .2}},
        "train": {"model": {"n": 80, "positive": 8, "brier": .1}, "seasonal": {"brier": .2}}}}}}
    assert not _validation_support(metrics, "weather", "holdout")[0]
    assert _validation_support(metrics, "weather", "train")[0]


def test_exact_area_reuses_verified_mask_without_network(tmp_path, monkeypatch):
    coverage = tmp_path / "coverage"
    coverage.mkdir()
    data = coverage / "mask_summary.parquet"
    pd.DataFrame([{"latitude": -11.5, "longitude": -46.875,
                   "period_start": "2025-08-29", "event": None}]).to_parquet(data, index=False)
    (coverage / "summary.json").write_text(json.dumps({"output_sha256": sha256_file(data)}), encoding="utf-8")
    monkeypatch.setattr("earthfire.mask_location.requests.get", lambda *a, **k: pytest.fail("não deve acessar a rede"))
    run = SimpleNamespace(path=tmp_path, config={})
    assert _mask_for_location(run, -11.5, -46.875, date(2025, 8, 29))["event"] is None
    data.write_bytes(b"alterado")
    with pytest.raises(ValueError, match="não confere"):
        _mask_for_location(run, -11.5, -46.875, date(2025, 8, 29))
