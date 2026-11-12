"""Calendário sintético: datas de produção impedem usar reprocessamento futuro."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from earthfire.io import sha256_file, write_json, read_json


def test_calendar_excludes_unavailable_historical_publications(tmp_path):
    module_path = Path(__file__).resolve().parents[1] / "scripts/relatar_expansao_v6.py"
    spec = importlib.util.spec_from_file_location("report_expansion_v6", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / "features/synthetic.parquet"
    path.parent.mkdir()
    rows = []
    for year, event in [(2020, 1), (2021, 0), (2022, 1), (2023, 0), (2024, 1)]:
        start = pd.Timestamp(year, 1, 1)
        rows.append(
            {
                "region_id": "synthetic",
                "parent_id": "synthetic",
                "period_start": start,
                "period_end_exclusive": start + pd.Timedelta(days=8),
                "produced_at": pd.Timestamp(2026, 1, 10)
                if year == 2022
                else start + pd.Timedelta(days=10),
                "event": event,
                "fire_pixels": event,
                "valid_pixels": 1681,
                "cloud_pixels": 0,
                "total_pixels": 1681,
                "observation_state": "positive_detection"
                if event
                else "observed_no_detection",
                "weather_complete": False,
            }
        )
    pd.DataFrame(rows).to_parquet(path, index=False)
    write_json(
        tmp_path / "model_registry.json",
        {"features": {"features/synthetic.parquet": sha256_file(path)}},
    )
    module.calendar(tmp_path)
    result = pd.read_parquet(tmp_path / "evaluation/calendar_parts/synthetic.parquet")
    assert result.reference_n.tolist() == [0, 1, 2, 2, 3]
    assert result.observed_anomaly.iloc[:4].isna().all()
    assert result.reference_probability.iloc[-1] == pytest.approx(0.375)
    assert result.observed_anomaly.iloc[-1] == pytest.approx(0.625)
    manifest = read_json(tmp_path / "evaluation/calendar_manifest.json")
    assert (
        manifest["parts"]["evaluation/calendar_parts/synthetic.parquet"][
            "anomalies_supported"
        ]
        == 1
    )
