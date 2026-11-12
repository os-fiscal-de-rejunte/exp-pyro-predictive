"""Contratos sintéticos; não constituem evidência de desempenho geográfico."""

import json

import numpy as np
import pandas as pd
import pytest

from earthfire import spatial_predict as spatial
from earthfire.expansion_predict import base_result
from earthfire.io import sha256_file, write_json


@pytest.fixture
def adapter(tmp_path, monkeypatch):
    (tmp_path / "model_registry.json").write_text("{}")
    (tmp_path / "models").mkdir()
    (tmp_path / "models/mask_no_weather.pt").write_bytes(b"synthetic checkpoint")
    path = tmp_path / "processed/mask_examples.parquet"
    path.parent.mkdir()
    pd.DataFrame(
        {"region_id": ["synthetic"], "latitude": [0.0], "longitude": [0.0]}
    ).to_parquet(path)
    write_json(tmp_path / "training_sample.json", {"sample_sha256": sha256_file(path)})
    monkeypatch.setattr(
        spatial, "validated_forecast", lambda lat, lon, *args: base_result(lat, lon)
    )
    monkeypatch.setattr(
        spatial,
        "load",
        lambda root: (
            {"created_utc": "2026-01-01T00:00:00Z"},
            {},
            {"region_month": {}},
        ),
    )
    frames = []

    def posterior(checkpoint, frame, **kwargs):
        frames.append(frame.copy())
        return np.array([[0.2], [0.3], [0.4]])

    monkeypatch.setattr("earthfire.mask_model.posterior_probabilities", posterior)
    return tmp_path, frames


def test_unseen_location_is_explicitly_exploratory(adapter):
    root, frames = adapter
    value = spatial.forecast(40.7, -74.0, root, "2026-11-12T12:00:00Z")
    assert value["status"] == "exploratory" and value["is_predictive"]
    assert (
        not value["within_validated_coverage"]
        and not value["location_seen_in_training"]
    )
    assert value["probability"] == pytest.approx(0.3)
    assert value["probability_locally_calibrated"] is False
    assert np.isnan(frames[0].past_modis_event.iloc[0])
    assert frames[0].past_modis_missing.iloc[0] == 1
    assert value["valid_from"] == "2026-11-17T00:00:00Z"
    json.dumps(value, allow_nan=False)


def test_strict_mode_preserves_unavailable_result(adapter):
    root, frames = adapter
    result = spatial.forecast(40.7, -74.0, root, "2026-11-12", allow_exploratory=False)
    assert result["probability"] is None and not frames


def test_validated_path_preserves_prediction(adapter, monkeypatch):
    root, frames = adapter

    def validated(lat, lon, *args):
        return {**base_result(lat, lon), "status": "ok", "probability": 0.12}

    monkeypatch.setattr(spatial, "validated_forecast", validated)
    value = spatial.forecast(40.7, -74.0, root, "2026-11-12")
    assert value["probability"] == 0.12 and not value["exploratory"] and not frames


def test_changed_training_sample_is_rejected(adapter):
    root, _ = adapter
    (root / "processed/mask_examples.parquet").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Amostra"):
        spatial.forecast(40.7, -74.0, root, "2026-11-12")


def test_outdated_emission_does_not_fabricate_prediction(adapter):
    root, frames = adapter
    value = spatial.forecast(40.7, -74.0, root, "2030-01-01")
    assert value["probability"] is None and not frames
