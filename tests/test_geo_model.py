"""Treino sintético: as métricas aqui não medem desempenho sobre a Terra."""

import numpy as np
import pandas as pd
import pytest
import torch

from earthfire.geo_model import GEO_FEATURES, experiment, posterior
from earthfire.io import read_json


def test_geographic_training_preserves_temporal_spatial_boundaries(tmp_path):
    rows = []
    for group in ("train", "holdout"):
        for split, year, n in [
            ("train", 2022, 120),
            ("validation", 2023, 40),
            ("test", 2024, 40),
        ]:
            for i in range(n):
                item = {k: np.nan for k in GEO_FEATURES}
                item.update(
                    {
                        "latitude": 10.0 if group == "train" else 80.0,
                        "longitude": 20.0,
                        "season_sin": 0.5,
                        "season_cos": 0.5,
                        "geo_ndvi": 0.2 + 0.6 * (i % 2),
                        "geo_landcover_sample_n": 9,
                        "geo_ndvi_sample_n": 9,
                        "region_id": group,
                        "spatial_group": group,
                        "split": split,
                        "event": i % 2,
                        "period_start": pd.Timestamp(year, 1, 1) + pd.Timedelta(days=i),
                    }
                )
                rows.append(item)
    frame = pd.DataFrame(rows)
    cfg = {
        "seed": 42,
        "steps": 20,
        "max_training_examples": 1000,
        "max_training_seconds": 120,
    }
    result = experiment(frame, tmp_path, cfg)
    saved = torch.load(
        tmp_path / "with_geography.pt", map_location="cpu", weights_only=False
    )
    assert saved["training_regions"] == ["train"] and saved["training_n"] == 120
    assert saved["mean"][0] == 10.0
    assert result["test"]["with_geography"]["n"] == 80
    assert result["spatial_test"]["with_geography"]["n"] == 40
    values = posterior(tmp_path / "with_geography.pt", frame.iloc[:2], 10)
    assert values.shape == (10, 2) and np.isfinite(values).all()
    np.testing.assert_array_equal(
        values, posterior(tmp_path / "with_geography.pt", frame.iloc[:2], 10)
    )
    assert read_json(tmp_path / "selection.json")["prior_test_results_already_known"]
    with pytest.raises(ValueError, match="congelado"):
        experiment(frame, tmp_path, cfg)
