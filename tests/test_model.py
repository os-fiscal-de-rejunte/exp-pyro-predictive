from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from earthfire.features import WEATHER, build_examples
from earthfire.harmonize import harmonize, matched_training_weeks
from earthfire.model import posterior_counts, train


def test_harmonization_refuses_no_overlap():
    frame = pd.DataFrame({"region_id": ["a", "a"], "week_start": pd.to_datetime(["2010-01-04", "2019-01-07"]),
        "product": ["modis", "viirs-jpss1"], "satellite": ["Terra", "N20"], "detections": [2, 3]})
    result = harmonize(frame, "2021-12-31")
    assert result["status"] == "insufficient_data" and result["posterior"] is None


def test_harmonization_detects_actual_overlap_before_cutoff():
    frame = pd.DataFrame({"region_id": ["a", "a", "a"], "week_start": pd.to_datetime(["2019-01-07", "2019-01-07", "2022-01-03"]),
        "product": ["modis", "viirs-jpss1", "modis"], "satellite": ["Terra", "N20", "Terra"], "detections": [2, 3, 5]})
    pairs = matched_training_weeks(frame, "2021-12-31")
    assert len(pairs) == 1 and pairs.iloc[0].detections_viirs_n20 == 3


def test_harmonization_probabilistic_path_on_synthetic_pairs():
    weeks = pd.date_range("2019-01-07", periods=32, freq="W-MON")
    rows = []
    for week in weeks:
        rows.extend([{"region_id": "a", "week_start": week, "product": "modis", "satellite": "Terra", "detections": 2},
                     {"region_id": "a", "week_start": week, "product": "viirs-jpss1", "satellite": "N20", "detections": 4}])
    result = harmonize(pd.DataFrame(rows), "2021-12-31", steps=5)
    assert result["status"] == "exploratory" and result["matched_weeks"] == 32
    assert result["posterior"] is not None and result["baseline"]


def test_pyro_checkpoint_and_conditional_predictions(project):
    weeks = pd.date_range("2018-01-01", "2020-12-28", freq="W-MON")
    weekly = pd.DataFrame({"region_id": "a", "product": "viirs-jpss1", "satellite": "N20", "week_start": weeks,
                           "detections": 2 + (np.arange(len(weeks)) % 8)})
    weather = pd.DataFrame({"region_id": "a", "week_start": weeks})
    for i, name in enumerate(WEATHER): weather[name] = 10. + i + np.sin(np.arange(len(weeks)) / 8.)
    examples, _ = build_examples(weekly, weather, project.config)
    (project.path / "processed").mkdir(exist_ok=True)
    examples.to_parquet(project.path / "processed/examples.parquet", index=False)
    checkpoint = train(project, examples, include_weather=True, steps=5)
    output = examples.loc[examples.split == "test"].head(4)
    draws, logp = posterior_counts(checkpoint, output, n_samples=20, seed=3)
    assert draws.shape == (20, 4) and (draws > 0).all()
    assert np.isfinite(logp).all()
    assert train(project, examples, include_weather=True, steps=5) == checkpoint
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved["complete"] = False
    torch.save(saved, checkpoint)
    resumed = train(project, examples, include_weather=True, steps=8)
    assert torch.load(resumed, map_location="cpu", weights_only=False)["step"] == 8
    assert project.manifest["training_runs"] == 1
