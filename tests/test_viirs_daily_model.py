"""Os exemplos gerados aqui são sintéticos, exclusivamente para verificação de software."""
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from earthfire import viirs_daily_data
from earthfire.io import sha256_file, write_json
from earthfire.mask_features import FEATURES_BASE, FEATURES_WEATHER
from earthfire.mask_model import posterior_probabilities
from earthfire.viirs_daily_model import evaluate, evaluate_baselines, fit_logistic, logistic_predict, train_model


def synthetic_examples():
    rng = np.random.default_rng(71)
    n = 160
    frame = pd.DataFrame({key: rng.normal(size=n) for key in FEATURES_BASE + FEATURES_WEATHER})
    frame["past_modis_missing"] = 0
    frame["past_viirs_missing"] = 0
    frame.loc[:7, "past_viirs_log_records"] = np.nan
    frame["event"] = np.arange(n) % 2
    frame["split"] = "train"
    frame["spatial_group"] = "train"
    frame["region_id"] = "synthetic"
    return frame


def test_logistic_baseline_fits_training_only():
    frame = synthetic_examples()
    model = fit_logistic(frame.iloc[:120], list(FEATURES_BASE))
    p = logistic_predict(model, frame.iloc[120:])
    assert len(p) == 40 and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    assert model["fill"] == fit_logistic(frame.iloc[:120], list(FEATURES_BASE))["fill"]


def test_viirs_model_roundtrip_preserves_target(project):
    train = synthetic_examples()
    validation = train.iloc[:60].copy()
    validation["split"] = "validation"
    frame = pd.concat([train, validation], ignore_index=True)
    output = project.path / "processed"
    output.mkdir(parents=True)
    examples = output / "mask_examples.parquet"
    frame.to_parquet(examples, index=False)
    write_json(output / "viirs_daily_summary.json", {"examples_sha256": sha256_file(examples),
               "input_sha256": {"code": sha256_file(Path(viirs_daily_data.__file__))}})
    path = train_model(project, frame, steps=3)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    assert saved["model_version"] == "vj114a1_daily_weather_v4b"
    assert "VJ114A1" in saved["target"]
    samples = posterior_probabilities(path, frame.iloc[:3], n_samples=8, seed=42)
    assert samples.shape == (8, 3) and np.isfinite(samples).all()
    assert train_model(project, frame, steps=3) == path


def test_synthetic_end_to_end_exports_daily_probabilities(project):
    base = synthetic_examples()
    frames = []
    for split, year, n in (("train", 2021, 160), ("validation", 2023, 60), ("test", 2024, 60)):
        piece = base.iloc[:n].copy()
        piece["split"] = split
        piece["period_start"] = pd.date_range(f"{year}-01-01", periods=n, freq="D")
        piece["spatial_group"] = "train"
        if split != "train":
            piece.loc[piece.index[n // 2:], "spatial_group"] = "holdout"
        piece["region_id"] = np.where(piece.spatial_group == "train", "a", "b")
        piece["past_viirs_mask_event"] = np.arange(n) % 2
        frames.append(piece)
    all_data = pd.concat(frames, ignore_index=True)
    output = project.path / "processed"
    output.mkdir(parents=True)
    path = output / "mask_examples.parquet"
    all_data.to_parquet(path, index=False)
    write_json(output / "viirs_daily_summary.json", {"examples_sha256": sha256_file(path),
               "input_sha256": {"code": sha256_file(Path(viirs_daily_data.__file__))}})
    baseline = evaluate_baselines(project, all_data)
    assert baseline["validation_n"] == 60
    model = train_model(project, all_data, steps=3)
    report = evaluate(project, all_data, model, samples=8)
    test = pd.read_parquet(project.path / "evaluation/viirs_predictions_test.parquet")
    assert "VJ114A1" in report["event"] and report["splits"]["test"]["overall"]["model_probability"]["n"] == 60
    assert test.model_probability.between(0, 1).all()
    assert (project.path / "evaluation/viirs_validation_before_test.json").exists()
