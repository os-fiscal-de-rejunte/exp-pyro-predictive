"""Testes sintéticos de contratos; não são evidência de desempenho real."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from earthfire.global_grid import contains, footprint, locate
from earthfire.global_pipeline import predict
from earthfire.mask_features import weather_history


def test_global_grid_wraps_and_rejects_invalid_coordinates():
    assert locate(0, 180)["id"] == locate(0, -180)["id"]
    assert locate(89.9, 0)["area_km2"] < locate(0, 0)["area_km2"]
    geometry = locate(0, -180)["geometry"]
    assert len(geometry["coordinates"]) == 2
    for invalid in (float("nan"), float("inf"), 91):
        with pytest.raises(ValueError):
            locate(invalid, 0)


def test_footprint_matches_area_and_does_not_use_whole_climate_cell():
    row = {"xllcorner": 0, "yllcorner": 0, "cellsize_m": 1000, "ncols": 4, "nrows": 4}
    assert contains(row, 0.01, 0.01)
    assert not contains(row, 0.2, 0.2)
    assert footprint(row)["coordinates"][0][0] == [0, 0]


def test_weather_features_ignore_future_observations():
    weather = pd.DataFrame(
        {
            "region_id": "a",
            "date": pd.date_range("2022-01-01", periods=30),
            **{
                n: np.ones(30)
                for n in ("T2M", "RH2M", "PRECTOTCORR", "WS10M", "GWETROOT")
            },
        }
    )
    masks = pd.DataFrame(
        {"region_id": ["a"], "period_start": [pd.Timestamp("2022-01-20")]}
    )
    before = weather_history(weather, masks)
    weather.loc[weather.date > "2022-01-17", "T2M"] = 1000
    pd.testing.assert_frame_equal(before, weather_history(weather, masks))
    assert before.iloc[0].weather_last_date == pd.Timestamp("2022-01-17")
    assert before.iloc[0].weather_PRECTOTCORR == 8


def test_unknown_location_returns_null_without_training(tmp_path):
    from earthfire.io import sha256_file, write_json

    folder = tmp_path / "processed"
    folder.mkdir()
    data = folder / "mask_examples.parquet"
    pd.DataFrame({"region_id": ["elsewhere"]}).to_parquet(data, index=False)
    write_json(folder / "manifest.json", {"examples_sha256": sha256_file(data)})
    result = predict(0, 0, "2024-01-01", tmp_path)
    assert result["status"] == "insufficient_data"
    assert result["probability"] is None


def test_synthetic_training_reload_and_inference(project):
    from earthfire.global_pipeline import model
    from earthfire.io import sha256_file, write_json
    from earthfire.mask_features import FEATURES_BASE, FEATURES_WEATHER

    project.config["global_model"] = {"steps": 30}
    project.config["release"] = {
        "min_n": 10,
        "min_class_n": 1,
        "max_mean_calibration_error": 1.0,
        "max_brier_ratio": 100.0,
    }
    rng = np.random.default_rng(42)
    rows = []
    for split, year, n in [
        ("train", 2018, 120),
        ("validation", 2019, 40),
        ("test", 2020, 40),
    ]:
        for i in range(n):
            day = pd.Timestamp(year, 1, 1) + pd.Timedelta(days=i)
            item = {
                name: float(rng.uniform(0.1, 0.9))
                for name in FEATURES_BASE + FEATURES_WEATHER
            }
            item.update(
                region_id=locate(0, 0)["id"],
                region_name="synthetic",
                continent="synthetic",
                spatial_group="train",
                period_start=day,
                period_end_exclusive=day + pd.Timedelta(days=1),
                split=split,
                event=i % 2,
                produced_at=day + pd.Timedelta(days=2),
                weather_last_date=day - pd.Timedelta(days=3),
                latitude=0.0,
                longitude=0.0,
                xllcorner=-20000,
                yllcorner=-20000,
                ncols=40,
                nrows=40,
                cellsize_m=1000.0,
            )
            rows.append(item)
    path = project.path / "processed/mask_examples.parquet"
    path.parent.mkdir()
    pd.DataFrame(rows).to_parquet(path, index=False)
    write_json(
        path.parent / "manifest.json",
        {
            "examples_sha256": sha256_file(path),
            "code_sha256": sha256_file(
                Path(__file__).parents[1] / "src/earthfire/global_pipeline.py"
            ),
        },
    )
    measured = model(project)
    assert measured["training_n"] == 120
    result = predict(0, 0, "2020-01-01", project.path)
    assert result["status"] == "ok"
    assert 0 < result["probability"] < 1
    assert result["historical_replay"] is True
    assert model(project) == measured


def test_preparation_preserves_unknowns_and_temporal_boundaries(project, monkeypatch):
    from earthfire.global_pipeline import prepare
    from earthfire.io import sha256_file, write_json

    project.config.update(
        mask={"start_year": 2022, "end_year": 2024},
        weather_start="2021-12-01",
        weather_end="2024-12-31",
        split={
            "train_end": "2022-12-31",
            "validation_end": "2023-12-31",
            "test_start": "2024-01-01",
            "test_end": "2024-12-31",
        },
    )
    project.config["regions"] = [
        {
            "name": "test",
            "latitude": 0.0,
            "longitude": 0.0,
            "continent": "synthetic",
            "group": "train",
        }
    ]
    coverage = project.path / "coverage"
    coverage.mkdir()
    masks = []
    for year, event in [(2022, 0), (2023, None), (2024, 1)]:
        masks.append(
            dict(
                region_id=locate(0, 0)["id"],
                region_name="test",
                spatial_group="train",
                latitude=0.0,
                longitude=0.0,
                period_start=f"{year}-01-01",
                period_end_exclusive=f"{year}-01-09",
                proc_date=f"{year}010000000",
                event=event,
                cloud_pixels=int(event is None),
                xllcorner=-20000.0,
                yllcorner=-20000.0,
                nrows=40,
                ncols=40,
                cellsize_m=1000.0,
            )
        )
    source = coverage / "mask_summary.parquet"
    pd.DataFrame(masks).to_parquet(source, index=False)
    write_json(coverage / "summary.json", {"sha256": sha256_file(source)})
    sat = project.root / "runs" / (project.run_id + "_records") / "satellite"
    sat.mkdir(parents=True)
    raw = sat / "viirs-jpss1_2022.parquet"
    pd.DataFrame(
        {
            "region_id": [locate(0, 0)["id"]],
            "latitude": [0.0],
            "longitude": [0.0],
            "product": ["viirs-jpss1"],
            "satellite": ["N20"],
            "acquired_utc": pd.to_datetime(["2022-01-01"], utc=True),
        }
    ).to_parquet(raw, index=False)
    write_json(
        sat / "manifest.json",
        {
            "files": {
                "viirs-jpss1_2022_all_countries.zip": {
                    "output_sha256": sha256_file(raw)
                }
            }
        },
    )

    def weather(*args, **kwargs):
        dates = pd.date_range("2021-12-01", "2024-12-31").strftime("%Y%m%d")
        units = {
            "T2M": "C",
            "RH2M": "%",
            "PRECTOTCORR": "mm/day",
            "WS10M": "m/s",
            "GWETROOT": "1",
        }
        return {
            "request": {"parameters": ",".join(units)},
            "response": {
                "parameters": {k: {"units": v} for k, v in units.items()},
                "properties": {
                    "parameter": {k: dict.fromkeys(dates, 1.0) for k in units}
                },
            },
        }

    monkeypatch.setattr("earthfire.global_pipeline.NasaPowerClient.fetch", weather)
    result = prepare(project)
    data = pd.read_parquet(project.path / "processed/mask_examples.parquet")
    assert result["unknown"] == 1
    assert data.loc[data.event.isna(), "observation_state"].iloc[0] == "cloud_obscured"
    assert (data.weather_last_date < data.period_start).all()
    assert len(pd.read_parquet(project.path / "processed/coverage_gaps.parquet")) > 0
    assert data.split.tolist() == ["train", "validation", "test"]
