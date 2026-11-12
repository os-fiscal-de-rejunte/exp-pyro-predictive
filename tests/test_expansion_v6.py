"""Testes sintéticos de geometria e ausências; não medem desempenho científico."""

import numpy as np
import pandas as pd
import pytest

from earthfire.expansion_grid import (
    PIXEL,
    SIDE,
    XMIN,
    YMAX,
    geometry,
    point_cell,
    split_body,
)
from earthfire.expansion_sparse_weather import SparseWeather


def make_body():
    r0, c0 = 41 * 200 + 7, 41 * 400 + 9
    values = np.full((201, 201), 5, dtype=int)
    rr, cc = (-r0) % 41, (-c0) % 41
    values[rr, cc] = 7
    values[rr + 1, cc + 1] = 4
    return {
        "xllcorner": XMIN + c0 * PIXEL,
        "yllcorner": YMAX - (r0 + 201) * PIXEL,
        "nrows": 201,
        "ncols": 201,
        "cellsize": PIXEL,
        "band": "FireMask",
        "units": "class-flag",
        "subset": [
            {
                "data": values.ravel().tolist(),
                "calendar_date": "2020-01-01",
                "proc_date": "2020010000000",
            }
        ],
    }


def test_native_grid_subdivision_is_disjoint_and_cloud_is_unknown():
    rows = split_body(
        make_body(),
        {"name": "synthetic", "continent": "synthetic", "group": "train"},
        "synthetic",
    )
    assert len(rows) == 16 and len({x["region_id"] for x in rows}) == 16
    assert sum(x["area_km2"] for x in rows) == pytest.approx(
        16 * (SIDE * PIXEL) ** 2 / 1e6
    )
    assert rows[0]["fire_pixels"] == 1 and rows[0]["event"] is None
    assert all(x["event"] == 0 for x in rows[1:])
    for row in rows:
        assert (
            point_cell(row["latitude"], row["longitude"])["region_id"]
            == row["region_id"]
        )
    pairs = [(x["native_row"], x["native_column"]) for x in rows]
    assert all(
        a == b or abs(a[0] - b[0]) >= 41 or abs(a[1] - b[1]) >= 41
        for a in pairs
        for b in pairs
    )


def test_invalid_classes_are_rejected():
    body = make_body()
    body["subset"][0]["data"][0] = -1
    with pytest.raises(ValueError):
        split_body(body, {"name": "s", "continent": "s", "group": "train"}, "s")


def test_weather_budget_does_not_create_zero_weather(project, monkeypatch):
    project.config["operational"]["nasa_power_requests"] = 0
    weather = SparseWeather(project)
    cells = pd.DataFrame([{"region_id": "native", "latitude": 0.0, "longitude": 0.0}])
    monkeypatch.setattr(
        "earthfire.expansion_sparse_weather.NasaPowerClient.fetch",
        lambda *a, **k: pytest.fail("Rede após limite"),
    )
    mapping = weather.prepare(cells)
    frame = weather.features(mapping["native"], pd.to_datetime(["2024-01-01"]))
    assert (
        frame.filter(like="weather_")
        .drop(columns="weather_last_date")
        .isna()
        .all()
        .all()
    )
    assert weather.state["budget_exhausted"]
    weather.session.close()


def test_missing_weather_preserves_observational_labels(project):
    from earthfire.expansion_features import prepare_parent
    from earthfire.io import sha256_file, write_json

    meta = geometry(41 * 200, 41 * 400)
    folder = project.path / "blocks"
    folder.mkdir()
    rows = []
    for year, event in [(2022, 0), (2023, None), (2024, 1)]:
        rows.append(
            {
                **meta,
                "parent_id": "synthetic",
                "continent": "synthetic",
                "spatial_group": "train",
                "period_start": f"{year}-01-01",
                "proc_date": f"{year}010000000",
                "event": event,
                "train_support": True,
                "cloud_pixels": int(event is None),
                "legacy_context": False,
            }
        )
    path = folder / "synthetic.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    write_json(
        project.path / "mask_manifest.json",
        {"partitions": {"synthetic": {"sha256": sha256_file(path)}}},
    )

    class Missing:
        base = project.path / "weather"

        def prepare(self, cells):
            return {meta["region_id"]: "missing"}

        def climate_key(self, *args):
            return "missing", 0, 0

        def features(self, rid, dates):
            return pd.DataFrame(
                {
                    **{
                        f"weather_{v}": np.full(len(dates), np.nan)
                        for v in ("T2M", "RH2M", "PRECTOTCORR", "WS10M", "GWETROOT")
                    },
                    "weather_last_date": pd.DatetimeIndex(dates) - pd.Timedelta(days=3),
                    "climate_cell_id": rid,
                }
            )

    result = prepare_parent(project, "synthetic", Missing(), set())
    data = pd.read_parquet(project.path / "features/synthetic.parquet")
    assert result["cells"] == 1 and len(data) == 3
    assert data.event.notna().sum() == 2 and not data.weather_complete.any()
    assert data.loc[data.event.isna(), "observation_state"].iloc[0] == "cloud_obscured"


@pytest.mark.parametrize("weather_available", [True, False])
def test_synthetic_finalization_and_future_projection(project, weather_available):
    from earthfire.expansion_evaluate import finalize
    from earthfire.expansion_predict import forecast, predict
    from earthfire.io import write_json, sha256_file
    from earthfire.mask_features import FEATURES_BASE, FEATURES_WEATHER

    project.config["global_model"] = {"steps": 20}
    project.config["expansion"] = {"max_training_examples": 1000}
    project.config["release"] = {
        "min_n": 30,
        "min_class_n": 5,
        "max_mean_calibration_error": 0.1,
        "max_brier_ratio": 1.0,
    }
    folder = project.path / "features"
    folder.mkdir()
    rng = np.random.default_rng(18)
    first = None
    for index in range(2):
        cell = geometry(41 * (200 + 30 * index), 41 * 400)
        first = first or cell
        rows = []
        for split, year, n in [
            ("train", 2022, 120),
            ("validation", 2023, 40),
            ("test", 2024, 40),
        ]:
            for i in range(n):
                day = pd.Timestamp(year, 1, 1) + pd.Timedelta(days=i)
                item = {
                    name: float(rng.uniform(0.1, 0.9))
                    for name in FEATURES_BASE + FEATURES_WEATHER
                }
                item.update(cell)
                if not weather_available:
                    for name in FEATURES_WEATHER:
                        item[name] = np.nan
                item.update(
                    parent_id=f"synthetic_{index}",
                    continent="synthetic",
                    spatial_group="train",
                    legacy_context=False,
                    period_start=day,
                    period_end_exclusive=day + pd.Timedelta(days=8),
                    event=i % 2 if i != 10 or split != "test" else None,
                    weather_complete=weather_available,
                    split=split,
                    weather_last_date=day - pd.Timedelta(days=3),
                )
                rows.append(item)
        path = folder / f"synthetic_{index}.parquet"
        pd.DataFrame(rows).to_parquet(path, index=False)
        write_json(path.with_suffix(".json"), {"sha256": sha256_file(path)})
    result = finalize(project)
    assert result["test_n"] == 78 and result["released_cells"] == 2
    prediction = forecast(
        first["latitude"], first["longitude"], project.path, "2026-11-11T12:00:00Z"
    )
    assert prediction["status"] == "ok" and 0 < prediction["probability"] < 1
    assert (
        prediction["horizon_days"] == 8 and prediction["mode"] == "seasonal_projection"
    )
    historical = predict(
        first["latitude"], first["longitude"], "2024-01-11", project.path
    )
    assert historical["status"] == "ok"
    assert finalize(project) == result


def test_probability_scores_for_balanced_constant_prediction():
    from earthfire.expansion_evaluate import summarize

    result = summarize([0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5])
    assert result["brier"] == pytest.approx(0.25)
    assert result["log_loss"] == pytest.approx(np.log(2))
    assert result["roc_auc"] == pytest.approx(0.5)
    assert result["average_precision"] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        summarize([0], [np.nan])
