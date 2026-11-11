from __future__ import annotations

import pandas as pd
import pytest
import requests
import json
import numpy as np
import torch

from earthfire.features import WEATHER, build_examples
from earthfire.power import NasaPowerClient, WeatherUnavailable, grid_center, power_frame
from earthfire.power import EXPECTED_UNITS
from earthfire.scenario import _forecast, _modify, simulate_fire_scenario


class Response:
    status_code = 200
    headers = {}
    def raise_for_status(self): pass
    def json(self):
        return {"header": {"fill_value": -999}, "properties": {"parameter": {"T2M": {"20200101": 25, "20200102": -999}}}}


class Session:
    def __init__(self): self.calls = 0
    def get(self, *args, **kwargs): self.calls += 1; return Response()


def test_power_cell_cache_and_missing(tmp_path):
    session = Session()
    client = NasaPowerClient(tmp_path, session=session, min_interval=0)
    first = client.fetch(-10., -64.375, "2020-01-01", "2020-01-02", parameters=["T2M"])
    second = client.fetch(-10.01, -64.36, "2020-01-01", "2020-01-01", parameters=["T2M"])
    assert session.calls == 1 and second["effective_request"]["end"] == "20200101"
    assert pd.isna(power_frame(first).T2M.iloc[1])
    assert len(power_frame(second)) == 1
    with pytest.raises(ValueError): grid_center(91, 0)
    with pytest.raises(ValueError): client.fetch(0, 0, "2021-01-02", "2021-01-01")


def test_power_network_failure_explicit(tmp_path):
    class Failed:
        def get(self, *args, **kwargs): raise requests.Timeout("offline")
    client = NasaPowerClient(tmp_path, session=Failed(), retries=1, min_interval=0)
    with pytest.raises(WeatherUnavailable, match="offline"):
        client.fetch(0, 0, "2020-01-01", "2020-01-02")


def test_feature_climatology_has_no_test_leakage(project):
    weeks = pd.date_range("2018-01-01", "2020-12-28", freq="W-MON")
    weekly = pd.DataFrame({"region_id": "a", "product": "viirs-jpss1", "satellite": "N20", "week_start": weeks, "detections": 3})
    weather = pd.DataFrame({"region_id": "a", "week_start": weeks})
    for name in WEATHER: weather[name] = 10.0
    examples, climate = build_examples(weekly, weather, project.config)
    changed = weather.copy()
    changed.loc[changed.week_start >= "2019-07-01", WEATHER] = 999.
    examples2, climate2 = build_examples(weekly, changed, project.config)
    pd.testing.assert_frame_equal(climate, climate2)
    pd.testing.assert_frame_equal(examples[examples.split == "train"], examples2[examples2.split == "train"])
    assert (examples.week_start - examples.origin == pd.Timedelta(seconds=1)).all()
    assert examples.loc[examples.split == "train", "week_end"].max() <= pd.Timestamp("2018-12-31")


def test_scenario_bounds_and_missing_model(tmp_path):
    values = {"T2M": 25., "T2M_MAX": 30., "T2M_MIN": 20., "RH2M": 50., "PRECTOTCORR": 10., "WS10M": 2.}
    altered = _modify(values, {"relative_humidity_delta_percentage_points": -5, "precipitation_multiplier": .8})
    assert altered["RH2M"] == 45 and altered["PRECTOTCORR"] == 8
    with pytest.raises(ValueError): _modify(values, {"relative_humidity_delta_percentage_points": -100})
    response = simulate_fire_scenario(0, 0, "2024-09-29", 7, {"run_dir": tmp_path, "power_cache": tmp_path})
    assert response["status"] == "insufficient_data" and response["event_probability"] is None


def test_forecast_temporal_spatial_and_daily_validation(tmp_path):
    days = pd.date_range("2024-01-08", periods=7)
    doc = {"issued_at": "2024-01-07T12:00:00Z", "source": "synthetic forecast for test", "units": EXPECTED_UNITS,
           "location": {"latitude": 0, "longitude": 0}, "daily": [
               {"date": d.date().isoformat(), "T2M": 25, "T2M_MAX": 30, "T2M_MIN": 20,
                "RH2M": 50, "PRECTOTCORR": 1, "WS10M": 2} for d in days]}
    target = tmp_path / "forecast.json"
    def save(): target.write_text(json.dumps(doc), encoding="utf-8")
    save()
    values, metadata = _forecast(target, pd.Timestamp("2024-01-07T23:59:59"), days[0], 0., 0.)
    assert values["PRECTOTCORR"] == 7 and metadata["source"] == doc["source"]
    doc["issued_at"] = "2024-01-08T00:00:00Z"; save()
    with pytest.raises(ValueError, match="emissão"): _forecast(target, pd.Timestamp("2024-01-07T23:59:59"), days[0], 0., 0.)
    doc["issued_at"] = "2024-01-07T12:00:00Z"; doc["daily"][0]["RH2M"] = 101; save()
    with pytest.raises(ValueError, match="físicos"): _forecast(target, pd.Timestamp("2024-01-07T23:59:59"), days[0], 0., 0.)


def test_scenario_blocks_extrapolation_and_never_invents_probability(tmp_path, monkeypatch):
    import earthfire.scenario as module
    from earthfire.ingest import region_id
    from earthfire.features import BASE_FEATURES, PAST_WEATHER, FUTURE_WEATHER
    rid = region_id(0, 0)
    run = tmp_path / "run"; (run / "models").mkdir(parents=True); (run / "processed").mkdir()
    values = {"T2M": 25., "T2M_MAX": 30., "T2M_MIN": 20., "RH2M": 50., "PRECTOTCORR": 7., "WS10M": 2.}
    phase = 2 * np.pi * pd.Timestamp("2024-01-08").dayofyear / 365.2425
    expected = [np.log1p(7), np.sin(phase), np.cos(phase)] + list(values.values()) + list(values.values())
    features = BASE_FEATURES + PAST_WEATHER + FUTURE_WEATHER
    model = {"model_version": "synthetic", "preprocessing_version": "test", "complete": True,
             "training_end": "2021-12-31", "regions": [rid], "features": features,
             "feature_min": expected, "feature_max": expected, "feature_scale": [1.] * len(features)}
    torch.save(model, run / "models/weather.pt")
    (run / "run.json").write_text(json.dumps({"config": {"operational": {"nasa_power_requests": 5,
        "network_timeout_seconds": 2, "network_retries": 1}}}), encoding="utf-8")
    pd.DataFrame([{"region_id": rid, "product": "viirs-jpss1", "satellite": "N20",
                   "week_start": pd.Timestamp("2024-01-01"), "detections": 7}]).to_parquet(run / "processed/weekly_positive.parquet")
    pd.DataFrame([{**values, "region_id": rid, "month": 1}]).to_parquet(run / "processed/climatology.parquet")
    dates = pd.date_range("2023-12-25", periods=7)
    daily = {k: {d.strftime("%Y%m%d"): (1. if k == "PRECTOTCORR" else v) for d in dates} for k, v in values.items()}
    envelope = {"retrieved_utc": "2024-01-01T00:00:00Z", "request": {"parameters": ",".join(WEATHER),
        "start": "20231225", "end": "20231231"}, "response": {"header": {"fill_value": -999},
        "parameters": {k: {"units": v} for k, v in EXPECTED_UNITS.items()}, "properties": {"parameter": daily}}}
    class FakeClient:
        def __init__(self, *a, **k): pass
        def fetch(self, *a, **k): return envelope
    monkeypatch.setattr(module, "NasaPowerClient", FakeClient)
    monkeypatch.setattr(module, "posterior_counts", lambda *a, **k: (np.ones((10, 1)) * 3, np.zeros((10, 1))))
    result = module.simulate_fire_scenario(0, 0, "2024-01-07", 7, {"run_dir": str(run), "power_cache": str(tmp_path)},
                                           {"temperature_delta_c": 5})
    assert result["status"] == "out_of_distribution"
    assert result["modified_scenario"]["expected_count_conditional"] is None
    assert result["event_probability"] is None
    class Offline:
        def __init__(self, *a, **k): pass
        def fetch(self, *a, **k): raise WeatherUnavailable("offline")
    monkeypatch.setattr(module, "NasaPowerClient", Offline)
    unavailable = module.simulate_fire_scenario(0, 0, "2024-01-07", 7, {"run_dir": str(run), "power_cache": str(tmp_path)})
    assert unavailable["status"] == "weather_unavailable" and unavailable["event_probability"] is None
