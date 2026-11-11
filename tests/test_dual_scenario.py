import json

import numpy as np
import pandas as pd

from earthfire.dual_features import BASE, FUTURE, PAST, ROLLING, SENSOR
from earthfire.ingest import region_id
from earthfire.io import sha256_file
from earthfire.power_zarr import VARIABLES


def test_dual_scenario_uses_only_past_and_null_occurrence(project, monkeypatch):
    import earthfire.dual_scenario as module
    root = project.path
    rid = region_id(0, 0)
    project.manifest["config"]["spatial_holdout_regions"] = []
    project.save()
    for name in ("models_v2", "processed_v2", "weather_zarr", "evaluation_v2"):
        (root / name).mkdir()
    (root / "models_v2/weather.pt").write_bytes(b"model")
    (root / "evaluation_v2/selection.json").write_text(json.dumps({"variant": "weather"}), encoding="utf-8")
    (root / "processed_v2/preparation.json").write_text(json.dumps({"inputs": [
        {"archive": "modis_2019_all_countries.zip", "nominal_year": 2019},
        {"archive": "viirs-jpss1_2019_all_countries.zip", "nominal_year": 2019}]}), encoding="utf-8")
    columns = BASE + SENSOR + PAST + ROLLING + FUTURE
    example = {name: 1. for name in columns}
    example.update(region_id=rid, week_start=pd.Timestamp("2018-01-08"), split="train", spatial_group="train_regions")
    pd.DataFrame([example]).to_parquet(root / "processed_v2/examples.parquet", index=False)
    saved = {"complete": True, "data_sha256": sha256_file(root / "processed_v2/examples.parquet"),
        "model_version": "test", "training_end": "2018-12-31", "features": columns,
        "feature_min": [-10000.] * len(columns), "feature_max": [10000.] * len(columns), "feature_scale": [1.] * len(columns)}
    monkeypatch.setattr(module.torch, "load", lambda *args, **kwargs: saved)
    seen = []
    def sample(path, frame, n_samples, seed):
        seen.append(frame.copy()); return np.ones((10,1)) * 3, np.zeros((10,1))
    monkeypatch.setattr(module, "posterior_counts", sample)
    weeks = pd.date_range("2019-05-13", "2019-07-01", freq="W-MON")
    values = {"T2M": 25., "T2M_MAX": 30., "T2M_MIN": 20., "RH2M": 50., "PRECTOTCORR": 7.,
              "WS10M": 2., "GWETROOT": .4, "GWETTOP": .3}
    pd.DataFrame([{"region_id": rid, "week_start": w, **values} for w in weeks]).to_parquet(root / "weather_zarr/weekly.parquet")
    pd.DataFrame([{"region_id": rid, "week_start": pd.Timestamp("2019-07-01"), "product": p, "satellite": s, "detections": 4}
        for p,s in [("modis","Terra"),("viirs-jpss1","N20")]]).to_parquet(root / "processed_v2/weekly_positive.parquet")
    pd.DataFrame([{"region_id": rid, "acquired_utc": pd.Timestamp(w, tz="UTC"), "product": p, "satellite": s, "version": "1"}
        for w in ["2018-01-01", "2019-07-01"] for p,s in [("modis","Terra"),("viirs-jpss1","N20")]]).to_parquet(root / "processed_v2/records.parquet")
    pd.DataFrame([{**values, "region_id": rid, "month": 7}]).to_parquet(root / "processed_v2/climatology.parquet")
    out = module.predict_week(0, 0, "2019-07-07", root, {"temperature_delta_c": 1, "relative_humidity_delta_percentage_points": -3})
    assert out["status"] == "insufficient_data" and out["conditional_count_status"] == "ok"
    assert out["event_probability"] is None and out["previous_week"]["viirs_n20_detections"] == 4
    assert out["data_quality"]["sensor_version_support"] == "seen_versions"
    assert seen[1]["future_T2M"].iloc[0] == seen[0]["future_T2M"].iloc[0] + 1
    assert seen[1]["future_RH2M"].iloc[0] == seen[0]["future_RH2M"].iloc[0] - 3
