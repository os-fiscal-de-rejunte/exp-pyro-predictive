import numpy as np
import pandas as pd

from earthfire.dual_features import BASE, FUTURE, SENSOR, build_examples, candidates, with_climatology
from earthfire.ingest import region_id
from earthfire.power_zarr import VARIABLES


def test_causal_features_spatial_holdout_and_no_weather_leakage(project):
    regions = [{"name": "a", "latitude": 0., "longitude": 0.}, {"name": "b", "latitude": 1., "longitude": 1.25}]
    project.config["regions"] = regions
    holdout = region_id(1., 1.25)
    project.config["spatial_holdout_regions"] = [holdout]
    weeks = pd.date_range("2018-01-01", "2020-12-28", freq="W-MON")
    rows, weather = [], []
    for r in regions:
        rid = region_id(r["latitude"], r["longitude"])
        for i, week in enumerate(weeks):
            for product, satellite in [("modis", "Terra"), ("viirs-jpss1", "N20")]:
                rows.append({"region_id": rid, "product": product, "satellite": satellite, "week_start": week, "detections": i % 6 + 1})
            weather.append({"region_id": rid, "week_start": week, **{name: 10. + j + i / 100. for j, name in enumerate(VARIABLES)}})
    fire, met = pd.DataFrame(rows), pd.DataFrame(weather)
    examples, climate = build_examples(fire, met, project.config)
    assert not examples.empty and examples.loc[examples.split == "test", FUTURE].isna().all().all()
    assert examples.loc[(examples.split == "train") & (examples.spatial_group == "holdout"), FUTURE].isna().all().all()
    assert set(climate.region_id) == {region_id(0., 0.), "__pooled_train_regions__"}
    assert (examples.week_start - examples.origin == pd.Timedelta(seconds=1)).all()
    assert (examples.viirs_lag1 > 0).all() and (examples.modis_terra_lag1 > 0).all()
    changed = met.copy()
    changed.loc[(changed.region_id == holdout) | (changed.week_start >= "2019-01-01"), list(VARIABLES)] = 999.
    _, climate_changed = build_examples(fire, changed, project.config)
    pd.testing.assert_frame_equal(climate, climate_changed)
    predicted = with_climatology(examples.loc[examples.split == "test"], climate)
    assert np.isfinite(predicted[FUTURE].to_numpy()).all()


def test_candidates_do_not_require_future_label(project):
    regions = [{"name": "a", "latitude": 0., "longitude": 0.}]
    project.config["regions"] = regions; project.config["spatial_holdout_regions"] = []
    rid = region_id(0., 0.)
    weeks = pd.date_range("2020-01-06", periods=8, freq="W-MON")
    weather = pd.DataFrame([{"region_id": rid, "week_start": w, **{v: 2. for v in VARIABLES}} for w in weeks])
    fire = pd.DataFrame([{"region_id": rid, "week_start": weeks[-1], "product": p, "satellite": s, "detections": 4}
                         for p, s in [("modis", "Terra"), ("viirs-jpss1", "N20")]])
    result = candidates(fire, weather, project.config)
    assert len(result) == 1 and result.iloc[0].week_start == weeks[-1] + pd.Timedelta(days=7)
    assert np.isfinite(result[BASE + SENSOR].to_numpy()).all()
