import numpy as np
import pandas as pd

from earthfire.dual_features import BASE, FUTURE, PAST, ROLLING, SENSOR
from earthfire.power_zarr import VARIABLES


def test_selection_precedes_test_and_spatial_groups(project, monkeypatch):
    import earthfire.dual_evaluate as module
    rows = []
    for split, year, regions in [("train", 2018, ["a"]), ("validation", 2019, ["a", "b"]), ("test", 2020, ["a", "b"])]:
        for region in regions:
            for i in range(6):
                week = pd.Timestamp(year=year, month=1, day=7) + pd.Timedelta(days=7*i)
                item = {x: float(i + 1) for x in BASE + SENSOR + PAST + ROLLING + FUTURE}
                item.update(region_id=region, week_start=week, detections=5, viirs_lag1=4,
                            split=split, spatial_group="holdout" if region == "b" else "train_regions")
                rows.append(item)
    frame = pd.DataFrame(rows)
    frame.loc[frame.split != "train", FUTURE] = np.nan
    (project.path / "processed_v2").mkdir()
    observations = []
    for row in frame.itertuples():
        for product, satellite in [("modis", "Terra"), ("viirs-jpss1", "N20")]:
            observations.append({"region_id": row.region_id, "acquired_utc": (row.week_start - pd.Timedelta(days=7)).tz_localize("UTC"),
                                 "product": product, "satellite": satellite, "version": "1"})
    pd.DataFrame(observations).to_parquet(project.path / "processed_v2/records.parquet", index=False)
    climate = pd.DataFrame([{"region_id": rid, "month": month, **{x: 1. for x in VARIABLES}}
                            for rid in ["a", "__pooled_train_regions__"] for month in range(1, 13)])
    monkeypatch.setattr(module, "posterior_counts", lambda model, data, n_samples, seed:
                        (np.full((n_samples, len(data)), 5. if str(model) == "weather" else 2.), np.zeros((n_samples, len(data)))))
    metrics = module.evaluate(project, frame, climate, {"weather": "weather", "no_weather": "no_weather"}, n_samples=10)
    assert metrics["selection"]["variant"] == "weather" and metrics["selection"]["locked_before_test"]
    assert set(metrics["test"]["weather"]["spatial_groups"]) == {"train_regions", "holdout"}
    assert (project.path / "evaluation_v2/prediction_test.png").exists()
