from datetime import date

import pandas as pd

from earthfire.mask_features import FEATURES_WEATHER, build_examples, period_start
from earthfire.nasa_complement import _mask_summary, mask_plan, split_mask_query


def test_modis_mask_distinguishes_clear_cloud_and_fire():
    region = {"name": "a", "latitude": -11.5, "longitude": -46.875, "group": "train"}
    body = {"nrows": 2, "ncols": 2, "band": "FireMask", "units": "class-flag",
            "xllcorner": 0, "yllcorner": 0, "cellsize": 926.625,
            "subset": [
                {"calendar_date": "2024-01-01", "modis_date": "A2024001", "data": [5, 5, 5, 5]},
                {"calendar_date": "2024-01-09", "modis_date": "A2024009", "data": [5, 4, 5, 5]},
                {"calendar_date": "2024-01-17", "modis_date": "A2024017", "data": [5, 7, 5, 5]},
            ]}
    entries = _mask_summary({"body": body, "body_sha256": "x", "retrieved_utc": "2026-01-01"}, region)
    assert entries[0]["event"] == 0 and entries[0]["fully_observed"]
    assert entries[1]["event"] is None and not entries[1]["fully_observed"]
    assert entries[2]["event"] == 1 and entries[2]["fire_pixels"] == 1


def test_composite_calendar_resets_in_january():
    days = pd.Series(pd.to_datetime(["2023-12-31", "2024-01-01", "2024-01-09"], utc=True))
    assert period_start(days).dt.strftime("%Y-%m-%d").tolist() == ["2023-12-27", "2024-01-01", "2024-01-09"]
    plan = mask_plan({"mask": {"start_year": 2024, "end_year": 2024},
                      "regions": [{"latitude": 0., "longitude": 0., "name": "a", "group": "train"}]})
    assert len(plan) == 6
    assert all(int(x["end"][-3:]) - int(x["start"][-3:]) <= 72 for x in plan)
    children = split_mask_query(plan[0])
    assert [(x["start"], x["end"]) for x in children] == [("A2024001", "A2024033"), ("A2024041", "A2024073")]


def test_weather_features_use_past_and_unknown_coverage_remains_unknown():
    rid = "grid_-011.50_-046.875"
    starts = pd.to_datetime(["2023-01-01", "2023-01-09", "2023-01-17", "2023-01-25", "2023-02-02"])
    masks = pd.DataFrame([{"region_id": rid, "period_start": day, "period_end_exclusive": day + pd.Timedelta(days=8),
                           "event": event, "proc_date": (day + pd.Timedelta(days=9)).strftime("%Y%j%H%M%S"),
                           "latitude": -11.5, "longitude": -46.875, "spatial_group": "train"}
                          for day, event in zip(starts, [0, 1, None, 0, 1])])
    days = pd.date_range("2022-12-01", "2023-02-15", freq="D")
    weather = pd.DataFrame([{"region_id": rid, "date": day, "T2M": 25., "RH2M": 70.,
                             "PRECTOTCORR": 1., "WS10M": 2., "GWETROOT": .5} for day in days])
    counts = pd.DataFrame(columns=["region_id", "period_start", "records"])
    config = {"split": {"train_end": "2022-12-31", "validation_end": "2023-12-31",
                        "test_start": "2024-01-01", "test_end": "2025-12-31"}}
    result = build_examples(masks, counts, weather, config)
    assert result.loc[result.event.isna(), "split"].tolist() == ["unknown_coverage"]
    row = result.loc[result.period_start == pd.Timestamp("2023-02-02")].iloc[0]
    assert row.past_modis_missing == 1 and pd.isna(row.past_viirs_log_records)
    assert row.weather_last_date == pd.Timestamp("2023-01-30")
    altered = weather.copy()
    altered.loc[altered.date >= "2023-02-01", "T2M"] = 999
    second = build_examples(masks, counts, altered, config)
    a = result.loc[result.period_start == "2023-02-02", list(FEATURES_WEATHER)].reset_index(drop=True)
    b = second.loc[second.period_start == "2023-02-02", list(FEATURES_WEATHER)].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)
