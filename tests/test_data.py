from __future__ import annotations

import hashlib
import shutil
import zipfile

import pandas as pd

from earthfire.ingest import parse_datetime, validation_masks
from earthfire.prepare import make_calendar, prepare
from earthfire.ingest import region_id


def test_parsing_and_product_confidence():
    frame = pd.DataFrame({"acq_date": ["2020-01-01", "2020-01-01", "2020-02-30"],
                          "acq_time": ["5", "2400", "1200"], "latitude": ["0", "91", "0"],
                          "longitude": ["0", "0", "-181"], "confidence": ["0", "100", "n"],
                          "daynight": ["D", "N", "D"]})
    _, when = parse_datetime(frame)
    assert str(when.iloc[0]) == "2020-01-01 00:05:00+00:00"
    assert pd.isna(when.iloc[1]) and pd.isna(when.iloc[2])
    modis = validation_masks(frame, "modis")
    viirs = validation_masks(frame, "viirs-jpss1")
    assert modis["latitude_invalid"].sum() == 1
    assert modis["longitude_invalid"].sum() == 1
    assert modis["confidence_invalid"].tolist() == [False, False, True]
    assert viirs["confidence_invalid"].tolist() == [True, True, False]


def test_exact_dedup_and_unknown_calendar(project):
    root = project.root / "DATA"
    columns = "latitude,longitude,brightness,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_t31,frp,daynight,type\n"
    row = "0.1,0.1,310,1,1,2018-01-01,5,Terra,MODIS,62,6.2,300,10,D,0\n"
    invalid = "91,0.1,310,1,1,2018-01-01,5,Terra,MODIS,62,6.2,300,10,D,0\n"
    archive = root / "modis_2018_all_countries.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("modis/2018/modis_2018_Brazil.csv", columns + row + row + invalid)
    shutil.copyfile(archive, root / "modis_2018_all_countries (1).zip")
    vcolumns = columns.replace("brightness", "bright_ti4").replace("bright_t31", "bright_ti5")
    vrow = "0.1,0.1,310,1,1,2018-01-01,5,N20,VIIRS,n,2,300,10,D,0\n"
    with zipfile.ZipFile(root / "viirs-jpss1_2018_all_countries.zip", "w") as z:
        z.writestr("viirs-jpss1/2018/viirs-jpss1_2018_Brazil.csv", vcolumns + vrow)
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}
    result = prepare(project)
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}
    assert before == after
    assert result["filter_counts"]["exact_duplicates_removed"] == 1
    assert result["filter_counts"]["duplicate_archives_skipped"] == 1
    assert result["filter_counts"]["invalid_coordinates_brazil"] == 1
    records = pd.read_parquet(project.path / "processed/records.parquet")
    assert len(records) == 2 and set(records.instrument) == {"MODIS", "VIIRS"}
    weekly = pd.read_parquet(project.path / "processed/weekly_positive.parquet")
    assert len(weekly) == 2 and weekly.detections.sum() == 2
    calendar = pd.read_parquet(project.path / "processed/calendar.parquet")
    absent = calendar[calendar.observation_state == "missing_unknown"]
    assert len(absent) and absent.detections.isna().all()


def test_calendar_anomaly_uses_only_previous_positive_values(project):
    rid = region_id(0., 0.)
    weekly = pd.DataFrame({"region_id": [rid] * 4, "product": ["modis"] * 4, "satellite": ["Terra"] * 4,
        "week_start": pd.to_datetime(["2001-01-01", "2002-01-07", "2003-01-06", "2004-01-05"]),
        "detections": [2, 4, 6, 20]})
    calendar = make_calendar(weekly, project.config, [])
    result = calendar.loc[(calendar.week_start == "2004-01-05") & (calendar["product"] == "modis")].iloc[0]
    assert result.historical_reference == 4 and result.observed_anomaly == 16
