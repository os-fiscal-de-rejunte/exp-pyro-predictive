"""Rastreia versões literais dos sensores usadas nas defasagens."""
from __future__ import annotations

import pandas as pd


def build_version_index(records: pd.DataFrame) -> pd.DataFrame:
    raw = records.loc[((records["product"] == "modis") & (records.satellite == "Terra")) |
                      ((records["product"] == "viirs-jpss1") & (records.satellite == "N20"))].copy()
    raw["week_start"] = raw.acquired_utc.dt.tz_localize(None).dt.to_period("W-SUN").dt.start_time
    result = raw.groupby(["region_id", "week_start", "product"])["version"].agg(
        lambda values: "|".join(sorted(set(str(x) for x in values.dropna() if str(x)))))
    result = result.unstack("product").reset_index()
    result = result.rename(columns={"modis": "modis_terra_versions", "viirs-jpss1": "viirs_n20_versions"})
    for name in ("modis_terra_versions", "viirs_n20_versions"):
        if name not in result:
            result[name] = pd.NA
    return result


def annotate_versions(frame: pd.DataFrame, training: pd.DataFrame, version_index: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    def merge(source: pd.DataFrame) -> pd.DataFrame:
        left = source.copy()
        left["_previous_week"] = pd.to_datetime(left.week_start) - pd.Timedelta(days=7)
        return left.merge(version_index, left_on=["region_id", "_previous_week"], right_on=["region_id", "week_start"],
                          how="left", suffixes=("", "_versions"), validate="many_to_one").drop(columns=["_previous_week", "week_start_versions"])
    train = merge(training)
    known: dict[str, list[str]] = {}
    for col in ("modis_terra_versions", "viirs_n20_versions"):
        known[col] = sorted({x for entry in train[col].dropna().astype(str) for x in entry.split("|") if x})
    result = merge(frame)
    def status(row: pd.Series) -> str:
        for col in known:
            value = row[col]
            if pd.isna(value) or not str(value): return "unknown"
            if any(v not in known[col] for v in str(value).split("|")): return "unseen_version"
        return "seen_versions"
    result["sensor_version_support"] = result.apply(status, axis=1)
    return result, known
