"""Features causais para previsão semanal condicionada e dois sensores."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .features import temporal_features
from .ingest import region_id
from .power_zarr import VARIABLES

BASE = ["log_viirs_lag1", "season_sin", "season_cos", "latitude", "longitude"]
SENSOR = ["log_modis_terra_lag1"]
PAST = ["past_" + x for x in VARIABLES]
ROLLING = ["past4_T2M", "past4_RH2M", "past4_PRECTOTCORR", "past4_GWETROOT"]
FUTURE = ["future_" + x for x in VARIABLES]


def weather_features(weekly_weather: pd.DataFrame) -> pd.DataFrame:
    weather = weekly_weather.sort_values(["region_id", "week_start"]).copy()
    weather["week_start"] = pd.to_datetime(weather["week_start"])
    delta = weather.groupby("region_id").week_start.diff()
    weather["_sequence"] = delta.ne(pd.Timedelta(days=7)).groupby(weather.region_id).cumsum()
    grouped = weather.groupby(["region_id", "_sequence"], sort=False)
    for source, target, op in [("T2M", "past4_T2M", "mean"), ("RH2M", "past4_RH2M", "mean"),
                               ("PRECTOTCORR", "past4_PRECTOTCORR", "sum"), ("GWETROOT", "past4_GWETROOT", "mean")]:
        rolling = grouped[source].rolling(4, min_periods=4)
        weather[target] = (rolling.mean() if op == "mean" else rolling.sum()).reset_index(level=[0, 1], drop=True).sort_index()
    weather = weather.drop(columns="_sequence")
    weather = weather.rename(columns={v: "past_" + v for v in VARIABLES})
    # O último dia meteorológico utilizado antecede a emissão em pelo menos sete dias.
    weather["week_start"] += pd.Timedelta(days=14)
    return weather


def candidates(weekly_fire: pd.DataFrame, weekly_weather: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    data = weekly_fire.copy()
    data["week_start"] = pd.to_datetime(data.week_start)
    viirs = data.loc[(data["product"] == "viirs-jpss1") & (data.satellite == "N20")]
    modis = data.loc[(data["product"] == "modis") & (data.satellite == "Terra")]
    viirs = viirs.groupby(["region_id", "week_start"], as_index=False).detections.sum().rename(columns={"detections": "viirs_lag1"})
    modis = modis.groupby(["region_id", "week_start"], as_index=False).detections.sum().rename(columns={"detections": "modis_terra_lag1"})
    viirs["week_start"] += pd.Timedelta(days=7)
    modis["week_start"] += pd.Timedelta(days=7)
    frame = viirs.merge(modis, on=["region_id", "week_start"], how="inner", validate="one_to_one")
    frame = frame.merge(weather_features(weekly_weather), on=["region_id", "week_start"], how="inner", validate="one_to_one")
    coordinates = pd.DataFrame([{"region_id": region_id(r["latitude"], r["longitude"]), "latitude": r["latitude"],
                                 "longitude": r["longitude"]} for r in config["regions"]])
    frame = frame.merge(coordinates, on="region_id", how="inner", validate="many_to_one")
    frame["log_viirs_lag1"], frame["season_sin"], frame["season_cos"] = temporal_features(frame.viirs_lag1, frame.week_start.dt.dayofyear)
    frame["log_modis_terra_lag1"] = np.log1p(frame.modis_terra_lag1)
    frame["origin"] = frame.week_start - pd.Timedelta(seconds=1)
    frame["week_end"] = frame.week_start + pd.Timedelta(days=6)
    frame["spatial_group"] = np.where(frame.region_id.isin(config["spatial_holdout_regions"]), "holdout", "train_regions")
    frame = frame.dropna(subset=BASE + SENSOR + PAST + ROLLING)
    return frame.sort_values(["week_start", "region_id"]).reset_index(drop=True)


def fit_climatology(weekly_weather: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    weather = weekly_weather.copy()
    weather["week_start"] = pd.to_datetime(weather.week_start)
    train_end = pd.Timestamp(config["split"]["train_end"])
    eligible = {region_id(r["latitude"], r["longitude"]) for r in config["regions"]} - set(config["spatial_holdout_regions"])
    weather = weather.loc[(weather.week_start + pd.Timedelta(days=6) <= train_end) &
                          (weather.region_id.isin(eligible))].copy()
    weather["month"] = weather.week_start.dt.month
    local = weather.groupby(["region_id", "month"], as_index=False)[list(VARIABLES)].mean()
    pooled = weather.groupby("month", as_index=False)[list(VARIABLES)].mean()
    pooled["region_id"] = "__pooled_train_regions__"
    return pd.concat([local, pooled], ignore_index=True)


def with_climatology(frame: pd.DataFrame, climatology: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    month = result.week_start.dt.month
    local = climatology.set_index(["region_id", "month"])
    rows = []
    for rid, m in zip(result.region_id, month):
        key = (rid, int(m))
        if key not in local.index:
            key = ("__pooled_train_regions__", int(m))
        rows.append(local.loc[key, list(VARIABLES)].to_numpy(dtype=float) if key in local.index else np.full(len(VARIABLES), np.nan))
    arr = np.asarray(rows, dtype=float).reshape(len(result), len(VARIABLES))
    for i, name in enumerate(FUTURE):
        result[name] = arr[:, i]
    return result


def build_examples(weekly_fire: pd.DataFrame, weekly_weather: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_rows = candidates(weekly_fire, weekly_weather, config)
    labels = weekly_fire.loc[(weekly_fire["product"] == "viirs-jpss1") & (weekly_fire.satellite == "N20")].copy()
    labels = labels.groupby(["region_id", "week_start"], as_index=False).detections.sum()
    labels["week_start"] = pd.to_datetime(labels.week_start)
    frame = feature_rows.merge(labels, on=["region_id", "week_start"], how="inner", validate="one_to_one")
    frame["split"] = "excluded"
    frame.loc[frame.week_end <= pd.Timestamp(config["split"]["train_end"]), "split"] = "train"
    frame.loc[(frame.week_start > pd.Timestamp(config["split"]["train_end"])) &
              (frame.week_end <= pd.Timestamp(config["split"]["validation_end"])), "split"] = "validation"
    frame.loc[(frame.week_start >= pd.Timestamp(config["split"]["test_start"])) &
              (frame.week_end <= pd.Timestamp(config["split"]["test_end"])), "split"] = "test"
    # Trajetória meteorológica da própria semana só entra na estimação para
    # exemplos de treino; nunca é usada como previsão em validação ou teste.
    future = weekly_weather.rename(columns={v: "future_" + v for v in VARIABLES})
    frame = frame.merge(future, on=["region_id", "week_start"], how="left", validate="one_to_one")
    fitted = (frame["split"] == "train") & (frame.spatial_group == "train_regions")
    frame.loc[~fitted, FUTURE] = np.nan
    climate = fit_climatology(weekly_weather, config)
    return frame.sort_values(["week_start", "region_id"]).reset_index(drop=True), climate
