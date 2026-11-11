"""Construção cronológica de exemplos e meteorologia condicional."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .ingest import region_id
from .power import EXPECTED_UNITS, PARAMETERS, NasaPowerClient, power_frame
from .runtime import Run
from .io import sha256_file, write_json

WEATHER = ["T2M", "T2M_MAX", "T2M_MIN", "RH2M", "PRECTOTCORR", "WS10M"]
BASE_FEATURES = ["log_lag1", "season_sin", "season_cos"]
PAST_WEATHER = ["past_" + name for name in WEATHER]
FUTURE_WEATHER = ["future_" + name for name in WEATHER]


def temporal_features(lag1: Any, day_of_year: Any) -> tuple[Any, Any, Any]:
    phase = 2 * np.pi * day_of_year / 365.2425
    return np.log1p(lag1), np.sin(phase), np.cos(phase)


def weekly_weather(frame: pd.DataFrame) -> pd.DataFrame:
    """Média diária, exceto precipitação, somada em mm/semana; exige sete dias válidos."""
    data = frame.copy()
    data["week_start"] = data["date"].dt.to_period("W-SUN").dt.start_time
    grouped = data.groupby("week_start", sort=True)
    sizes = grouped[WEATHER].count()
    values = grouped[WEATHER].mean()
    values["PRECTOTCORR"] = grouped["PRECTOTCORR"].sum(min_count=7)
    values = values.loc[sizes.eq(7).all(axis=1)].reset_index()
    return values


def prepare_weather(run: Run, client: NasaPowerClient, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    period = run.config.get("weather_period", {})
    start = start or period.get("start", "2018-01-01")
    end = end or period.get("end", "2025-12-31")
    parts = []
    provenance = []
    for region in run.config["regions"]:
        run.guard()
        envelope = client.fetch(region["latitude"], region["longitude"], start, end)
        actual_units = {k: v.get("units") for k, v in envelope["response"].get("parameters", {}).items()}
        if any(actual_units.get(p) != EXPECTED_UNITS[p] for p in PARAMETERS):
            raise ValueError(f"Unidades POWER inesperadas: {actual_units}")
        weekly = weekly_weather(power_frame(envelope))
        weekly["region_id"] = region_id(region["latitude"], region["longitude"])
        parts.append(weekly)
        key = hashlib.sha256(json.dumps(envelope["request"], sort_keys=True).encode()).hexdigest()
        cached = client.cache_dir / (key + ".json")
        provenance.append({"region_id": weekly["region_id"].iloc[0] if len(weekly) else region_id(region["latitude"], region["longitude"]),
            "cache_path": str(cached), "cache_sha256": sha256_file(cached) if cached.exists() else None,
            "retrieved_utc": envelope["retrieved_utc"], "request": envelope["request"],
            "header": envelope["response"].get("header"), "geometry": envelope["response"].get("geometry"),
            "parameters": envelope["response"].get("parameters")})
    weather = pd.concat(parts, ignore_index=True)
    weather.to_parquet(run.path / "processed" / "weather_weekly.parquet", index=False)
    write_json(run.path / "processed" / "weather_provenance.json", {"sources": provenance})
    return weather


def feature_names(regions: list[str], include_weather: bool) -> list[str]:
    region_terms = ["region=" + x for x in regions[1:]]
    return BASE_FEATURES + (PAST_WEATHER + FUTURE_WEATHER if include_weather else []) + region_terms


def build_examples(weekly: pd.DataFrame, weather: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    fire = weekly.loc[(weekly["product"] == "viirs-jpss1") & (weekly["satellite"] == "N20")].copy()
    fire = fire.groupby(["region_id", "week_start"], as_index=False)["detections"].sum()
    fire["week_start"] = pd.to_datetime(fire["week_start"])
    fire["week_end"] = fire["week_start"] + pd.Timedelta(days=6)
    lag = fire[["region_id", "week_start", "detections"]].copy()
    lag["week_start"] += pd.Timedelta(days=7)
    lag = lag.rename(columns={"detections": "lag1"})
    frame = fire.merge(lag, on=["region_id", "week_start"], how="inner", validate="one_to_one")
    past = weather.copy()
    # Uma semana inteira de folga reduz o risco de tratar a latência POWER como zero.
    # As revisões históricas ainda impedem afirmar disponibilidade operacional exata.
    past["week_start"] += pd.Timedelta(days=14)
    past = past.rename(columns={n: "past_" + n for n in WEATHER})
    future = weather.rename(columns={n: "future_" + n for n in WEATHER})
    frame = frame.merge(past, on=["region_id", "week_start"], how="inner", validate="one_to_one")
    frame = frame.merge(future, on=["region_id", "week_start"], how="inner", validate="one_to_one")
    frame["log_lag1"], frame["season_sin"], frame["season_cos"] = temporal_features(frame["lag1"], frame["week_start"].dt.dayofyear)
    frame["origin"] = frame["week_start"] - pd.Timedelta(seconds=1)
    frame["split"] = "excluded"
    frame.loc[frame["week_end"] <= pd.Timestamp(config["split"]["train_end"]), "split"] = "train"
    frame.loc[(frame["week_start"] > pd.Timestamp(config["split"]["train_end"])) &
              (frame["week_end"] <= pd.Timestamp(config["split"]["validation_end"])), "split"] = "validation"
    frame.loc[(frame["week_start"] >= pd.Timestamp(config["split"]["test_start"])) &
              (frame["week_end"] <= pd.Timestamp(config["split"]["test_end"])), "split"] = "test"
    # Trajetória sazonal futura estimada apenas a partir de meteorologia do treino.
    train_weather = weather.loc[weather["week_start"] + pd.Timedelta(days=6) <= pd.Timestamp(config["split"]["train_end"])].copy()
    train_weather["month"] = train_weather["week_start"].dt.month
    climate = train_weather.groupby(["region_id", "month"], as_index=False)[WEATHER].mean()
    # O vocabulário regional é definido somente pelo conjunto de treino.
    for region in sorted(frame.loc[frame.split == "train", "region_id"].unique()):
        frame["region=" + region] = (frame.region_id == region).astype(float)
    return frame.sort_values(["week_start", "region_id"]).reset_index(drop=True), climate


def replace_future_by_climatology(frame: pd.DataFrame, climate: pd.DataFrame) -> pd.DataFrame:
    temp = frame.copy()
    temp["month"] = temp["week_start"].dt.month
    temp = temp.drop(columns=FUTURE_WEATHER, errors="ignore")
    climate = climate.rename(columns={name: "future_" + name for name in WEATHER})
    return temp.merge(climate, on=["region_id", "month"], how="left", validate="many_to_one").drop(columns="month")
