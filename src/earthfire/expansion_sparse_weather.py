"""POWER Daily para amostragem esparsa; ausência de clima nunca vira valor zero."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import requests

from .ingest import region_id
from .io import read_json, sha256_file, write_json
from .power import NasaPowerClient, WeatherUnavailable, grid_center
from .power import power_frame
from .expansion_weather import VARIABLES, UNITS


class SparseWeather:
    def __init__(self, run):
        self.run = run
        self.base = run.path / "weather"
        self.base.mkdir(exist_ok=True)
        self.cells = self.base / "cells"
        self.cells.mkdir(exist_ok=True)
        self.state_path = self.base / "sparse_provider.json"
        self.state = (
            read_json(self.state_path)
            if self.state_path.exists()
            else {"failures": {}, "mode": "NASA_POWER_DAILY"}
        )
        zarr_budget = self.base / "cache/manifest.json"
        self.spent_zarr = (
            read_json(zarr_budget)["requests"] if zarr_budget.exists() else 0
        )
        self.cache = self.base / "daily_api"
        self.cache.mkdir(exist_ok=True)
        self.counter = self.cache / "request_budget.json"
        self.session = requests.Session()
        self.store = SimpleNamespace(session=self.session)
        self.limit = max(
            0, run.config["operational"]["nasa_power_requests"] - self.spent_zarr
        )
        self.state.update(
            zarr_requests_spent=self.spent_zarr, daily_request_limit=self.limit
        )
        write_json(self.state_path, self.state)
        write_json(
            self.base / "active_source.json",
            {
                "source": "NASA POWER Daily UTC",
                "units": UNITS,
                "reason": "amostragem esparsa; menor volume de transferência que blocos Zarr de 16 anos",
                "missing_weather_policy": "referência sazonal somente quando validada; sem meteorologia inventada",
            },
        )

    def climate_key(self, lat, lon):
        lat, lon = grid_center(float(lat), float(lon))
        return region_id(lat, lon), lat, lon

    def prepare(self, cells):
        mapping = {}
        unique = {}
        for row in cells.itertuples():
            rid, lat, lon = self.climate_key(row.latitude, row.longitude)
            mapping[row.region_id] = rid
            unique[rid] = (lat, lon)
        for rid, (lat, lon) in unique.items():
            self.run.guard(require_training_seconds=1200)
            path = self.cells / (rid + ".parquet")
            meta = path.with_suffix(".json")
            if path.exists() and meta.exists():
                if sha256_file(path) != read_json(meta)["sha256"]:
                    raise ValueError("Clima alterado.")
                continue
            if rid in self.state["failures"]:
                continue
            used = read_json(self.counter)["requests"] if self.counter.exists() else 0
            if used >= self.limit:
                self.state["budget_exhausted"] = True
                continue
            cache = self.cache / rid
            client = NasaPowerClient(
                cache,
                max_requests=self.limit,
                timeout=45,
                retries=3,
                session=self.session,
            )
            client.counter_path = self.counter
            try:
                envelope = client.fetch(
                    lat, lon, "2019-12-01", "2025-12-31", parameters=VARIABLES
                )
                units = {
                    k: envelope["response"]["parameters"][k]["units"] for k in VARIABLES
                }
                if units != UNITS:
                    raise ValueError("Unidades meteorológicas inesperadas.")
                frame = power_frame(envelope)
                for name in ("WS10M", "PRECTOTCORR"):
                    frame.loc[frame[name] < 0, name] = np.nan
                frame.loc[~frame.RH2M.between(0, 100), "RH2M"] = np.nan
                frame.loc[~frame.GWETROOT.between(0, 1), "GWETROOT"] = np.nan
                frame.to_parquet(path, index=False)
                write_json(
                    meta,
                    {
                        "sha256": sha256_file(path),
                        "source": "NASA POWER Daily UTC",
                        "units": units,
                        "retrieved_utc": envelope["retrieved_utc"],
                        "climate_cell_id": rid,
                    },
                )
            except WeatherUnavailable as exc:
                self.state["failures"][rid] = type(exc).__name__
        write_json(self.state_path, self.state)
        return mapping

    def features(self, rid, dates):
        path = self.cells / (rid + ".parquet")
        target = pd.DatetimeIndex(dates) - pd.Timedelta(days=3)
        if not path.exists():
            frame = pd.DataFrame(
                {f"weather_{name}": np.full(len(target), np.nan) for name in VARIABLES}
            )
            frame["weather_last_date"] = target
            frame["climate_cell_id"] = rid
            return frame
        if sha256_file(path) != read_json(path.with_suffix(".json"))["sha256"]:
            raise ValueError("Clima alterado.")
        daily = pd.read_parquet(path).set_index("date")
        rolled = daily[list(VARIABLES)].rolling(8, min_periods=8).mean()
        rolled["PRECTOTCORR"] = daily.PRECTOTCORR.rolling(8, min_periods=8).sum()
        frame = (
            rolled.reindex(target)
            .reset_index(drop=True)
            .rename(columns={k: "weather_" + k for k in VARIABLES})
        )
        frame["weather_last_date"] = target
        frame["climate_cell_id"] = rid
        return frame
