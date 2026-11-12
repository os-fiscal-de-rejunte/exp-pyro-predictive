"""Recorte particionado do POWER Zarr; reutiliza blocos e associa a célula climática correta."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from numcodecs import blosc

from .ingest import region_id
from .io import read_json, sha256_file, write_json
from .power import grid_center
from .power_zarr import CachedZarrStore
from .runtime import Run

VARIABLES = ("T2M", "RH2M", "PRECTOTCORR", "WS10M", "GWETROOT")
UNITS = {
    "T2M": "C",
    "RH2M": "%",
    "PRECTOTCORR": "mm/day",
    "WS10M": "m/s",
    "GWETROOT": "1",
}


class ReusableStore(CachedZarrStore):
    def __getitem__(self, key):
        keypath = Path(key)
        if keypath.is_absolute() or ".." in keypath.parts or "\\" in key:
            raise KeyError("Chave Zarr inválida.")
        target = self.cache_dir / "objects" / key
        if not target.exists():
            for name in ("zarr_brasil_v2_20261111", "zarr_v2_control_20261111"):
                folder = self.run.root / "runs" / name / "weather_zarr/cache"
                source = folder / "objects" / key
                if source.exists() and (folder / "manifest.json").exists():
                    item = (
                        read_json(folder / "manifest.json").get("objects", {}).get(key)
                    )
                    if item and sha256_file(source) == item["sha256"]:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(source, target)
                        self.manifest["objects"][key] = {**item, "reused_from": name}
                        write_json(self.manifest_path, self.manifest)
                        break
        return super().__getitem__(key)


class WeatherTiles:
    def __init__(self, run: Run):
        self.run = run
        self.base = run.path / "weather"
        self.base.mkdir(exist_ok=True)
        self.store = ReusableStore(run, self.base / "cache")
        blosc.set_nthreads(1)
        self.group = zarr.open_consolidated(zarr.storage.KVStore(self.store), mode="r")
        self.lat = self.group["lat"][:]
        self.lon = self.group["lon"][:]
        attrs = dict(self.group["time"].attrs)
        if not attrs["units"].startswith("days since "):
            raise ValueError("Eixo temporal inesperado.")
        dates = pd.Timestamp(
            attrs["units"].removeprefix("days since ")
        ) + pd.to_timedelta(self.group["time"][:], unit="D")
        pos = np.flatnonzero(
            (dates >= pd.Timestamp("2019-12-01"))
            & (dates <= pd.Timestamp("2025-12-31"))
        )
        if len(pos) != 2223 or (np.diff(pos) != 1).any():
            expected = len(pd.date_range("2019-12-01", "2025-12-31"))
            if len(pos) != expected or (np.diff(pos) != 1).any():
                raise ValueError("Clima não cobre todos os dias solicitados.")
        self.first, self.after = int(pos[0]), int(pos[-1] + 1)
        self.dates = dates[pos]
        self.chunks = self.group["T2M"].chunks
        for name in VARIABLES:
            if (
                self.group[name].attrs.get("units") != UNITS[name]
                or self.group[name].chunks != self.chunks
            ):
                raise ValueError("Variável/unidade Zarr incompatível.")
        write_json(
            self.base / "source.json",
            {
                "source": self.store.base_url,
                "metadata_sha256": self.store.manifest["objects"][".zmetadata"][
                    "sha256"
                ],
                "units": UNITS,
                "start": "2019-12-01",
                "end": "2025-12-31",
                "chunks": list(self.chunks),
            },
        )

    def climate_key(self, latitude, longitude):
        lat, lon = grid_center(float(latitude), float(longitude))
        i, j = int(np.argmin(abs(self.lat - lat))), int(np.argmin(abs(self.lon - lon)))
        return region_id(lat, lon), i, j

    def prepare(self, cells: pd.DataFrame) -> dict[str, str]:
        groups = {}
        result = {}
        for row in cells.itertuples():
            rid, i, j = self.climate_key(row.latitude, row.longitude)
            result[row.region_id] = rid
            groups.setdefault((i // self.chunks[1], j // self.chunks[2]), {})[rid] = (
                i,
                j,
            )
        target_dir = self.base / "cells"
        target_dir.mkdir(exist_ok=True)
        for (ti, tj), requested in groups.items():
            missing = {}
            for rid, ij in requested.items():
                path = target_dir / (rid + ".parquet")
                meta = path.with_suffix(".json")
                if not path.exists() or not meta.exists():
                    missing[rid] = ij
                elif sha256_file(path) != read_json(meta)["sha256"]:
                    raise ValueError("Cache meteorológico alterado.")
            if not missing:
                continue
            self.run.guard(require_training_seconds=1200)
            i0, j0 = ti * self.chunks[1], tj * self.chunks[2]
            i1, j1 = (
                min(i0 + self.chunks[1], len(self.lat)),
                min(j0 + self.chunks[2], len(self.lon)),
            )
            records = {rid: {"date": self.dates} for rid in missing}
            for name in VARIABLES:
                cube = self.group[name][self.first : self.after, i0:i1, j0:j1]
                fill = self.group[name].fill_value
                for rid, (i, j) in missing.items():
                    values = cube[:, i - i0, j - j0].astype(float)
                    if fill is not None:
                        values[values == fill] = np.nan
                    values[values == -999] = np.nan
                    if name in ("WS10M", "PRECTOTCORR"):
                        values[values < 0] = np.nan
                    if name == "RH2M":
                        values[(values < 0) | (values > 100)] = np.nan
                    if name == "GWETROOT":
                        values[(values < 0) | (values > 1)] = np.nan
                    records[rid][name] = values
                del cube
            for rid, data in records.items():
                frame = pd.DataFrame(data)
                path = target_dir / (rid + ".parquet")
                frame.to_parquet(path, index=False)
                write_json(
                    path.with_suffix(".json"),
                    {
                        "sha256": sha256_file(path),
                        "climate_cell_id": rid,
                        "units": UNITS,
                    },
                )
            print(
                f"Clima: tile {ti},{tj}; {len(missing)} células meteorológicas preparadas",
                flush=True,
            )
        return result

    def features(self, climate_id, dates):
        path = self.base / "cells" / (climate_id + ".parquet")
        if sha256_file(path) != read_json(path.with_suffix(".json"))["sha256"]:
            raise ValueError("Clima alterado.")
        daily = pd.read_parquet(path).set_index("date")
        rolled = daily[list(VARIABLES)].rolling(8, min_periods=8).mean()
        rolled["PRECTOTCORR"] = daily.PRECTOTCORR.rolling(8, min_periods=8).sum()
        target = pd.DatetimeIndex(dates) - pd.Timedelta(days=3)
        values = (
            rolled.reindex(target)
            .reset_index(drop=True)
            .rename(columns={name: "weather_" + name for name in VARIABLES})
        )
        values["weather_last_date"] = target
        values["climate_cell_id"] = climate_id
        return values
