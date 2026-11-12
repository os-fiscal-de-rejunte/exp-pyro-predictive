"""Features por partição; alvos incompletos permanecem ausentes."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from .io import read_json, sha256_file, write_json
from .runtime import ResourceLimit, Run


class WeatherProvider(Protocol):
    base: Path

    def prepare(self, cells: pd.DataFrame) -> dict[str, str]: ...
    def features(self, climate_id: str, dates) -> pd.DataFrame: ...
    def climate_key(self, latitude: float, longitude: float): ...


def prepare_parent(
    run: Run, name: str, weather: WeatherProvider, seen: set[str]
) -> dict:
    out = run.path / "features"
    out.mkdir(exist_ok=True)
    target = out / (name + ".parquet")
    mp = out / (name + ".json")
    if mp.exists():
        result = read_json(mp)
        if sha256_file(target) != result["sha256"]:
            raise ValueError("Features alteradas.")
        seen.update(result["cell_ids"])
        return result
    assets = run.root / "runs" / run.config.get("mask_storage_run", run.run_id)
    manifest = read_json(assets / "mask_manifest.json")["partitions"][name]
    source = assets / "blocks" / (name + ".parquet")
    if sha256_file(source) != manifest["sha256"]:
        raise ValueError("Máscaras de entrada alteradas.")
    frame = pd.read_parquet(source)
    frame = frame.loc[frame.train_support & ~frame.region_id.isin(seen)].copy()
    if frame.empty:
        return {"parent": name, "rows": 0, "cells": 0, "provisional": 0, "cell_ids": []}
    for col in ("period_start",):
        frame[col] = pd.to_datetime(frame[col])
    frame["period_end_exclusive"] = frame.period_start + pd.Timedelta(days=8)
    years = pd.to_datetime((frame.period_start.dt.year + 1).astype(str) + "-01-01")
    frame["period_end_exclusive"] = frame.period_end_exclusive.where(
        frame.period_end_exclusive < years, years
    )
    frame["produced_at"] = pd.to_datetime(
        frame.proc_date, format="%Y%j%H%M%S", errors="coerce"
    )
    unique = frame.groupby("region_id", sort=True).first().reset_index()
    limited = False
    try:
        climate_map = weather.prepare(unique)
    except ResourceLimit:
        limited = True
        climate_map = {}
        for row in unique.itertuples():
            climate_map[row.region_id] = weather.climate_key(
                row.latitude, row.longitude
            )[0]
    parts = []
    provisional = 0
    for rid, one in frame.groupby("region_id", sort=True):
        run.guard(require_training_seconds=1200)
        clim = climate_map[rid]
        one = one.sort_values("period_start").reset_index(drop=True)
        hist = []
        for row in one.itertuples():
            past = one.loc[
                (one.period_end_exclusive <= row.period_start)
                & (one.produced_at < row.period_start)
                & (one.period_start >= row.period_start - pd.Timedelta(days=40))
            ]
            previous = past.iloc[-1] if len(past) else None
            event = previous.event if previous is not None else pd.NA
            hist.append(
                {
                    "past_modis_event": float(event) if pd.notna(event) else np.nan,
                    "past_modis_missing": int(pd.isna(event)),
                    "history_last_produced_at": previous.produced_at
                    if previous is not None
                    else pd.NaT,
                }
            )
        one = pd.concat(
            [one, pd.DataFrame(hist), weather.features(clim, one.period_start)], axis=1
        )
        # VIIRS permanece fora dos regressores novos devido à lacuna 2023; sem inventar zeros.
        one["past_viirs_log_records"] = np.nan
        one["past_viirs_missing"] = 1
        one["season_sin"] = np.sin(2 * np.pi * one.period_start.dt.dayofyear / 365.2425)
        one["season_cos"] = np.cos(2 * np.pi * one.period_start.dt.dayofyear / 365.2425)
        one["split"] = "excluded"
        one.loc[one.period_end_exclusive <= pd.Timestamp("2023-01-01"), "split"] = (
            "train"
        )
        one.loc[
            (one.period_start >= pd.Timestamp("2023-01-01"))
            & (one.period_end_exclusive <= pd.Timestamp("2024-01-01")),
            "split",
        ] = "validation"
        one.loc[
            (one.period_start >= pd.Timestamp("2024-01-01"))
            & (one.period_end_exclusive <= pd.Timestamp("2026-01-01")),
            "split",
        ] = "test"
        one["observation_state"] = np.where(
            one.event.isna(),
            np.where(one.cloud_pixels > 0, "cloud_obscured", "missing_unknown"),
            np.where(
                (one.event == 1).fillna(False),
                "positive_detection",
                "observed_no_detection",
            ),
        )
        finite = np.isfinite(
            one[
                [
                    "weather_" + v
                    for v in ("T2M", "RH2M", "PRECTOTCORR", "WS10M", "GWETROOT")
                ]
            ].to_numpy(float)
        ).all(axis=1)
        one["weather_complete"] = finite
        tr = one.loc[(one.split == "train") & one.event.notna() & finite]
        val = one.loc[(one.split == "validation") & one.event.notna() & finite]
        if (
            one.iloc[0].spatial_group == "train"
            and len(val) >= 30
            and 5 <= val.event.sum() <= len(val) - 5
            and len(tr)
        ):
            rate = (float(tr.event.sum()) + 0.5) / (len(tr) + 1)
            monthly = {
                int(m): (float(g.event.sum()) + 2 * rate) / (len(g) + 2)
                for m, g in tr.groupby(tr.period_start.dt.month)
            }
            p = val.period_start.dt.month.map(monthly).fillna(rate).to_numpy()
            if abs(p.mean() - val.event.astype(float).mean()) <= 0.1:
                provisional += 1
        parts.append(one)
    if not parts:
        return {
            "parent": name,
            "rows": 0,
            "cells": 0,
            "provisional": 0,
            "cell_ids": [],
            "weather_limit": limited,
        }
    result_frame = pd.concat(parts, ignore_index=True).sort_values(
        ["region_id", "period_start"]
    )
    result_frame.to_parquet(target, index=False, row_group_size=276)
    ids = sorted(result_frame.region_id.unique())
    seen.update(ids)
    result = {
        "parent": name,
        "rows": len(result_frame),
        "cells": len(ids),
        "provisional": provisional,
        "cell_ids": ids,
        "sha256": sha256_file(target),
        "source_sha256": manifest["sha256"],
        "code_sha256": sha256_file(Path(__file__)),
        "weather_limit": limited,
    }
    write_json(mp, result)
    print(
        f"Features {name}: {len(ids)} células; {provisional} candidatas sazonais (provisório)",
        flush=True,
    )
    return result
