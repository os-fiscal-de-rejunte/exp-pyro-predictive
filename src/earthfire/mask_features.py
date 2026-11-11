"""Exemplos de oito dias para máscara MODIS e histórico VIIRS, sem fabricar zeros."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .ingest import grid_indices, parse_datetime, region_id, validation_masks
from .io import read_json, sha256_file, write_json
from .runtime import Run

EARTH_RADIUS_M = 6371007.181
WEATHER = ("T2M", "RH2M", "PRECTOTCORR", "WS10M", "GWETROOT")
FEATURES_BASE = ("latitude", "longitude", "season_sin", "season_cos", "past_modis_event", "past_modis_missing",
                 "past_viirs_log_records", "past_viirs_missing")
FEATURES_WEATHER = tuple("weather_" + name for name in WEATHER)


def period_start(values: pd.Series) -> pd.Series:
    """Calendário MOD14A2 reinicia no primeiro dia de cada ano."""
    dates = pd.to_datetime(values).dt.tz_localize(None).dt.normalize()
    first = pd.to_datetime(dates.dt.year.astype(str) + "-01-01")
    offset = ((dates.dt.dayofyear - 1) // 8) * 8
    return first + pd.to_timedelta(offset, unit="D")


def select_2023(run: Run) -> tuple[pd.DataFrame, dict[str, Any]]:
    source_run = run.root / "runs" / run.config.get("firms_run", run.run_id)
    source = source_run / "firms_raw/viirs_jpss1_2023_Brazil.csv"
    digest = sha256_file(source)
    manifest = read_json(source_run / "firms_raw/manifest.json")
    if manifest["files"][source.name]["sha256"] != digest:
        raise ValueError("CSV VIIRS 2023 incompatível com o manifesto.")
    wanted = {}
    for item in run.config["regions"]:
        ilat, ilon = grid_indices(pd.Series([item["latitude"]]), pd.Series([item["longitude"]]))
        wanted[(int(ilat.iloc[0]), int(ilon.iloc[0]))] = region_id(item["latitude"], item["longitude"])
    chunks: list[pd.DataFrame] = []
    totals = {"rows_read": 0, "invalid_coordinates": 0, "selected": 0,
              "selected_invalid_datetime": 0, "selected_invalid_confidence": 0}
    use = ["latitude", "longitude", "acq_date", "acq_time", "satellite", "instrument", "confidence", "version"]
    for frame in pd.read_csv(source, usecols=use, dtype=str, chunksize=50000, keep_default_na=False):
        run.guard()
        totals["rows_read"] += len(frame)
        latitude = pd.to_numeric(frame.latitude, errors="coerce")
        longitude = pd.to_numeric(frame.longitude, errors="coerce")
        good = latitude.between(-90, 90) & longitude.between(-180, 180)
        totals["invalid_coordinates"] += int((~good).sum())
        candidates = frame.loc[good].copy()
        ilat, ilon = grid_indices(latitude[good], longitude[good])
        keys = list(zip(ilat, ilon))
        keep = np.fromiter((x in wanted for x in keys), dtype=bool, count=len(keys))
        candidates = candidates.loc[keep].copy()
        if candidates.empty:
            continue
        candidates["region_id"] = [wanted[x] for x, yes in zip(keys, keep) if yes]
        _, acquired = parse_datetime(candidates)
        masks = validation_masks(candidates, "viirs-jpss1")
        totals["selected_invalid_datetime"] += int(acquired.isna().sum())
        totals["selected_invalid_confidence"] += int(masks["confidence_invalid"].sum())
        candidates["acquired_utc"] = acquired
        candidates["source_row"] = candidates.index + 2
        candidates["product"] = "viirs-jpss1"
        candidates["source_archive"] = source.name
        candidates["source_sha256"] = digest
        chunks.append(candidates.loc[acquired.notna() & (candidates.satellite == "N20")])
    selected = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    totals["selected"] = len(selected)
    totals["source_sha256"] = digest
    return selected, totals


def _inside_mask(frame: pd.DataFrame, geometry: dict[str, float]) -> pd.Series:
    latitude = pd.to_numeric(frame.latitude, errors="coerce")
    longitude = pd.to_numeric(frame.longitude, errors="coerce")
    y = EARTH_RADIUS_M * np.deg2rad(latitude)
    x = EARTH_RADIUS_M * np.deg2rad(longitude) * np.cos(np.deg2rad(latitude))
    x0, y0 = geometry["xllcorner"], geometry["yllcorner"]
    return x.between(x0, x0 + geometry["ncols"] * geometry["cellsize_m"], inclusive="left") & y.between(
        y0, y0 + geometry["nrows"] * geometry["cellsize_m"], inclusive="left")


def build_viirs_counts(records: pd.DataFrame, masks: pd.DataFrame) -> pd.DataFrame:
    viirs = records.loc[(records["product"] == "viirs-jpss1") & (records.satellite == "N20")].copy()
    parts = []
    for rid, geometry_rows in masks.groupby("region_id"):
        geometry = geometry_rows.iloc[0][["xllcorner", "yllcorner", "nrows", "ncols", "cellsize_m"]].to_dict()
        if any(geometry_rows[key].nunique() != 1 for key in geometry):
            raise ValueError(f"Geometria ORNL mudou entre períodos: {rid}")
        selected = viirs.loc[viirs.region_id == rid].copy()
        selected = selected.loc[_inside_mask(selected, geometry)].copy()
        if selected.empty:
            continue
        selected["period_start"] = period_start(selected.acquired_utc)
        # A contagem é apenas das linhas positivas encontradas no arquivo; lacunas não viram zero.
        grouped = selected.groupby(["region_id", "period_start"]).size().rename("records").reset_index()
        parts.append(grouped)
    if not parts:
        return pd.DataFrame(columns=["region_id", "period_start", "records"])
    return pd.concat(parts, ignore_index=True)


def weather_history(weather: pd.DataFrame, masks: pd.DataFrame) -> pd.DataFrame:
    daily = weather.copy()
    daily["date"] = pd.to_datetime(daily.date)
    result: list[dict[str, Any]] = []
    for rid, periods in masks.groupby("region_id"):
        one = daily.loc[daily.region_id == rid].set_index("date").sort_index()
        if one.index.has_duplicates:
            raise ValueError(f"Meteorologia diária duplicada: {rid}")
        if not one.empty:
            one = one.reindex(pd.date_range(one.index.min(), one.index.max(), freq="D"))
        rolls = one[list(WEATHER)].rolling(8, min_periods=8).mean()
        rolls["PRECTOTCORR"] = one.PRECTOTCORR.rolling(8, min_periods=8).sum()
        for row in periods.itertuples():
            origin = pd.Timestamp(row.period_start)
            last_weather = origin - pd.Timedelta(days=3)
            item = {"region_id": rid, "period_start": origin, "weather_last_date": last_weather}
            for name in WEATHER:
                item["weather_" + name] = float(rolls.at[last_weather, name]) if last_weather in rolls.index else np.nan
            result.append(item)
    return pd.DataFrame(result)


def build_examples(masks: pd.DataFrame, viirs_counts: pd.DataFrame, weather: pd.DataFrame,
                   config: dict[str, Any]) -> pd.DataFrame:
    periods = masks.copy()
    periods["period_start"] = pd.to_datetime(periods.period_start)
    periods["period_end_exclusive"] = pd.to_datetime(periods.period_end_exclusive)
    periods = periods.sort_values(["region_id", "period_start"])
    groups = periods.groupby("region_id", sort=False)
    for column in ("period_start", "period_end_exclusive", "event", "proc_date"):
        periods["past_" + column] = groups[column].shift(2)
    gap = (periods.period_start - periods.past_period_end_exclusive).dt.days
    periods["past_gap_days"] = gap
    expected_previous = period_start(periods.period_start - pd.Timedelta(days=1))
    expected_past = period_start(expected_previous - pd.Timedelta(days=1))
    periods["past_period_aligned"] = periods.past_period_start == expected_past
    # Datas de produção posteriores à emissão não podem ser usadas como histórico.
    produced = pd.to_datetime(periods.past_proc_date.astype("string"), format="%Y%j%H%M%S", errors="coerce")
    available = periods.past_period_aligned & gap.between(1, 16) & produced.notna() & (produced < periods.period_start)
    periods["past_modis_event"] = pd.to_numeric(periods.past_event, errors="coerce").where(available)
    periods["past_modis_missing"] = periods.past_modis_event.isna().astype(int)
    counts = viirs_counts.rename(columns={"period_start": "past_period_start", "records": "past_viirs_records"})
    periods = periods.merge(counts, on=["region_id", "past_period_start"], how="left", validate="many_to_one")
    periods["past_viirs_records"] = periods.past_viirs_records.where(
        periods.past_period_aligned & periods.past_gap_days.between(1, 16))
    periods["past_viirs_records"] = pd.to_numeric(periods.past_viirs_records, errors="coerce")
    periods["past_viirs_log_records"] = np.log1p(periods.past_viirs_records)
    periods["past_viirs_missing"] = periods.past_viirs_records.isna().astype(int)
    day = periods.period_start.dt.dayofyear.to_numpy()
    periods["season_sin"] = np.sin(2 * np.pi * day / 365.2425)
    periods["season_cos"] = np.cos(2 * np.pi * day / 365.2425)
    weather_features = weather_history(weather, periods)
    periods = periods.merge(weather_features, on=["region_id", "period_start"], how="left", validate="one_to_one")
    train_end = pd.Timestamp(config["split"]["train_end"])
    validation_end = pd.Timestamp(config["split"]["validation_end"])
    test_start = pd.Timestamp(config["split"]["test_start"])
    test_end = pd.Timestamp(config["split"]["test_end"])
    last_day = periods.period_end_exclusive - pd.Timedelta(days=1)
    periods["split"] = "excluded"
    periods.loc[last_day <= train_end, "split"] = "train"
    periods.loc[(periods.period_start > train_end) & (last_day <= validation_end), "split"] = "validation"
    periods.loc[(periods.period_start >= test_start) & (last_day <= test_end), "split"] = "test"
    periods.loc[periods.event.isna(), "split"] = "unknown_coverage"
    return periods


def prepare(run: Run) -> dict[str, Any]:
    mask_path = run.path / "coverage/mask_summary.parquet"
    weather_path = run.root / "runs" / run.config["weather_run"] / "weather_zarr/daily.parquet"
    original_path = run.root / "runs" / run.config["satellite_run"] / "processed/records.parquet"
    source_run = run.root / "runs" / run.config.get("firms_run", run.run_id)
    extra_path = source_run / "firms_raw/viirs_jpss1_2023_Brazil.csv"
    output = run.path / "processed"
    saved_path = output / "summary.json"
    if saved_path.exists():
        prior = read_json(saved_path)
        checks = [(mask_path, prior["mask_sha256"]), (weather_path, prior["weather_sha256"]),
                  (original_path, prior["original_records_sha256"]), (extra_path, prior["audit_2023"]["source_sha256"]),
                  (output / "mask_examples.parquet", prior["examples_sha256"]), (Path(__file__), prior["features_code_sha256"])]
        if prior["config_sha256"] != run.config_hash or any(not p.exists() or sha256_file(p) != digest for p,digest in checks):
            raise ValueError("Checkpoint das features incompatível com os insumos ou código.")
        return prior
    run.stage("mask_features", "running")
    masks = pd.read_parquet(mask_path)
    weather = pd.read_parquet(weather_path)
    original = pd.read_parquet(original_path)
    new, audit = select_2023(run)
    output.mkdir(parents=True, exist_ok=True)
    new.to_parquet(output / "viirs_2023_selected.parquet", index=False)
    combined = pd.concat([original, new], ignore_index=True)
    counts = build_viirs_counts(combined, masks)
    counts.to_parquet(output / "viirs_positive_8day.parquet", index=False)
    examples = build_examples(masks, counts, weather, run.config)
    examples.to_parquet(output / "mask_examples.parquet", index=False)
    summary = {"audit_2023": audit, "config_sha256": run.config_hash,
        "features_code_sha256": sha256_file(Path(__file__)),
        "mask_sha256": sha256_file(mask_path), "weather_sha256": sha256_file(weather_path),
        "original_records_sha256": sha256_file(original_path),
        "examples_sha256": sha256_file(output / "mask_examples.parquet"),
        "split_counts": {f"{a}/{b}": int(n) for (a,b), n in examples.groupby(["split", "spatial_group"]).size().items()},
        "event_by_split": {f"{a}/{b}": {str(k): int(v) for k,v in g.event.value_counts().items()}
                           for (a,b),g in examples.groupby(["split", "spatial_group"])},
        "interpretation": "Evento MOD14A2 condicionado a máscara integralmente válida; não é cobertura VIIRS."}
    write_json(output / "summary.json", summary)
    run.stage("mask_features", "complete", examples=len(examples), split_counts=summary["split_counts"])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/nasa_complement_v3.json"))
    parser.add_argument("--run-id", default="nasa_complement_v3_20261111")
    args = parser.parse_args()
    run = Run(args.config, args.run_id)
    print(prepare(run))


if __name__ == "__main__":
    main()
