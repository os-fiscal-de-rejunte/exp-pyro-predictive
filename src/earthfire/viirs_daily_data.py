"""Audita máscara VIIRS diária e constrói exemplos retrospectivos sem zeros inventados."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from netCDF4 import Dataset

from .io import read_json, sha256_file, write_json
from .mask_features import WEATHER, _inside_mask, weather_history
from .runtime import Run

VALID = np.array([3, 5, 7, 8, 9])
FIRE = np.array([7, 8, 9])


def read_mask(path: Path, root: Path, geometry: dict[str, Any], expected_date: str,
              neighbors: list[Path] | None = None) -> dict[str, Any]:
    nrows, ncols = int(geometry["nrows"]), int(geometry["ncols"])
    array = np.full((nrows, ncols), -1, dtype=np.int16)
    covered = np.zeros((nrows, ncols), dtype=bool)
    sources, productions = [], []
    invalid_pixels = 0
    first_row = first_col = None
    for part in [path, *(neighbors or [])]:
        if not part.resolve().is_relative_to(root.resolve()):
            raise ValueError("Grânulo fora do diretório autorizado.")
        # A biblioteca NetCDF no Windows pode falhar com caminho absoluto Unicode.
        try:
            filename = str(part.relative_to(Path.cwd().resolve()))
        except ValueError:
            filename = str(part)
        with Dataset(filename) as ds:
            if (getattr(ds, "ShortName", "") != "VJ114A1" or str(getattr(ds, "VersionID", "")) != "002"
                    or getattr(ds, "PlatformShortName", "") != "JPSS-1"
                    or getattr(ds, "RangeBeginningDate", "") != expected_date):
                raise ValueError(f"Produto, plataforma ou data inesperados: {part.name}")
            grid = ds.groups["HDFEOS"].groups["GRIDS"].groups["VIIRS_Grid_Daily_Fire"]
            fields = grid.groups["Data Fields"]
            projection = fields.variables["Projection"]
            if (getattr(projection, "grid_mapping_name", "") != "sinusoidal" or
                    abs(float(projection.earth_radius) - 6371007.181) > .01):
                raise ValueError("Projeção VIIRS incompatível com o recorte MODIS.")
            x, y = grid.variables["XDim"], grid.variables["YDim"]
            cell = float(geometry["cellsize_m"])
            if abs(float(x[1] - x[0]) - cell) > .02 or abs(float(y[0] - y[1]) - cell) > .02:
                raise ValueError("Tamanho de pixel VIIRS difere do recorte de referência.")
            left = (float(geometry["xllcorner"]) - float(x[0])) / cell
            top = (float(y[0]) - (float(geometry["yllcorner"]) + nrows * cell)) / cell
            row, col = round(top), round(left)
            if abs(top - row) > .02 or abs(left - col) > .02:
                raise ValueError("Máscara VIIRS não se alinha à geometria analisada.")
            if first_row is None:
                first_row, first_col = row, col
            sr0, sr1 = max(row, 0), min(row + nrows, len(y))
            sc0, sc1 = max(col, 0), min(col + ncols, len(x))
            if sr0 >= sr1 or sc0 >= sc1:
                raise ValueError("Grânulo informado não cruza a área de análise.")
            dr0, dc0 = sr0 - row, sc0 - col
            dr1, dc1 = dr0 + (sr1 - sr0), dc0 + (sc1 - sc0)
            if covered[dr0:dr1, dc0:dc1].any():
                raise ValueError("Grânulos VIIRS se sobrepõem na mesma área e data.")
            values = fields.variables["FireMask"][sr0:sr1, sc0:sc1]
            part_array = np.ma.filled(np.ma.asarray(values, dtype=np.int16), -1)
            if part_array.shape != (dr1 - dr0, dc1 - dc0):
                raise ValueError("Forma da máscara VIIRS inesperada.")
            invalid = (part_array < -1) | (part_array > 9)
            invalid_pixels += int(invalid.sum())
            part_array[invalid] = -1
            array[dr0:dr1, dc0:dc1] = part_array
            covered[dr0:dr1, dc0:dc1] = True
            sources.append({"file": part.name, "sha256": sha256_file(part)})
            productions.append(str(getattr(ds, "ProductionTime", "")))
    counts = {f"class_{i}": int(np.count_nonzero(array == i)) for i in range(10)}
    valid = int(np.isin(array, VALID).sum())
    fire = int(np.isin(array, FIRE).sum())
    total = int(array.size)
    return {**counts, "masked_pixels": int(np.count_nonzero(array == -1)), "invalid_pixels": invalid_pixels,
            "uncovered_pixels": int((~covered).sum()), "total_pixels": total,
            "valid_pixels": valid, "fire_pixels": fire, "cloud_pixels": counts["class_4"],
            "unknown_pixels": total - valid - counts["class_4"], "valid_fraction": valid / total,
            "fully_observed": valid == total, "event": int(fire > 0) if valid == total else None,
            "production_time": max(productions) if all(productions) else "", "source_file": path.name,
            "source_sha256": sources[0]["sha256"], "source_files": [x["file"] for x in sources],
            "source_hashes": [x["sha256"] for x in sources], "source_tiles_complete": bool(covered.all()),
            "grid_row": first_row, "grid_column": first_col}


def history_features(target: pd.Timestamp, viirs_dates: pd.DatetimeIndex,
                     modis: pd.DataFrame) -> dict[str, Any]:
    # A falta de registros FIRMS na janela não é uma contagem de zero confirmada.
    start, end = target - pd.Timedelta(days=9), target - pd.Timedelta(days=2)
    count = int(((viirs_dates >= start) & (viirs_dates <= end)).sum())
    candidates = modis.loc[(modis.period_end_exclusive <= target) &
                           (modis.period_end_exclusive >= target - pd.Timedelta(days=24)) &
                           (modis.produced_at < target)].sort_values("period_end_exclusive")
    previous = candidates.iloc[-1] if len(candidates) else None
    event = previous.event if previous is not None and pd.notna(previous.event) else None
    day = target.dayofyear
    return {"past_viirs_records": count if count else None,
            "past_viirs_log_records": math.log1p(count) if count else np.nan,
            "past_viirs_missing": int(count == 0), "viirs_history_start": start,
            "viirs_history_end": end, "past_modis_event": float(event) if event is not None else np.nan,
            "past_modis_missing": int(event is None),
            "past_modis_period_end": previous.period_end_exclusive if previous is not None else pd.NaT,
            "past_modis_produced_at": previous.produced_at if previous is not None else pd.NaT,
            "season_sin": math.sin(2 * math.pi * day / 365.2425),
            "season_cos": math.cos(2 * math.pi * day / 365.2425)}


def add_previous_daily_mask(frame: pd.DataFrame) -> pd.DataFrame:
    """Última máscara VIIRS amostrada disponível em até dez dias, sem inventar ausência."""
    result = frame.sort_values(["region_id", "period_start"]).copy()
    group = result.groupby("region_id", sort=False)
    prior_date = group.period_start.shift(1)
    prior_event = group.event.shift(1)
    prior_produced = pd.to_datetime(group.production_time.shift(1), errors="coerce")
    gap = (result.period_start - prior_date).dt.days
    available = gap.between(1, 10) & prior_produced.notna() & (prior_produced < result.period_start)
    result["past_viirs_mask_event"] = pd.to_numeric(prior_event, errors="coerce").where(available)
    result["past_viirs_mask_date"] = prior_date.where(available)
    result["past_viirs_mask_produced_at"] = prior_produced.where(available)
    return result


def prepare(run: Run) -> dict[str, Any]:
    base = run.path / "viirs_coverage"
    progress_path, manifest_path = base / "daily_progress.json", base / "manifest.json"
    progress, manifest = read_json(progress_path), read_json(manifest_path)
    if progress["config_sha256"] != run.config_hash:
        raise ValueError("A coleta foi feita com outra configuração.")
    out = run.path / "processed"
    summary_path = out / "viirs_daily_summary.json"
    inputs = {"progress": sha256_file(progress_path), "manifest": sha256_file(manifest_path),
              "code": sha256_file(Path(__file__))}
    halo_path = base / "halo_progress.json"
    halo = read_json(halo_path) if halo_path.exists() else {"partitions": {}}
    if halo_path.exists():
        if halo["config_sha256"] != run.config_hash:
            raise ValueError("A coleta de borda foi feita com outra configuração.")
        inputs["halo_progress"] = sha256_file(halo_path)
    extra_path = base / "extra_validation_progress.json"
    extra = read_json(extra_path) if extra_path.exists() else {"partitions": {}}
    if extra_path.exists():
        if extra["config_sha256"] != run.config_hash:
            raise ValueError("A validação complementar tem outra configuração.")
        inputs["extra_validation_progress"] = sha256_file(extra_path)
    old = run.root / "runs/nasa_complement_v3b_20261111"
    mask_path = old / "coverage/mask_summary.parquet"
    raw_path = run.root / "runs/sat_zarr_v2_20261111/processed/records.parquet"
    extra_path = old / "processed/viirs_2023_selected.parquet"
    weather_path = run.root / "runs/zarr_brasil_v2_20261111/weather_zarr/daily.parquet"
    inputs.update({"modis_masks": sha256_file(mask_path), "satellite_records": sha256_file(raw_path),
                   "viirs_2023": sha256_file(extra_path), "weather": sha256_file(weather_path)})
    if summary_path.exists():
        previous = read_json(summary_path)
        if previous["input_sha256"] != inputs or sha256_file(out / "mask_examples.parquet") != previous["examples_sha256"]:
            raise ValueError("Checkpoint de preparação incompatível; use outro run_id.")
        return previous
    run.stage("viirs_daily_prepare", "running")
    masks = pd.read_parquet(mask_path)
    geometry_cols = ["xllcorner", "yllcorner", "ncols", "nrows", "cellsize_m"]
    geometries = {}
    for name, group in masks.groupby("region_name"):
        if any(group[column].nunique() != 1 for column in geometry_cols):
            raise ValueError(f"Geometria variável para {name}.")
        geometries[name] = group.iloc[0][geometry_cols + ["region_id", "latitude", "longitude", "spatial_group"]].to_dict()
    records: list[dict[str, Any]] = []
    halo_index: dict[tuple[str, str], list[Path]] = {}
    for key, partition in halo["partitions"].items():
        region_name, _ = key.split("/", 1)
        for filename in partition["filenames"]:
            item = manifest["files"].get(filename)
            path = base / filename
            if not item or sha256_file(path) != item["sha256"]:
                raise ValueError("Grânulo de borda ausente ou alterado.")
            halo_index.setdefault((region_name, item["start_utc"][:10]), []).append(path)
    for source in (progress, extra):
        for key, partition in source["partitions"].items():
            region_name, _ = key.split("/", 1)
            geometry = geometries[region_name]
            for filename in partition["filenames"]:
                run.guard()
                item = manifest["files"].get(filename)
                path = base / filename
                if not item or sha256_file(path) != item["sha256"]:
                    raise ValueError("Grânulo ausente ou alterado.")
                target_day = item["start_utc"][:10]
                attrs = read_mask(path, run.root, geometry, target_day, halo_index.get((region_name, target_day)))
                records.append({"region_name": region_name, **geometry, "period_start": pd.Timestamp(target_day),
                                "period_end_exclusive": pd.Timestamp(target_day) + pd.Timedelta(days=1), **attrs})
    frame = pd.DataFrame(records)
    if frame.empty or frame.duplicated(["region_id", "period_start"]).any():
        raise ValueError("Não há máscaras únicas suficientes para preparar.")
    frame["event"] = pd.array(frame.event, dtype="Int64")
    frame = add_previous_daily_mask(frame)
    raw = pd.concat([pd.read_parquet(raw_path), pd.read_parquet(extra_path)], ignore_index=True)
    raw = raw.loc[(raw["product"] == "viirs-jpss1") & (raw.satellite == "N20")].copy()
    raw["acquired_utc"] = pd.to_datetime(raw.acquired_utc, utc=True)
    masks["period_end_exclusive"] = pd.to_datetime(masks.period_end_exclusive)
    masks["produced_at"] = pd.to_datetime(masks.proc_date.astype("string"), format="%Y%j%H%M%S", errors="coerce")
    history_rows = []
    for name, group in frame.groupby("region_name", sort=False):
        geometry = geometries[name]
        viirs = raw.loc[raw.region_id == geometry["region_id"]]
        viirs = viirs.loc[_inside_mask(viirs, geometry)]
        dates = pd.DatetimeIndex(viirs.acquired_utc.dt.tz_localize(None).dt.normalize())
        previous_modis = masks.loc[masks.region_id == geometry["region_id"]]
        for item in group.itertuples():
            history_rows.append({"region_id": item.region_id, "period_start": item.period_start,
                                 **history_features(item.period_start, dates, previous_modis)})
    frame = frame.merge(pd.DataFrame(history_rows), on=["region_id", "period_start"], validate="one_to_one")
    weather = weather_history(pd.read_parquet(weather_path), frame)
    frame = frame.merge(weather, on=["region_id", "period_start"], validate="one_to_one")
    frame["split"] = "excluded"
    frame.loc[frame.period_start <= pd.Timestamp("2022-12-31"), "split"] = "train"
    frame.loc[frame.period_start.between("2023-01-01", "2023-12-31"), "split"] = "validation"
    frame.loc[frame.period_start.between("2024-01-01", "2025-12-31"), "split"] = "test"
    frame.loc[frame.event.isna(), "split"] = "unknown_coverage"
    out.mkdir(parents=True, exist_ok=True)
    examples = out / "mask_examples.parquet"
    frame.to_parquet(examples, index=False)
    summary = {"product": "VJ114A1.002", "input_sha256": inputs, "examples_sha256": sha256_file(examples),
        "rows": len(frame), "unknown_coverage": int(frame.event.isna().sum()),
        "split_counts": {f"{a}/{b}": int(n) for (a, b), n in frame.groupby(["split", "spatial_group"]).size().items()},
        "events": {f"{a}/{b}": {str(k): int(v) for k, v in g.event.value_counts().items()}
                   for (a, b), g in frame.groupby(["split", "spatial_group"])},
        "missing_days_catalog_or_download": sum(len(x["missing_days"]) for s in (progress, extra) for x in s["partitions"].values()),
        "ambiguous_days": sum(len(x["ambiguous_days"]) for s in (progress, extra) for x in s["partitions"].values()),
        "weather_fields": list(WEATHER), "limitation": "Evento VIIRS de 1 km condicionado à máscara diária integralmente válida; não é o produto FIRMS de 375 m."}
    write_json(summary_path, summary)
    run.stage("viirs_daily_prepare", "complete", rows=len(frame), unknown=summary["unknown_coverage"])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/viirs_daily_v4b.json"))
    parser.add_argument("--run-id", default="viirs_daily_v4b_20261111")
    args = parser.parse_args()
    print(json.dumps(prepare(Run(args.config, args.run_id)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
