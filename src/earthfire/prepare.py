"""Seleção espacial, deduplicação exata e calendário de observações positivas."""
from __future__ import annotations

import math
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from .ingest import archive_product, csv_chunks, grid_indices, parse_datetime, region_id, validation_masks
from .io import sha256_file, write_json
from .runtime import Run


def _cell_indices(lat: float, lon: float) -> tuple[int, int]:
    a, b = grid_indices(pd.Series([lat]), pd.Series([lon]))
    return int(a.iloc[0]), int(b.iloc[0])


def prepare(run: Run) -> dict[str, Any]:
    output = run.path / "processed"
    output.mkdir(parents=True, exist_ok=True)
    run.stage("prepare", "running")
    wanted = {_cell_indices(x["latitude"], x["longitude"]): x for x in run.config["regions"]}
    countries = run.config.get("source_countries", ["Brazil"])
    if not countries or not all(isinstance(x, str) and x and "/" not in x and "\\" not in x for x in countries):
        raise ValueError("source_countries deve conter nomes de partições CSV válidos.")
    data = run.root / run.config["data_dir"]
    paths = sorted(data.glob("*.zip"), key=lambda x: (" (1)" in x.name, x.name))
    seen_hashes: set[str] = set()
    chunks: list[pd.DataFrame] = []
    counts: Counter[str] = Counter()
    inputs: list[dict[str, Any]] = []
    archive_inventory: list[dict[str, Any]] = []
    for path in paths:
        run.guard()
        digest = sha256_file(path)
        archive_inventory.append({"archive": path.name, "sha256": digest, "bytes": path.stat().st_size})
        if digest in seen_hashes:
            counts["duplicate_archives_skipped"] += 1
            continue
        seen_hashes.add(digest)
        product, year = archive_product(path)
        with zipfile.ZipFile(path) as z:
            members = [n for n in z.namelist() if any(n.endswith("_" + country + ".csv") for country in countries)]
        for member in members:
            source_rows = 0
            for raw in csv_chunks(path, member):
                run.guard()
                source_rows += len(raw)
                counts["rows_read_source_countries"] += len(raw)
                if countries == ["Brazil"]:
                    counts["rows_read_brazil"] += len(raw)
                required = {"latitude", "longitude", "acq_date", "acq_time", "satellite", "instrument"}
                if not required.issubset(raw.columns):
                    counts["rows_skipped_missing_required_columns"] += len(raw)
                    continue
                lat = pd.to_numeric(raw["latitude"], errors="coerce")
                lon = pd.to_numeric(raw["longitude"], errors="coerce")
                coords = lat.between(-90, 90) & lon.between(-180, 180)
                counts["invalid_coordinates_source_countries"] += int((~coords).sum())
                if countries == ["Brazil"]:
                    counts["invalid_coordinates_brazil"] += int((~coords).sum())
                candidate = raw.loc[coords].copy()
                ilat, ilon = grid_indices(lat[coords], lon[coords])
                candidate["_ilat"] = ilat
                candidate["_ilon"] = ilon
                candidate = candidate.loc[[pair in wanted for pair in zip(ilat, ilon)]].copy()
                if candidate.empty:
                    continue
                counts["selected_before_filter"] += len(candidate)
                _, acquired = parse_datetime(candidate)
                valid_time = acquired.notna()
                counts["selected_invalid_datetime_removed"] += int((~valid_time).sum())
                candidate = candidate.loc[valid_time].copy()
                if candidate.empty:
                    continue
                for key, mask in validation_masks(candidate, product).items():
                    counts["selected_" + key] += int(mask.sum())
                candidate["acquired_utc"] = acquired.loc[valid_time]
                candidate["region_id"] = [region_id(wanted[(i, j)]["latitude"], wanted[(i, j)]["longitude"])
                                          for i, j in zip(candidate.pop("_ilat"), candidate.pop("_ilon"))]
                candidate["product"] = product
                candidate["source_archive"] = path.name
                candidate["source_member"] = member
                candidate["source_row"] = candidate.index + 2
                candidate["source_sha256"] = digest
                # Colunas próprias de cada produto continuam distintas; não converter confiança.
                chunks.append(candidate)
            inputs.append({"archive": path.name, "member": member, "sha256": digest,
                           "nominal_year": year, "rows_read": source_rows})
    if not chunks:
        raise ValueError("Nenhum registro válido nas células configuradas.")
    records = pd.concat(chunks, ignore_index=True)
    # Todas as colunas originais disponíveis integram a identidade da linha.
    provenance = ["source_archive", "source_member", "source_row", "source_sha256", "acquired_utc", "region_id"]
    identity = [x for x in records.columns if x not in provenance]
    duplicates = records.duplicated(identity, keep="first")
    counts["exact_duplicates_removed"] = int(duplicates.sum())
    records = records.loc[~duplicates].copy()
    counts["records_retained"] = len(records)
    records.to_parquet(output / "records.parquet", index=False)
    records["week_start"] = records["acquired_utc"].dt.tz_localize(None).dt.to_period("W-SUN").dt.start_time
    weekly = (records.groupby(["region_id", "product", "satellite", "week_start"], dropna=False)
              .size().rename("detections").reset_index())
    weekly.to_parquet(output / "weekly_positive.parquet", index=False)
    calendar = make_calendar(weekly, run.config, inputs)
    calendar.to_parquet(output / "calendar.parquet", index=False)
    summary = {"inputs": inputs, "archive_inventory": archive_inventory, "filter_counts": dict(counts), "regions": run.config["regions"],
               "week_definition": "segunda-feira 00:00 UTC a segunda-feira seguinte 00:00 UTC",
               "source_scope": "arquivos anuais por país: " + ", ".join(countries) + "; polígonos completos não fornecidos",
               "absence": "missing_unknown; nenhum zero foi imputado", "aggregation_rows": len(weekly)}
    write_json(output / "preparation.json", summary)
    run.stage("prepare", "complete", records=len(records), weekly_rows=len(weekly), summary=str(output / "preparation.json"))
    return summary


def make_calendar(weekly: pd.DataFrame, config: dict[str, Any], inputs: list[dict[str, Any]]) -> pd.DataFrame:
    years = {(x["archive"].split("_")[0], x["nominal_year"]) for x in inputs}
    available_years = [x["nominal_year"] for x in inputs] or pd.to_datetime(weekly["week_start"]).dt.year.tolist()
    if not available_years:
        raise ValueError("Não há anos disponíveis para construir o calendário.")
    first_week = pd.Timestamp(year=min(available_years), month=1, day=1).to_period("W-SUN").start_time
    last_day = pd.Timestamp(year=max(available_years), month=12, day=31)
    products = sorted(set(x["archive"].split("_")[0] for x in inputs) or set(weekly["product"]))
    rows: list[dict[str, Any]] = []
    lookup = {(r.region_id, r.product, pd.Timestamp(r.week_start)): int(r.detections)
              for r in weekly.groupby(["region_id", "product", "week_start"], as_index=False)["detections"].sum().itertuples()}
    for region in config["regions"]:
        rid = region_id(region["latitude"], region["longitude"])
        for product in products:
            for week in pd.date_range(start=first_week, end=last_day, freq="W-MON"):
                key = (rid, product, week)
                value = lookup.get(key)
                week_years = {int((week + pd.Timedelta(days=d)).year) for d in range(7)}
                rows.append({"region_id": rid, "product": product, "week_start": week,
                             "detections": value, "observation_state": "positive" if value is not None else "missing_unknown",
                             "archive_present_for_all_calendar_years": all((product, y) in years for y in week_years)})
    result = pd.DataFrame(rows)
    result["detections"] = result["detections"].astype("Int64")
    # Referência puramente retrospectiva, mesmo mês e região/produto; outras semanas são desconhecidas.
    result["historical_reference"] = float("nan")
    result["observed_anomaly"] = float("nan")
    history: dict[tuple[str, str, int], list[int]] = {}
    for idx, row in result.sort_values("week_start").iterrows():
        if pd.isna(row.detections):
            continue
        key = (row.region_id, row["product"], row.week_start.month)
        past = history.setdefault(key, [])
        if len(past) >= 3:
            median = float(pd.Series(past).median())
            result.at[idx, "historical_reference"] = median
            result.at[idx, "observed_anomaly"] = int(row.detections) - median
        past.append(int(row.detections))
    return result
