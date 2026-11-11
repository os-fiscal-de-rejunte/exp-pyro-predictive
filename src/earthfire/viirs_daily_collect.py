"""Coleta particionada de dias pré-definidos do VIIRS diário, com retomada."""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .io import read_json, sha256_file, utc_now, write_json
from .runtime import Run
from .viirs_coverage import download_granules, query_granules


def collect(run: Run, token: str, regions: set[str] | None = None,
            start_year: int | None = None, end_year: int | None = None,
            months: set[int] | None = None) -> dict[str, Any]:
    cfg = run.config["viirs_daily"]
    first = max(cfg["start_year"], start_year or cfg["start_year"])
    last = min(cfg["end_year"], end_year or cfg["end_year"])
    if not token or first > last or (months and not months <= set(range(1, 13))):
        raise ValueError("Credencial, anos ou meses inválidos.")
    requested_regions = [r for r in run.config["regions"] if not regions or r["name"] in regions]
    if not requested_regions or (regions and set(r["name"] for r in requested_regions) != regions):
        raise ValueError("Região fora da configuração de coleta.")
    ref = pd.read_parquet(run.root / "runs/nasa_complement_v3b_20261111/coverage/mask_summary.parquet")
    tiles = {name: frame.tile.unique().tolist() for name, frame in ref.groupby("region_name")}
    if any(len(tiles.get(region["name"], [])) != 1 for region in requested_regions):
        raise ValueError("Não há um tile MODIS único para uma das áreas solicitadas.")
    base = run.path / "viirs_coverage"
    base.mkdir(parents=True, exist_ok=True)
    progress_path = base / "daily_progress.json"
    progress = read_json(progress_path) if progress_path.exists() else {
        "schema_version": "1.0", "product": "VJ114A1.002", "config_sha256": run.config_hash,
        "days_of_month": cfg["days_of_month"], "partitions": {}, "created_utc": utc_now()}
    if progress["config_sha256"] != run.config_hash:
        raise ValueError("Configuração diferente da coleta já iniciada.")
    manifest_path = base / "manifest.json"
    if manifest_path.exists():
        for filename, item in read_json(manifest_path)["files"].items():
            path = base / filename
            if not path.exists() or sha256_file(path) != item["sha256"]:
                raise ValueError("Grânulo já coletado ausente ou incompatível com o manifesto.")
    run.stage("viirs_daily_collect", "running", partitions_complete=len(progress["partitions"]))
    session = requests.Session()
    try:
        for region in requested_regions:
            name, tile = region["name"], tiles[region["name"]][0]
            half = cfg["bbox_half_width_degrees"]
            lat, lon = region["latitude"], region["longitude"]
            bbox = (lon - half, lat - half, lon + half, lat + half)
            for year in range(first, last + 1):
                for month in sorted(months or set(range(1, 13))):
                    key = f"{name}/{year:04d}-{month:02d}"
                    if key in progress["partitions"]:
                        continue
                    run.guard()
                    start, end = date(year, month, 1), date(year, month, max(cfg["days_of_month"]))
                    time.sleep(1.)
                    found = query_granules(run, bbox, start.isoformat(), end.isoformat(), limit=30, session=session)
                    if found["truncated"]:
                        raise ValueError(f"Catálogo truncou a partição {key}; subdivida a consulta.")
                    wanted = {date(year, month, day).isoformat() for day in cfg["days_of_month"]}
                    matching = [item for item in found["granules"]
                                if f".{tile}." in item["title"] and item["start_utc"][:10] in wanted]
                    by_date: dict[str, list[dict[str, Any]]] = {}
                    for item in matching:
                        by_date.setdefault(item["start_utc"][:10], []).append(item)
                    duplicates = [day for day, items in by_date.items() if len(items) != 1]
                    eligible = [items[0] for day, items in by_date.items() if day not in duplicates]
                    if eligible:
                        time.sleep(1.)
                        subset = {**found, "granules": eligible}
                        result = download_granules(run, subset, token, session=session)
                        if result["status"] != "ok":
                            raise ConnectionError(f"Coleta da partição {key}: {result['status']}")
                    missing = sorted(wanted - set(by_date))
                    progress["partitions"][key] = {"tile": tile, "expected_days": sorted(wanted),
                        "filenames": [x["filename"] for x in eligible], "missing_days": missing,
                        "ambiguous_days": sorted(duplicates), "completed_utc": utc_now(),
                        "cmr_response_sha256": found["cmr_response_sha256"]}
                    write_json(progress_path, progress)
                    complete = len(progress["partitions"])
                    if complete % 12 == 0:
                        run.stage("viirs_daily_collect", "running", partitions_complete=complete, latest=key)
                        print(f"Partições concluídas: {complete}; última: {key}", flush=True)
    except BaseException:
        run.stage("viirs_daily_collect", "interrupted", partitions_complete=len(progress["partitions"]))
        raise
    finally:
        session.close()
    parts = progress["partitions"].values()
    result = {"status": "ok", "partitions": len(progress["partitions"]),
              "files": sum(len(item["filenames"]) for item in parts),
              "missing_days": sum(len(item["missing_days"]) for item in parts),
              "ambiguous_days": sum(len(item["ambiguous_days"]) for item in parts),
              "progress": str(progress_path)}
    run.stage("viirs_daily_collect", "complete", **result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Coleta uma amostra diária sistemática de VJ114A1")
    parser.add_argument("--config", type=Path, default=Path("config/viirs_daily_v4.json"))
    parser.add_argument("--run-id", default="viirs_daily_v4_20261111")
    parser.add_argument("--region", action="append")
    parser.add_argument("--start-year", type=int)
    parser.add_argument("--end-year", type=int)
    parser.add_argument("--month", type=int, action="append")
    args = parser.parse_args()
    result = collect(Run(args.config, args.run_id), os.environ.get("EARTHDATA_TOKEN", ""),
                     set(args.region) if args.region else None, args.start_year, args.end_year,
                     set(args.month) if args.month else None)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
