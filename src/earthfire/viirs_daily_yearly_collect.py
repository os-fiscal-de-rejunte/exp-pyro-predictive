"""Coleta NOAA-20 por ano/área, em dias sazonais fixados previamente."""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .io import read_json, sha256_file, utc_now, write_json
from .runtime import Run
from .viirs_coverage import download_with_gaps, query_granules


def collect(run: Run, token: str, regions: set[str] | None = None,
            years: set[int] | None = None) -> dict[str, Any]:
    cfg = run.config["viirs_daily"]
    if not token:
        raise ValueError("Token Earthdata não fornecido no ambiente.")
    regions_config = [x for x in run.config["regions"] if regions is None or x["name"] in regions]
    years_config = [x for x in cfg["year_order"] if years is None or x in years]
    if not regions_config or not years_config or (regions and {x["name"] for x in regions_config} != regions):
        raise ValueError("Regiões ou anos fora da configuração.")
    if years and set(years_config) != years:
        raise ValueError("Ano fora da configuração.")
    reference = pd.read_parquet(run.root / "runs/nasa_complement_v3b_20261111/coverage/mask_summary.parquet")
    tiles = {name: frame.tile.unique().tolist() for name, frame in reference.groupby("region_name")}
    if any(len(tiles.get(region["name"], [])) != 1 for region in regions_config):
        raise ValueError("Tile não identificado univocamente.")
    output = run.path / "viirs_coverage"
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / "daily_progress.json"
    progress = read_json(progress_path) if progress_path.exists() else {"schema_version": "2.0",
        "product": "VJ114A1.002", "config_sha256": run.config_hash, "days_of_month": cfg["days_of_month"],
        "months": cfg["months"], "partitions": {}, "created_utc": utc_now()}
    if progress["config_sha256"] != run.config_hash:
        raise ValueError("Configuração incompatível com o checkpoint.")
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        for filename, item in read_json(manifest_path)["files"].items():
            path = output / filename
            if not path.exists() or sha256_file(path) != item["sha256"]:
                raise ValueError("Grânulo salvo não confere com o manifesto.")
    session = requests.Session()
    run.stage("viirs_daily_collect", "running", partitions_complete=len(progress["partitions"]))
    try:
        for year in years_config:
            for region in regions_config:
                name, tile = region["name"], tiles[region["name"]][0]
                key = f"{name}/{year}"
                if key in progress["partitions"]:
                    continue
                run.guard()
                half = cfg["bbox_half_width_degrees"]
                lat, lon = region["latitude"], region["longitude"]
                bbox = (lon - half, lat - half, lon + half, lat + half)
                wanted = {date(year, month, day).isoformat() for month in cfg["months"] for day in cfg["days_of_month"]}
                time.sleep(1.)
                found = query_granules(run, bbox, f"{year}-01-01", f"{year}-12-31", limit=500, session=session)
                if found["truncated"]:
                    raise ValueError(f"Catálogo truncou {key}; subdivida a consulta.")
                matching = [x for x in found["granules"]
                            if f".{tile}." in x["title"] and x["start_utc"][:10] in wanted]
                by_date: dict[str, list[dict[str, Any]]] = {}
                for item in matching:
                    by_date.setdefault(item["start_utc"][:10], []).append(item)
                ambiguous = sorted(day for day, group in by_date.items() if len(group) != 1)
                eligible = [group[0] for day, group in by_date.items() if day not in ambiguous]
                unavailable: list[str] = []
                if eligible:
                    time.sleep(1.)
                    outcome = download_with_gaps(run, {**found, "granules": eligible}, token, session=session)
                    unavailable = outcome["unavailable_files"]
                unavailable_days = sorted(item["start_utc"][:10] for item in eligible if item["filename"] in unavailable)
                progress["partitions"][key] = {"tile": tile, "expected_days": sorted(wanted),
                    "filenames": [x["filename"] for x in eligible if x["filename"] not in unavailable],
                    "missing_days": sorted((wanted - set(by_date)) | set(unavailable_days)),
                    "download_failed_days": unavailable_days,
                    "ambiguous_days": ambiguous, "completed_utc": utc_now(), "cmr_sha256": found["cmr_response_sha256"]}
                write_json(progress_path, progress)
                count = len(progress["partitions"])
                run.stage("viirs_daily_collect", "running", partitions_complete=count, latest=key)
                print(f"Partição concluída: {key}; total: {count}", flush=True)
    except BaseException:
        run.stage("viirs_daily_collect", "interrupted", partitions_complete=len(progress["partitions"]))
        raise
    finally:
        session.close()
    parts = list(progress["partitions"].values())
    result = {"status": "ok", "partitions": len(parts), "files": sum(len(x["filenames"]) for x in parts),
              "missing_days": sum(len(x["missing_days"]) for x in parts),
              "ambiguous_days": sum(len(x["ambiguous_days"]) for x in parts), "progress": str(progress_path)}
    run.stage("viirs_daily_collect", "complete", **result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Coleta sistemática VJ114A1 por ano")
    parser.add_argument("--config", type=Path, default=Path("config/viirs_daily_v4b.json"))
    parser.add_argument("--run-id", default="viirs_daily_v4b_20261111")
    parser.add_argument("--region", action="append")
    parser.add_argument("--year", type=int, action="append")
    args = parser.parse_args()
    result = collect(Run(args.config, args.run_id), os.environ.get("EARTHDATA_TOKEN", ""),
                     set(args.region) if args.region else None, set(args.year) if args.year else None)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
