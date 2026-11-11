"""Coleta tiles adjacentes quando uma área VIIRS cruza a borda de um grânulo."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from netCDF4 import Dataset

from .io import read_json, sha256_file, utc_now, write_json
from .runtime import Run
from .viirs_coverage import download_with_gaps, query_granules

RADIUS = 6371007.181


def adjacent_tile(geometry: dict[str, Any], x0: float, y0: float, nx: int, ny: int,
                  tile: str) -> tuple[str, float, float] | None:
    cell, nr, nc = float(geometry["cellsize_m"]), int(geometry["nrows"]), int(geometry["ncols"])
    left = round((float(geometry["xllcorner"]) - x0) / cell)
    top = round((y0 - (float(geometry["yllcorner"]) + nr * cell)) / cell)
    edges = [edge for edge, condition in (("west", left < 0), ("east", left + nc > nx),
                                          ("north", top < 0), ("south", top + nr > ny)) if condition]
    if not edges:
        return None
    if len(edges) != 1:
        raise ValueError("Recorte cruza múltiplas bordas; mosaico de canto requer extensão específica.")
    match = re.fullmatch(r"h(\d{2})v(\d{2})", tile)
    if not match:
        raise ValueError("Identificador de tile inválido.")
    h, v = map(int, match.groups())
    x = float(geometry["xllcorner"]) + nc * cell / 2
    y = float(geometry["yllcorner"]) + nr * cell / 2
    if edges[0] == "west": h, x = h - 1, x0 - 2 * cell
    elif edges[0] == "east": h, x = h + 1, x0 + (nx + 2) * cell
    elif edges[0] == "north": v, y = v - 1, y0 + 2 * cell
    else: v, y = v + 1, y0 - (ny + 2) * cell
    latitude = math.degrees(y / RADIUS)
    if not 0 <= h <= 35 or not 0 <= v <= 17 or abs(latitude) >= 89.9:
        raise ValueError("Tile adjacente ou coordenada fora do domínio suportado.")
    longitude = math.degrees(x / (RADIUS * math.cos(math.radians(latitude))) )
    return f"h{h:02d}v{v:02d}", latitude, longitude


def collect(run: Run, token: str) -> dict[str, Any]:
    if not token:
        raise ValueError("Token Earthdata não fornecido no ambiente.")
    base = run.path / "viirs_coverage"
    progress = read_json(base / "daily_progress.json")
    manifest = read_json(base / "manifest.json")
    extra_path = base / "extra_validation_progress.json"
    extra = read_json(extra_path) if extra_path.exists() else {"partitions": {}}
    if progress["config_sha256"] != run.config_hash:
        raise ValueError("Configuração não corresponde à coleta principal.")
    if extra_path.exists() and extra["config_sha256"] != run.config_hash:
        raise ValueError("Configuração não corresponde à validação complementar.")
    reference = pd.read_parquet(run.root / "runs/nasa_complement_v3b_20261111/coverage/mask_summary.parquet")
    geometries = {name: group.iloc[0].to_dict() for name, group in reference.groupby("region_name")}
    plans: dict[str, tuple[str, float, float]] = {}
    for key, partition in progress["partitions"].items():
        region, _ = key.split("/", 1)
        if region in plans or not partition["filenames"]:
            continue
        filename = partition["filenames"][0]
        file = base / filename
        if sha256_file(file) != manifest["files"][filename]["sha256"]:
            raise ValueError("Grânulo central não corresponde ao manifesto.")
        with Dataset(str(file.relative_to(Path.cwd().resolve()))) as ds:
            grid = ds.groups["HDFEOS"].groups["GRIDS"].groups["VIIRS_Grid_Daily_Fire"]
            x, y = grid.variables["XDim"], grid.variables["YDim"]
            adjacent = adjacent_tile(geometries[region], float(x[0]), float(y[0]), len(x), len(y), partition["tile"])
        if adjacent is not None:
            plans[region] = adjacent
    path = base / "halo_progress.json"
    halo = read_json(path) if path.exists() else {"schema_version": "1.0", "config_sha256": run.config_hash,
                                                   "partitions": {}, "created_utc": utc_now()}
    if halo["config_sha256"] != run.config_hash:
        raise ValueError("Checkpoint de borda incompatível.")
    run.stage("viirs_halo_collect", "running", regions=sorted(plans), partitions_complete=len(halo["partitions"]))
    session = requests.Session()
    try:
        for key, center in progress["partitions"].items():
            region, year = key.split("/", 1)
            if region not in plans or key in halo["partitions"]:
                continue
            tile, lat, lon = plans[region]
            half = run.config["viirs_daily"]["bbox_half_width_degrees"]
            bbox = (lon - half, lat - half, lon + half, lat + half)
            filenames = center["filenames"] + extra["partitions"].get(key, {}).get("filenames", [])
            wanted = {manifest["files"][filename]["start_utc"][:10] for filename in filenames}
            # A geometria geográfica do catálogo pode tocar tiles vizinhos: dividir o ano
            # mantém cada resposta abaixo do limite e evita perder um tile por paginação.
            chunks = []
            for first, last in (("01-01", "03-31"), ("04-01", "06-30"), ("07-01", "09-30"), ("10-01", "12-31")):
                time.sleep(1.)
                response = query_granules(run, bbox, f"{year}-{first}", f"{year}-{last}", limit=500, session=session)
                if response["truncated"]:
                    raise ValueError(f"Catálogo truncou trimestre do tile adjacente de {key}.")
                chunks.append(response)
            matching = [item for chunk in chunks for item in chunk["granules"] if f".{tile}." in item["title"]
                        and item["start_utc"][:10] in wanted]
            by_date: dict[str, list[dict[str, Any]]] = {}
            for item in matching:
                by_date.setdefault(item["start_utc"][:10], []).append(item)
            ambiguous = sorted(day for day, group in by_date.items() if len(group) != 1)
            selected = [group[0] for day, group in by_date.items() if day not in ambiguous]
            unavailable: list[str] = []
            if selected:
                time.sleep(1.)
                result = download_with_gaps(run, {"product": "VJ114A1.002", "granules": selected}, token, session=session)
                unavailable = result["unavailable_files"]
            unavailable_days = sorted(item["start_utc"][:10] for item in selected if item["filename"] in unavailable)
            halo["partitions"][key] = {"tile": tile, "expected_days": sorted(wanted),
                "filenames": [x["filename"] for x in selected if x["filename"] not in unavailable],
                "missing_days": sorted((wanted - set(by_date)) | set(unavailable_days)), "download_failed_days": unavailable_days,
                "ambiguous_days": ambiguous, "cmr_sha256": [x["cmr_response_sha256"] for x in chunks],
                "completed_utc": utc_now()}
            write_json(path, halo)
            run.stage("viirs_halo_collect", "running", regions=sorted(plans),
                      partitions_complete=len(halo["partitions"]), latest=key)
            print(f"Borda concluída: {key}", flush=True)
    except BaseException:
        run.stage("viirs_halo_collect", "interrupted", partitions_complete=len(halo["partitions"]))
        raise
    finally:
        session.close()
    result = {"status": "ok", "regions": sorted(plans), "partitions": len(halo["partitions"]),
              "files": sum(len(x["filenames"]) for x in halo["partitions"].values()),
              "missing_days": sum(len(x["missing_days"]) for x in halo["partitions"].values())}
    write_json(path, halo)
    run.stage("viirs_halo_collect", "complete", **result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/viirs_daily_v4b.json"))
    parser.add_argument("--run-id", default="viirs_daily_v4b_20261111")
    args = parser.parse_args()
    print(json.dumps(collect(Run(args.config, args.run_id), os.environ.get("EARTHDATA_TOKEN", "")),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
