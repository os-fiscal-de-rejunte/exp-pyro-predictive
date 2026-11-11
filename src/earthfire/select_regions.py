"""Seleciona células brasileiras usando apenas semanas positivas de treino."""
from __future__ import annotations

import argparse
import hashlib
import math
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from .ingest import grid_indices, region_id
from .io import write_json
from .runtime import Run


def select_regions(run: Run, limit: int = 16, minimum_pairs: int = 25,
                   min_distance_deg: float = 1.5) -> dict[str, Any]:
    """Seleção deliberada por sobreposição, não uma amostra representativa do Brasil."""
    root = run.root / run.config["data_dir"]
    observed: dict[tuple[int, int, str], set[int]] = defaultdict(set)
    files = []
    for year in range(2018, 2022):
        for product, satellite in (("modis", "Terra"), ("viirs-jpss1", "N20")):
            path = root / f"{product}_{year}_all_countries.zip"
            if not path.exists():
                raise FileNotFoundError(path)
            with zipfile.ZipFile(path) as archive:
                member = next((n for n in archive.namelist() if n.endswith("_Brazil.csv")), None)
                if member is None:
                    raise ValueError(f"Partição Brazil ausente em {path}")
                with archive.open(member) as stream:
                    for frame in pd.read_csv(stream, usecols=["latitude", "longitude", "acq_date", "satellite"], chunksize=100000):
                        run.guard()
                        frame = frame.loc[(frame.satellite == satellite) & frame.latitude.between(-90,90) &
                                          frame.longitude.between(-180,180)].copy()
                        if frame.empty:
                            continue
                        dates = pd.to_datetime(frame["acq_date"], format="%Y-%m-%d", errors="coerce")
                        valid = dates.notna() & (dates.dt.year == year)
                        frame, dates = frame.loc[valid], dates.loc[valid]
                        lat, lon = grid_indices(frame.latitude, frame.longitude)
                        weeks = ((dates.dt.normalize() - pd.Timestamp("1970-01-05")).dt.days // 7).astype(int)
                        distinct = pd.DataFrame({"i": lat, "j": lon, "week": weeks}).drop_duplicates()
                        for i, j, week in distinct.itertuples(index=False, name=None):
                            observed[(int(i), int(j), product)].add(int(week))
            files.append({"archive": path.name, "member": member})
            run.logger.info("select train file=%s", path.name)
    cells = {(i, j) for i, j, _ in observed}
    candidates = []
    for i, j in cells:
        a = observed.get((i, j, "modis"), set())
        b = observed.get((i, j, "viirs-jpss1"), set())
        n = len(a & b)
        if n >= minimum_pairs:
            lat, lon = i * .5 - 90., j * .625 - 180.
            candidates.append({"latitude": lat, "longitude": lon, "paired_positive_weeks": n,
                               "modis_terra_positive_weeks": len(a), "viirs_n20_positive_weeks": len(b)})
    # Sorteio determinístico dentro de blocos da grade, preservando diversidade
    # geográfica e sem usar dados de validação ou teste.
    seed = run.config["seed"]
    def stable(value: str) -> str:
        return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()
    groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for item in candidates:
        i = round((item["latitude"] + 90) / .5)
        j = round((item["longitude"] + 180) / .625)
        groups[(i // 15, j // 15)].append(item)
    for group in groups.values():
        group.sort(key=lambda x: stable(f"{x['latitude']},{x['longitude']}"))
    tiles = sorted(groups, key=lambda x: stable(str(x)))
    selected: list[dict[str, Any]] = []
    while len(selected) < limit and any(groups.values()):
        for tile in tiles:
            while groups[tile]:
                candidate = groups[tile].pop(0)
                lat, lon = candidate["latitude"], candidate["longitude"]
                if all(math.hypot(lat-x["latitude"], (lon-x["longitude"]) * math.cos(math.radians(lat))) >= min_distance_deg for x in selected):
                    selected.append({**candidate, "name": f"br_cell_{len(selected)+1:02}",
                                     "region_id": region_id(lat, lon), "weather_tile": list(tile)})
                    break
            if len(selected) == limit:
                break
    if len(selected) < min(8, limit):
        raise ValueError(f"Apenas {len(selected)} células elegíveis; reduza os critérios explicitamente.")
    result = {"selection_period": ["2018-01-01", "2021-12-31"], "selection_only_train": True,
              "method": "sorteio determinístico estratificado por bloco da grade, com mínimo de pares positivos e separação espacial aproximada",
              "minimum_pairs": minimum_pairs, "min_distance_degrees_approx": min_distance_deg,
              "scope": "partições Brazil, MODIS/Terra e VIIRS/NOAA-20; seleção intencional de atividade frequente",
              "candidate_count": len(candidates), "files": files, "regions": selected}
    write_json(run.path / "selection" / "regions.json", result)
    run.stage("select_regions", "complete", count=len(selected), candidates=len(candidates))
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("config/default.json"))
    p.add_argument("--run-id", default="selection_zarr_v2_20261111")
    p.add_argument("--limit", type=int, default=16)
    p.add_argument("--minimum-pairs", type=int, default=25)
    args = p.parse_args()
    selected = select_regions(Run(args.config, args.run_id), args.limit, args.minimum_pairs)
    print(f"{len(selected['regions'])} células; {selected['candidate_count']} candidatas")
    for x in selected["regions"]:
        print(x["name"], x["latitude"], x["longitude"], x["paired_positive_weeks"])


if __name__ == "__main__":
    main()
