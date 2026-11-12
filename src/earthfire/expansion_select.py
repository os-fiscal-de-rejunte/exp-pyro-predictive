"""Escolhe blocos globais por atividade MODIS/Terra exclusivamente no treino."""

from __future__ import annotations

import math
import zipfile
from collections import defaultdict
from pathlib import Path

import pandas as pd

from .global_catalog import _in_ring
from .io import read_json, sha256_file, write_json
from .power import grid_center
from .runtime import Run


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    h = (
        math.sin((lat2 - lat1) / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    )
    return 6371.007181 * 2 * math.asin(min(1, math.sqrt(h)))


def select(run: Run) -> dict:
    output = run.path / "selection.json"
    if output.exists():
        return read_json(output)
    active = defaultdict(set)
    sources = []
    count = 0
    for year in (2020, 2021, 2022):
        source = run.root / run.config["data_dir"] / f"modis_{year}_all_countries.zip"
        digest = sha256_file(source)
        with zipfile.ZipFile(source) as archive:
            for member in archive.namelist():
                if not member.endswith(".csv"):
                    continue
                with archive.open(member) as stream:
                    for frame in pd.read_csv(
                        stream,
                        usecols=[
                            "latitude",
                            "longitude",
                            "acq_date",
                            "satellite",
                            "confidence",
                            "type",
                        ],
                        chunksize=10000,
                    ):
                        run.guard()
                        count += len(frame)
                        frame = frame.loc[
                            (frame.satellite == "Terra")
                            & (pd.to_numeric(frame["type"], errors="coerce") == 0)
                            & frame.latitude.between(-55, 65)
                            & frame.longitude.between(-180, 180)
                            & frame.confidence.between(0, 100)
                        ].copy()
                        dates = pd.to_datetime(frame.acq_date, errors="coerce")
                        frame = frame.loc[dates.notna() & (dates.dt.year == year)]
                        if frame.empty:
                            continue
                        ii = ((frame.latitude + 90) // 2).astype(int)
                        jj = ((frame.longitude + 180) // 2).astype(int)
                        week = (
                            (
                                dates.loc[frame.index] - pd.Timestamp("2020-01-01")
                            ).dt.days
                            // 7
                        ).astype(int)
                        group = pd.DataFrame(
                            {"i": ii, "j": jj, "w": week}
                        ).drop_duplicates()
                        for i, j, w in group.itertuples(index=False, name=None):
                            active[(int(i), int(j))].add(int(w))
        sources.append({"path": source.name, "sha256": digest})
        print(
            f"Seleção: ano {year}, {count} registros lidos, {len(active)} candidatos",
            flush=True,
        )
        run.stage("expansion_select", "running", year=year, rows=count)
    countries = read_json(run.root / "artifacts/global_v5/catalog/countries.geojson")
    candidates = defaultdict(list)
    for (i, j), weeks in active.items():
        if len(weeks) < run.config["expansion"]["min_training_active_weeks"]:
            continue
        lat, lon = grid_center(i * 2 - 89, j * 2 - 179)
        for feature in countries["features"]:
            g = feature["geometry"]
            polygons = (
                g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
            )
            if any(
                _in_ring(lon, lat, p[0])
                and not any(_in_ring(lon, lat, h) for h in p[1:])
                for p in polygons
            ):
                continent = feature["properties"]["CONTINENT"]
                if continent not in ("Antarctica", "Seven seas (open ocean)"):
                    candidates[continent].append(
                        {
                            "latitude": lat,
                            "longitude": lon,
                            "continent": continent,
                            "country": feature["properties"]["ADMIN"],
                            "training_active_weeks": len(weeks),
                        }
                    )
                break
    for items in candidates.values():
        items.sort(
            key=lambda r: (-r["training_active_weeks"], r["latitude"], r["longitude"])
        )
    cycle = [
        "Africa",
        "Asia",
        "South America",
        "Oceania",
        "North America",
        "Europe",
        "Africa",
        "Asia",
        "Oceania",
        "South America",
        "Africa",
    ]
    selected = []
    legacy = read_json(run.root / "config/global_v5b.json")["regions"]
    while len(selected) < run.config["expansion"]["max_blocks"]:
        added = 0
        for continent in cycle:
            items = candidates[continent]
            while items:
                item = items.pop(0)
                point = (item["latitude"], item["longitude"])
                if any(
                    distance(point, (r["latitude"], r["longitude"]))
                    < run.config["expansion"]["minimum_parent_separation_km"]
                    for r in selected
                ):
                    continue
                item.update(
                    name=f"block_{len(selected)+1:03d}",
                    group="holdout" if len(selected) % 5 == 4 else "train",
                    legacy_context=any(
                        distance(point, (r["latitude"], r["longitude"])) < 250
                        for r in legacy
                    ),
                )
                selected.append(item)
                added += 1
                break
            if len(selected) >= run.config["expansion"]["max_blocks"]:
                break
        if not added:
            break
    result = {
        "selection_period": ["2020-01-01", "2022-12-31"],
        "uses_test": False,
        "rows_read": count,
        "sources": sources,
        "type_filter": 0,
        "type_definition": "presumed vegetation fire; NASA FIRMS; classificação inferida",
        "code_sha256": sha256_file(Path(__file__)),
        "method": "atividade Terra de vegetação presumida no treino; diversidade continental e separação geodésica; seleção intencional, não representativa",
        "blocks": selected,
    }
    write_json(output, result)
    run.stage("expansion_select", "complete", blocks=len(selected))
    return result
