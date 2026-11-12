"""Inventário incremental e associação administrativa aproximada, sem alterar DATA."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

from .io import sha256_file, utc_now, write_json


def _in_ring(x: float, y: float, ring: list) -> bool:
    inside = False
    for a, b in zip(ring, ring[1:]):
        if (a[1] > y) != (b[1] > y) and x < (b[0] - a[0]) * (y - a[1]) / (
            b[1] - a[1]
        ) + a[0]:
            inside = not inside
    return inside


def country_names(latitude: float, longitude: float, geojson: dict) -> list[str]:
    names = []
    for feature in geojson["features"]:
        geometry = feature["geometry"]
        polygons = (
            geometry["coordinates"]
            if geometry["type"] == "MultiPolygon"
            else [geometry["coordinates"]]
        )
        if any(
            _in_ring(longitude, latitude, p[0])
            and not any(_in_ring(longitude, latitude, h) for h in p[1:])
            for p in polygons
        ):
            names.append(feature["properties"]["ADMIN"])
    return names


def inventory(data_dir: Path, destination: Path) -> dict[str, Any]:
    items = []
    for path in sorted(data_dir.iterdir()):
        if not path.is_file():
            continue
        item = {
            "name": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "format": path.suffix,
        }
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as archive:
                members = [v for v in archive.infolist() if v.filename.endswith(".csv")]
                item.update(
                    csv_partitions=len(members),
                    uncompressed_bytes=sum(v.file_size for v in members),
                )
            item["coverage"] = (
                "partições de países; completude observacional não estabelecida pelos registros positivos"
            )
        elif path.suffix.lower() == ".json":
            with path.open(encoding="utf-8") as stream:
                body = json.load(stream)
            item["top_level_keys"] = list(body)
            item["interpretation"] = (
                "contém anomalypercent/ssp585; não usado como meteorologia diária observada; origem não confirmada"
            )
        items.append(item)
    grouped = {}
    for item in items:
        grouped.setdefault(item["sha256"], []).append(item["name"])
    result = {
        "created_utc": utc_now(),
        "files": items,
        "exact_file_duplicates": [v for v in grouped.values() if len(v) > 1],
        "scope": "inventário incremental com hashes; auditorias de linhas anteriores permanecem nos relatórios v1–v4",
    }
    write_json(destination, result)
    return result
