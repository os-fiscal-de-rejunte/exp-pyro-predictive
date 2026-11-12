"""Grade geográfica estável e geometria real dos recortes MODIS."""

from __future__ import annotations
import math
from typing import Any
from .power import grid_center
from .ingest import region_id

RADIUS = 6371007.181


def locate(latitude: float, longitude: float) -> dict[str, Any]:
    if not math.isfinite(latitude) or not math.isfinite(longitude):
        raise ValueError("Coordenadas devem ser finitas.")
    lat, lon = grid_center(latitude, longitude)
    south, north = max(-90.0, lat - 0.25), min(90.0, lat + 0.25)
    west, east = lon - 0.3125, lon + 0.3125
    area = (
        RADIUS**2
        * math.radians(0.625)
        * (math.sin(math.radians(north)) - math.sin(math.radians(south)))
        / 1e6
    )
    rings = []
    for w, e in (
        [(west + 360, 180.0), (-180.0, east)] if west < -180 else [(west, east)]
    ):
        rings.append([[[w, south], [e, south], [e, north], [w, north], [w, south]]])
    return {
        "id": region_id(lat, lon),
        "center": [lon, lat],
        "area_km2": area,
        "geometry": {"type": "MultiPolygon", "coordinates": rings},
    }


def footprint(row: dict[str, Any]) -> dict[str, Any]:
    x, y = float(row["xllcorner"]), float(row["yllcorner"])
    width, height = (
        int(row["ncols"]) * float(row["cellsize_m"]),
        int(row["nrows"]) * float(row["cellsize_m"]),
    )
    points = []
    for xx, yy in [
        (x, y),
        (x + width, y),
        (x + width, y + height),
        (x, y + height),
        (x, y),
    ]:
        lat = yy / RADIUS
        points.append([math.degrees(xx / (RADIUS * math.cos(lat))), math.degrees(lat)])
    return {"type": "Polygon", "coordinates": [points]}


def contains(row: dict[str, Any], latitude: float, longitude: float) -> bool:
    x = RADIUS * math.radians(longitude) * math.cos(math.radians(latitude))
    y = RADIUS * math.radians(latitude)
    return float(row["xllcorner"]) <= x <= float(row["xllcorner"]) + int(
        row["ncols"]
    ) * float(row["cellsize_m"]) and float(row["yllcorner"]) <= y <= float(
        row["yllcorner"]
    ) + int(row["nrows"]) * float(row["cellsize_m"])
