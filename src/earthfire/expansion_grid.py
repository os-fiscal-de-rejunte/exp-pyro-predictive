"""Células canônicas MODIS de 41×41 pixels, sem sobreposição nem inflação de área."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .global_grid import RADIUS

PIXEL = 926.6254330558338
XMIN = -20015109.354
YMAX = 10007554.677
SIDE = 41


def geometry(row: int, column: int) -> dict[str, Any]:
    x = XMIN + column * PIXEL
    y = YMAX - (row + SIDE) * PIXEL
    latitude = math.degrees((y + SIDE * PIXEL / 2) / RADIUS)
    longitude = math.degrees(
        (x + SIDE * PIXEL / 2) / (RADIUS * math.cos(math.radians(latitude)))
    )
    return {
        "region_id": f"m41_r{row:05d}_c{column:05d}",
        "latitude": latitude,
        "longitude": longitude,
        "xllcorner": x,
        "yllcorner": y,
        "nrows": SIDE,
        "ncols": SIDE,
        "cellsize_m": PIXEL,
        "native_row": row,
        "native_column": column,
        "area_km2": (SIDE * PIXEL) ** 2 / 1e6,
    }


def point_cell(latitude: float, longitude: float) -> dict[str, Any]:
    if (
        not math.isfinite(latitude)
        or not math.isfinite(longitude)
        or not -90 < latitude < 90
        or not -180 <= longitude <= 180
    ):
        raise ValueError("Coordenadas inválidas ou polo sem célula consultável.")
    x = RADIUS * math.radians(longitude) * math.cos(math.radians(latitude))
    y = RADIUS * math.radians(latitude)
    row = math.floor((YMAX - y) / (SIDE * PIXEL)) * SIDE
    column = math.floor((x - XMIN) / (SIDE * PIXEL)) * SIDE
    return geometry(row, column)


def split_body(
    body: dict[str, Any], parent: dict[str, Any], source_sha: str
) -> list[dict[str, Any]]:
    if body.get("band") != "FireMask" or body.get("units") != "class-flag":
        raise ValueError("Máscara ou unidade inesperada.")
    nr, nc = int(body["nrows"]), int(body["ncols"])
    pixel = float(body["cellsize"])
    if abs(pixel - PIXEL) > 0.001:
        raise ValueError("Resolução inesperada.")
    first_col = round((float(body["xllcorner"]) - XMIN) / PIXEL)
    first_row = round((YMAX - (float(body["yllcorner"]) + nr * PIXEL)) / PIXEL)
    if abs(float(body["xllcorner"]) - (XMIN + first_col * PIXEL)) > 0.1:
        raise ValueError("Alinhamento MODIS inesperado.")
    roff, coff = (-first_row) % SIDE, (-first_col) % SIDE
    rows = []
    for entry in body["subset"]:
        raw = np.asarray(entry["data"])
        if (
            raw.size != nr * nc
            or not np.issubdtype(raw.dtype, np.integer)
            or ((raw < 0) | (raw > 9)).any()
        ):
            raise ValueError("Classes ou dimensões da máscara inconsistentes.")
        array = raw.reshape(nr, nc)
        for i in range(roff, nr - SIDE + 1, SIDE):
            for j in range(coff, nc - SIDE + 1, SIDE):
                cell = geometry(first_row + i, first_col + j)
                if abs(cell["latitude"]) >= 80 or abs(cell["longitude"]) > 180:
                    continue
                counts = np.bincount(
                    array[i : i + SIDE, j : j + SIDE].ravel(), minlength=10
                )
                valid = int(counts[[3, 5, 7, 8, 9]].sum())
                fire = int(counts[[7, 8, 9]].sum())
                rows.append(
                    {
                        **cell,
                        "parent_id": parent["name"],
                        "continent": parent["continent"],
                        "spatial_group": parent["group"],
                        "period_start": entry["calendar_date"],
                        "proc_date": str(entry.get("proc_date", "")),
                        "tile": entry.get("tile"),
                        "valid_pixels": valid,
                        "fire_pixels": fire,
                        "cloud_pixels": int(counts[4]),
                        "water_pixels": int(counts[3]),
                        "total_pixels": SIDE * SIDE,
                        "fully_observed": valid == SIDE * SIDE,
                        "event": int(fire > 0) if valid == SIDE * SIDE else None,
                        "source_sha256": source_sha,
                    }
                )
    return rows
