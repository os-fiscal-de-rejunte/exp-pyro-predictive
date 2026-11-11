"""Leitura em blocos dos CSVs sem extrair ou alterar os arquivos de entrada."""
from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd


def archive_product(path: Path) -> tuple[str, int]:
    match = re.fullmatch(r"(modis|viirs-jpss1)_(\d{4})_all_countries(?: \(1\))?\.zip", path.name, re.I)
    if not match:
        raise ValueError(f"Nome de produto não reconhecido: {path.name}")
    return match.group(1).lower(), int(match.group(2))


def csv_chunks(path: Path, member: str, chunk_size: int = 50000) -> Iterator[pd.DataFrame]:
    with zipfile.ZipFile(path) as archive, archive.open(member) as stream:
        yield from pd.read_csv(stream, chunksize=chunk_size, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def parse_datetime(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    raw = frame["acq_time"].astype(str).str.strip()
    lexical = raw.str.fullmatch(r"\d{1,4}").fillna(False)
    hhmm = raw.str.zfill(4)
    hour = pd.to_numeric(hhmm.str[:2], errors="coerce")
    minute = pd.to_numeric(hhmm.str[2:], errors="coerce")
    good_time = lexical & hour.between(0, 23) & minute.between(0, 59)
    dates = pd.to_datetime(frame["acq_date"], format="%Y-%m-%d", errors="coerce")
    timestamps = pd.to_datetime(frame["acq_date"] + " " + hhmm, format="%Y-%m-%d %H%M", errors="coerce", utc=True)
    return dates, timestamps.where(good_time)


def validation_masks(frame: pd.DataFrame, product: str) -> dict[str, pd.Series]:
    masks: dict[str, pd.Series] = {}
    if "latitude" in frame:
        masks["latitude_invalid"] = ~pd.to_numeric(frame["latitude"], errors="coerce").between(-90, 90)
    if "longitude" in frame:
        masks["longitude_invalid"] = ~pd.to_numeric(frame["longitude"], errors="coerce").between(-180, 180)
    if {"acq_date", "acq_time"}.issubset(frame.columns):
        dates, timestamps = parse_datetime(frame)
        masks["date_invalid"], masks["time_invalid"] = dates.isna(), timestamps.isna() & dates.notna()
    if "confidence" in frame:
        confidence = frame["confidence"].astype(str).str.strip()
        conf_good = (pd.to_numeric(confidence, errors="coerce").between(0, 100) if product == "modis"
                     else confidence.str.lower().isin(["l", "n", "h"]))
        masks["confidence_invalid"] = ~conf_good
    if "daynight" in frame:
        masks["daynight_invalid"] = ~frame["daynight"].isin(["D", "N"])
    for col in ("frp", "scan", "track", "brightness", "bright_t31", "bright_ti4", "bright_ti5"):
        if col in frame:
            numeric = pd.to_numeric(frame[col], errors="coerce")
            masks[col + "_invalid"] = numeric.isna() | ~np.isfinite(numeric) | (numeric < 0)
    return masks


def grid_indices(lat: pd.Series, lon: pd.Series) -> tuple[pd.Series, pd.Series]:
    # Centros MERRA-2; as observações são agregadas em retângulos definidos pelo projeto.
    i = np.floor((lat + 90) / .5 + .5).astype(int)
    j = np.floor(((lon + 180) % 360) / .625 + .5).astype(int) % 576
    return i, j


def region_id(latitude: float, longitude: float) -> str:
    from .power import grid_center
    lat, lon = grid_center(latitude, longitude)
    return f"grid_{lat:+07.2f}_{lon:+08.3f}"
