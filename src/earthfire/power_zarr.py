"""Recorta o Zarr público da NASA POWER sem baixar o arquivo global.

Os objetos Zarr são lidos sequencialmente, com cache persistente e orçamento.
"""
from __future__ import annotations

import hashlib
import argparse
import json
import math
import os
import time
from collections.abc import Iterator, MutableMapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import zarr
from numcodecs import blosc

from .ingest import region_id
from .io import read_json, replace_with_retry, sha256_file, utc_now, write_json
from .power import EXPECTED_UNITS, NasaPowerClient, power_frame
from .runtime import ResourceLimit, Run

BASE_URL = "https://nasa-power.s3.us-west-2.amazonaws.com/merra2/temporal/power_merra2_daily_temporal_utc.zarr"
VARIABLES = ("T2M", "T2M_MAX", "T2M_MIN", "RH2M", "PRECTOTCORR", "WS10M", "GWETROOT", "GWETTOP")
UNITS = {**EXPECTED_UNITS, "GWETROOT": "1", "GWETTOP": "1"}


class CachedZarrStore(MutableMapping[str, bytes]):
    """Mapping somente leitura; cada tentativa HTTP é contada no manifesto."""

    def __init__(self, run: Run, cache_dir: Path, base_url: str = BASE_URL,
                 session: requests.Session | None = None, min_interval: float = .2) -> None:
        if not base_url.startswith("https://"):
            raise ValueError("A fonte Zarr deve usar HTTPS.")
        self.run, self.cache_dir, self.base_url = run, Path(cache_dir), base_url.rstrip("/")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = session or requests.Session()
        self.min_interval = min_interval
        self.manifest_path = self.cache_dir / "manifest.json"
        self.manifest = read_json(self.manifest_path) if self.manifest_path.exists() else {
            "source": self.base_url, "created_utc": utc_now(), "requests": 0, "objects": {}, "temporary_files": []}
        if self.manifest["source"] != self.base_url:
            raise ValueError("Cache associado a outra fonte Zarr.")

    def __getitem__(self, key: str) -> bytes:
        path = Path(key)
        if path.is_absolute() or ".." in path.parts or "\\" in key:
            raise KeyError("Chave Zarr inválida")
        target = self.cache_dir / "objects" / path
        if target.is_file():
            content = target.read_bytes()
            expected = self.manifest["objects"].get(key, {}).get("sha256")
            if not expected or hashlib.sha256(content).hexdigest() != expected:
                raise ValueError(f"Objeto em cache incompatível: {key}")
            return content
        ops = self.run.config["operational"]
        last_error = "sem resposta"
        for attempt in range(min(3, ops["network_retries"])):
            self.run.guard()
            if self.manifest["requests"] >= ops["nasa_power_requests"]:
                raise ResourceLimit("Orçamento de requisições NASA POWER/Zarr esgotado.")
            wait = self.min_interval - (time.time() - self.manifest.get("last_request_epoch", 0))
            if wait > 0:
                time.sleep(wait)
            self.manifest["requests"] += 1
            self.manifest["last_request_epoch"] = time.time()
            write_json(self.manifest_path, self.manifest)
            try:
                response = self.session.get(f"{self.base_url}/{key}", timeout=ops["network_timeout_seconds"], stream=True)
                with response:
                    if response.status_code == 404:
                        self.manifest.setdefault("not_found", []).append(key)
                        write_json(self.manifest_path, self.manifest)
                        raise KeyError(key)
                    if response.status_code == 429 or response.status_code >= 500:
                        last_error = f"HTTP {response.status_code}"
                        retry_after = response.headers.get("Retry-After", "")
                        if retry_after.isdigit() and int(retry_after) > 30:
                            raise ResourceLimit(f"POWER solicita aguardar {retry_after} segundos; operação encerrada.")
                        if attempt + 1 < min(3, ops["network_retries"]):
                            time.sleep(max(2 ** attempt, int(retry_after) if retry_after.isdigit() else 0))
                        continue
                    response.raise_for_status()
                    expected_size = int(response.headers.get("Content-Length", 0) or 0)
                    if expected_size > 64 * 1024 * 1024:
                        raise ResourceLimit("Objeto Zarr excede o limite de 64 MiB por bloco.")
                    pieces = bytearray()
                    for piece in response.iter_content(chunk_size=128 * 1024):
                        pieces.extend(piece)
                        if len(pieces) > 64 * 1024 * 1024:
                            raise ResourceLimit("Objeto Zarr excede o limite de 64 MiB por bloco.")
                    if expected_size and len(pieces) != expected_size:
                        raise IOError(f"Objeto Zarr incompleto: {key}")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temporary = target.with_name(target.name + ".download.tmp")
                    if str(temporary) not in self.manifest["temporary_files"]:
                        self.manifest["temporary_files"].append(str(temporary))
                        write_json(self.manifest_path, self.manifest)
                    temporary.write_bytes(pieces)
                    replace_with_retry(temporary, target)
                    self.manifest["objects"][key] = {"sha256": hashlib.sha256(pieces).hexdigest(),
                        "bytes": len(pieces), "etag": response.headers.get("ETag"),
                        "last_modified": response.headers.get("Last-Modified"), "retrieved_utc": utc_now()}
                    write_json(self.manifest_path, self.manifest)
                    return bytes(pieces)
            except (requests.RequestException, IOError) as exc:
                last_error = str(exc)
                if attempt + 1 < min(3, ops["network_retries"]):
                    time.sleep(2 ** attempt)
        raise ConnectionError(f"Falha ao obter objeto Zarr {key}: {last_error}")

    def __setitem__(self, key: str, value: bytes) -> None:
        raise PermissionError("Zarr remoto é somente leitura")

    def __delitem__(self, key: str) -> None:
        raise PermissionError("Zarr remoto é somente leitura")

    def __iter__(self) -> Iterator[str]:
        return iter(self.manifest["objects"])

    def __len__(self) -> int:
        return len(self.manifest["objects"])


def _tile_key(region: dict[str, Any], lat: np.ndarray, lon: np.ndarray, chunks: tuple[int, ...]) -> tuple[int, int, int, int]:
    i = int(np.argmin(np.abs(lat - region["latitude"])))
    j = int(np.argmin(np.abs(lon - region["longitude"])))
    if not np.isclose(lat[i], region["latitude"], atol=1e-6) or not np.isclose(lon[j], region["longitude"], atol=1e-6):
        raise ValueError("Centro configurado não coincide com a grade POWER.")
    return i // chunks[1], j // chunks[2], i, j


def fetch_weather_zarr(run: Run, start: str = "2018-01-01", end: str = "2025-12-31",
                       include_control: bool = True) -> dict[str, Any]:
    blosc.set_nthreads(1)
    base = run.path / "weather_zarr"
    base.mkdir(parents=True, exist_ok=True)
    store = CachedZarrStore(run, base / "cache")
    run.stage("weather_zarr", "running", source=BASE_URL)
    group = zarr.open_consolidated(store=zarr.storage.KVStore(store), mode="r")
    attrs = dict(group.attrs)
    lat, lon = group["lat"][:], group["lon"][:]
    time_attrs = dict(group["time"].attrs)
    units = time_attrs.get("units", "")
    if not units.startswith("days since "):
        raise ValueError(f"Unidades temporais Zarr não reconhecidas: {units}")
    origin = pd.Timestamp(units.removeprefix("days since "))
    dates = origin + pd.to_timedelta(group["time"][:], unit="D")
    begin, finish = pd.Timestamp(start), pd.Timestamp(end)
    positions = np.flatnonzero((dates >= begin) & (dates <= finish))
    if len(positions) != (finish - begin).days + 1 or np.any(np.diff(positions) != 1):
        raise ValueError("Eixo temporal Zarr não cobre todos os dias solicitados.")
    first, after = int(positions[0]), int(positions[-1] + 1)
    chunks = group[VARIABLES[0]].chunks
    for name in VARIABLES:
        if name not in group or group[name].attrs.get("units") != UNITS[name] or group[name].chunks != chunks:
            raise ValueError(f"Variável, unidade ou blocos inesperados no Zarr: {name}")
    regions = list(run.config["regions"])
    control = {"name": "api_control", "latitude": -18.5, "longitude": -57.5, "control_only": True}
    if include_control and not any(x["latitude"] == control["latitude"] and x["longitude"] == control["longitude"] for x in regions):
        regions.append(control)
    tiles: dict[tuple[int, int], list[tuple[dict[str, Any], int, int]]] = {}
    for r in regions:
        ti, tj, i, j = _tile_key(r, lat, lon, chunks)
        tiles.setdefault((ti, tj), []).append((r, i, j))
    tile_dir = base / "tiles"; tile_dir.mkdir(parents=True, exist_ok=True)
    metadata_sha = store.manifest["objects"][".zmetadata"]["sha256"]
    for (ti, tj), cells in sorted(tiles.items()):
        run.guard()
        output = tile_dir / f"tile_{ti}_{tj}.parquet"
        manifest = tile_dir / f"tile_{ti}_{tj}.json"
        ids = sorted(region_id(c[0]["latitude"], c[0]["longitude"]) for c in cells)
        if output.exists() and manifest.exists():
            old = read_json(manifest)
            if old["metadata_sha256"] != metadata_sha or old["region_ids"] != ids or old["start"] != start or old["end"] != end or old["parquet_sha256"] != sha256_file(output):
                raise ValueError("Checkpoint meteorológico incompatível.")
            continue
        lat0, lon0 = ti * chunks[1], tj * chunks[2]
        lat1, lon1 = min(lat0 + chunks[1], len(lat)), min(lon0 + chunks[2], len(lon))
        data: dict[str, dict[str, Any]] = {}
        for r, _, _ in cells:
            rid = region_id(r["latitude"], r["longitude"])
            data[rid] = {"date": dates[positions], "region_id": rid, "control_only": bool(r.get("control_only", False))}
        for variable in VARIABLES:
            run.guard()
            cube = group[variable][first:after, lat0:lat1, lon0:lon1]
            for r, i, j in cells:
                data[region_id(r["latitude"], r["longitude"])][variable] = cube[:, i-lat0, j-lon0]
            del cube
        frame = pd.concat([pd.DataFrame(x) for x in data.values()], ignore_index=True)
        frame.to_parquet(output, index=False)
        write_json(manifest, {"start": start, "end": end, "region_ids": ids, "metadata_sha256": metadata_sha,
                              "parquet_sha256": sha256_file(output), "completed_utc": utc_now()})
        run.logger.info("weather_tile=%s,%s regions=%s rows=%s", ti, tj, len(cells), len(frame))
    daily = pd.concat([pd.read_parquet(tile_dir / f"tile_{i}_{j}.parquet") for i, j in sorted(tiles)], ignore_index=True)
    daily.to_parquet(base / "daily.parquet", index=False)
    values = daily.copy()
    values["week_start"] = values.date.dt.to_period("W-SUN").dt.start_time
    grouped = values.groupby(["region_id", "week_start"], sort=True)
    counts = grouped[list(VARIABLES)].count()
    weekly = grouped[list(VARIABLES)].mean()
    weekly["PRECTOTCORR"] = grouped["PRECTOTCORR"].sum(min_count=7)
    weekly = weekly.loc[counts.eq(7).all(axis=1)].reset_index()
    weekly.to_parquet(base / "weekly.parquet", index=False)
    provenance = {"source": BASE_URL, "source_type": "NASA POWER Zarr v2/UTC", "metadata_sha256": metadata_sha,
        "dataset_attributes": attrs, "time_attributes": time_attrs, "variables": {v: dict(group[v].attrs) for v in VARIABLES},
        "start": start, "end": end, "cells": len(regions), "tiles": len(tiles),
        "requests": store.manifest["requests"], "finished_utc": utc_now(),
        "note": "Zarr pode ser revisado pela NASA; objetos e hashes foram preservados no cache local."}
    write_json(base / "provenance.json", provenance)
    _compare_api_control(run, daily, base)
    store.session.close()
    run.stage("weather_zarr", "complete", cells=len(regions), tiles=len(tiles), requests=store.manifest["requests"])
    return provenance


def _compare_api_control(run: Run, daily: pd.DataFrame, base: Path) -> None:
    """Compara somente com uma resposta API já arquivada; não abre outra conexão."""
    rid = region_id(-18.5, -57.5)
    subset = daily.loc[daily.region_id == rid].copy()
    cache = run.root / "runs/power_cache"
    selected = None
    for path in cache.glob("*.json"):
        if path.name == "request_budget.json": continue
        raw = read_json(path)
        request = raw.get("request", {})
        if request.get("latitude") == -18.5 and request.get("longitude") == -57.5 and request.get("start") == "20180101" and request.get("end") == "20251231":
            selected = (path, raw)
            break
    if selected is None or subset.empty:
        write_json(base / "api_comparison.json", {"status": "unavailable", "reason": "Resposta API histórica não encontrada."})
        return
    path, envelope = selected
    api = power_frame(envelope)
    cols = list(EXPECTED_UNITS)
    paired = subset[["date", *cols]].merge(api[["date", *cols]], on="date", suffixes=("_zarr", "_api"), validate="one_to_one")
    metrics = {}
    for name in cols:
        diff = (paired[name+"_zarr"] - paired[name+"_api"]).abs().dropna()
        metrics[name] = {"n": len(diff), "max_abs_difference": float(diff.max()) if len(diff) else None,
                         "mean_abs_difference": float(diff.mean()) if len(diff) else None, "unit": EXPECTED_UNITS[name]}
    write_json(base / "api_comparison.json", {"status": "compared", "region_id": rid,
        "api_response_path": str(path), "api_response_sha256": sha256_file(path), "metrics": metrics,
        "caveat": "Revisões temporais e conversões por comunidade podem produzir diferenças."})


def main() -> None:
    parser = argparse.ArgumentParser(description="Baixa recortes da grade diária UTC da NASA POWER")
    parser.add_argument("--config", type=Path, default=Path("config/zarr_brazil_v2.json"))
    parser.add_argument("--run-id", default="zarr_brasil_v2_20261111")
    args = parser.parse_args()
    run = Run(args.config, args.run_id)
    period = run.config.get("weather_period", {})
    try:
        result = fetch_weather_zarr(run, period.get("start", "2018-01-01"), period.get("end", "2025-12-31"))
        print(f"Zarr concluído: {result['cells']} células, {result['tiles']} blocos, {result['requests']} requisições")
    except (Exception, KeyboardInterrupt) as exc:
        run.stage("weather_zarr", "interrupted", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
