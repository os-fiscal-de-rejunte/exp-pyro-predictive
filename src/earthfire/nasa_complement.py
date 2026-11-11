"""Aquisição pública e reproduzível de FIRMS e máscaras MODIS/ORNL."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .ingest import region_id
from .io import read_json, replace_with_retry, sha256_file, utc_now, write_json
from .runtime import ResourceLimit, Run

FIRMS_FILES = {
    "viirs_jpss1_2023_Brazil.csv": "https://firms.modaps.eosdis.nasa.gov/data/country/viirs-jpss1/2023/viirs-jpss1_2023_Brazil.csv",
    "noaa20_south_america_7d.csv": "https://firms.modaps.eosdis.nasa.gov/data/active_fire/noaa-20-viirs-c2/csv/J1_VIIRS_C2_South_America_7d.csv",
    "modis_south_america_7d.csv": "https://firms.modaps.eosdis.nasa.gov/data/active_fire/modis-c6.1/csv/MODIS_C6_1_South_America_7d.csv",
}
ORNL_URL = "https://modis.ornl.gov/rst/api/v1/MOD14A2/subset"
VALID_MASK_CLASSES = {3, 5, 7, 8, 9}
FIRE_MASK_CLASSES = {7, 8, 9}


def download_firms(run: Run, names: tuple[str, ...] | None = None) -> dict[str, Any]:
    """Salva respostas brutas sem alterar DATA; jamais autentica ou envia registros."""
    folder = run.path / "firms_raw"
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / "manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {"files": {}, "temporary_files": []}
    session = requests.Session()
    run.stage("firms_download", "running")
    selected = FIRMS_FILES if names is None else {name: FIRMS_FILES[name] for name in names}
    for name, url in selected.items():
        run.guard()
        target = folder / name
        item = manifest["files"].get(name)
        if target.exists():
            if not item or sha256_file(target) != item["sha256"]:
                raise ValueError(f"Arquivo existente sem hash compatível: {target}")
            continue
        temporary = target.with_name(target.name + ".part")
        if str(temporary) not in manifest["temporary_files"]:
            manifest["temporary_files"].append(str(temporary))
            write_json(manifest_path, manifest)
        last_error = "sem resposta"
        for attempt in range(min(3, run.config["operational"]["network_retries"])):
            run.guard()
            try:
                with session.get(url, timeout=run.config["operational"]["network_timeout_seconds"], stream=True) as response:
                    if response.status_code == 429 or response.status_code >= 500:
                        last_error = f"HTTP {response.status_code}"
                        retry_after = response.headers.get("Retry-After", "")
                        if retry_after.isdigit() and int(retry_after) > 30:
                            raise ResourceLimit(f"FIRMS solicitou esperar {retry_after}s; operação interrompida.")
                        if attempt < 2:
                            time.sleep(max(2 ** attempt, int(retry_after) if retry_after.isdigit() else 0))
                        continue
                    if 400 <= response.status_code < 500:
                        raise ValueError(f"FIRMS recusou a consulta: HTTP {response.status_code}")
                    response.raise_for_status()
                    if "html" in response.headers.get("Content-Type", "").lower():
                        raise ValueError("FIRMS retornou HTML em vez de CSV.")
                    expected = int(response.headers.get("Content-Length", "0") or "0")
                    if expected > 512 * 1024 * 1024:
                        raise ResourceLimit("CSV remoto excede o limite individual de 512 MiB.")
                    size = 0
                    with temporary.open("wb") as stream:
                        for piece in response.iter_content(chunk_size=1024 * 1024):
                            if not piece:
                                continue
                            size += len(piece)
                            if size > 512 * 1024 * 1024:
                                raise ResourceLimit("CSV remoto excedeu o limite individual de 512 MiB.")
                            stream.write(piece)
                            if size % (8 * 1024 * 1024) < len(piece):
                                run.guard()
                    if expected and size != expected:
                        raise IOError(f"CSV incompleto: {size}/{expected} bytes")
                    with temporary.open("r", encoding="utf-8-sig") as stream:
                        header = set(stream.readline().strip().split(","))
                    if not {"latitude", "longitude", "acq_date", "acq_time", "satellite", "confidence"}.issubset(header):
                        raise ValueError("CSV FIRMS sem cabeçalho esperado.")
                    digest = sha256_file(temporary)
                    replace_with_retry(temporary, target)
                    manifest["files"][name] = {"url": url, "sha256": digest, "bytes": size,
                        "retrieved_utc": utc_now(), "etag": response.headers.get("ETag"),
                        "last_modified": response.headers.get("Last-Modified"),
                        "processing": "NRT; cobertura não estabelecida" if "7d" in name else "arquivo anual por país; cobertura não estabelecida"}
                    write_json(manifest_path, manifest)
                    run.logger.info("firms_download=%s bytes=%d", name, size)
                    break
            except (requests.RequestException, OSError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < 2:
                    time.sleep(2 ** attempt)
        else:
            raise ConnectionError(f"Download FIRMS falhou: {name}: {last_error}")
    session.close()
    run.stage("firms_download", "complete", files=len(manifest["files"]))
    return manifest


def modis_date(day: date) -> str:
    return f"A{day.year}{day.timetuple().tm_yday:03d}"


def mask_plan(config: dict[str, Any]) -> list[dict[str, Any]]:
    """No máximo dez imagens por consulta, como exigido pelo serviço ORNL."""
    mask = config["mask"]
    queries = []
    for region in config["regions"]:
        for year in range(mask["start_year"], mask["end_year"] + 1):
            first, last = date(year, 1, 1), date(year, 12, 31)
            for offset in range(0, 365, 80):
                begin = first + timedelta(days=offset)
                finish = min(last, begin + timedelta(days=72))
                queries.append({"region": region, "start": modis_date(begin), "end": modis_date(finish)})
        queries.append({"region": region, "start": "A2026241", "end": "A2026305"})
    return queries


def split_mask_query(query: dict[str, Any]) -> list[dict[str, Any]]:
    begin = datetime.strptime(query["start"], "A%Y%j").date()
    finish = datetime.strptime(query["end"], "A%Y%j").date()
    count = (finish - begin).days // 8 + 1
    if count <= 1:
        return []
    first_count = count // 2
    return [{"region": query["region"], "start": modis_date(begin),
             "end": modis_date(begin + timedelta(days=8 * (first_count - 1)))},
            {"region": query["region"], "start": modis_date(begin + timedelta(days=8 * first_count)),
             "end": query["end"]}]


def _mask_summary(envelope: dict[str, Any], region: dict[str, Any]) -> list[dict[str, Any]]:
    body = envelope["body"]
    if not isinstance(body, dict) or not isinstance(body.get("subset"), list):
        raise ValueError("Resposta ORNL sem lista de imagens.")
    nrows, ncols = int(body["nrows"]), int(body["ncols"])
    total = nrows * ncols
    if not total or body.get("band") != "FireMask" or body.get("units") != "class-flag":
        raise ValueError("Banda ou geometria inesperada na máscara MODIS.")
    results = []
    for entry in body["subset"]:
        values = entry["data"]
        if len(values) != total or any(not isinstance(x, int) or x < 0 or x > 9 for x in values):
            raise ValueError("Máscara MODIS inconsistente com a geometria ou classes.")
        counts = Counter(values)
        valid = sum(counts[x] for x in VALID_MASK_CLASSES)
        fire = sum(counts[x] for x in FIRE_MASK_CLASSES)
        start = date.fromisoformat(entry["calendar_date"])
        next_year = date(start.year + 1, 1, 1)
        end = min(start + timedelta(days=8), next_year)
        results.append({"region_id": region_id(region["latitude"], region["longitude"]),
            "region_name": region["name"], "spatial_group": region["group"],
            "latitude": region["latitude"], "longitude": region["longitude"],
            "period_start": start.isoformat(), "period_end_exclusive": end.isoformat(),
            "modis_date": entry["modis_date"], "tile": entry.get("tile"), "proc_date": entry.get("proc_date"),
            "total_pixels": total, "valid_pixels": valid, "fire_pixels": fire,
            "cloud_pixels": counts[4], "unknown_pixels": counts[0]+counts[1]+counts[2]+counts[6],
            "valid_fraction": valid / total, "fully_observed": valid == total,
            "event": int(fire > 0) if valid == total else None,
            "xllcorner": float(body["xllcorner"]), "yllcorner": float(body["yllcorner"]),
            "nrows": nrows, "ncols": ncols, "cellsize_m": float(body["cellsize"]),
            "cache_sha256": envelope["body_sha256"], "retrieved_utc": envelope["retrieved_utc"]})
    return results


def download_masks(run: Run) -> dict[str, Any]:
    """Consulta em série, salva cada resposta e retoma somente com hashes íntegros."""
    folder = run.path / "coverage"
    cache = folder / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / "manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {"requests": 0, "files": {}}
    run.stage("coverage", "running", requests=manifest["requests"])
    session = requests.Session()
    all_rows: list[dict[str, Any]] = []
    plan = mask_plan(run.config)
    index = 0
    while index < len(plan):
        query = plan[index]
        run.guard()
        region = query["region"]
        params = {"latitude": region["latitude"], "longitude": region["longitude"], "band": "FireMask",
                  "startDate": query["start"], "endDate": query["end"],
                  "kmAboveBelow": run.config["mask"]["radius_km"], "kmLeftRight": run.config["mask"]["radius_km"]}
        key = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()
        if key in manifest.get("subdivided", {}):
            children = split_mask_query(query)
            if not children:
                raise ValueError("Consulta indivisível marcada para subdivisão.")
            plan[index:index + 1] = children
            continue
        path = cache / (key + ".json")
        item = manifest["files"].get(key)
        if path.exists():
            if not item or sha256_file(path) != item["sha256"]:
                raise ValueError(f"Cache de cobertura incompatível: {path}")
            envelope = read_json(path)
        else:
            error = "sem resposta"
            for attempt in range(min(3, run.config["operational"]["network_retries"])):
                run.guard()
                if manifest["requests"] >= run.config["mask"]["max_requests"]:
                    raise ResourceLimit("Limite de requisições ORNL atingido.")
                delay = run.config["mask"]["min_interval_seconds"] - (time.time() - manifest.get("last_request_epoch", 0))
                if delay > 0:
                    time.sleep(delay)
                manifest["requests"] += 1
                manifest["last_request_epoch"] = time.time()
                write_json(manifest_path, manifest)
                try:
                    response = session.get(ORNL_URL, params=params, timeout=run.config["operational"]["network_timeout_seconds"])
                    if response.status_code == 429 or response.status_code >= 500:
                        error = f"HTTP {response.status_code}"
                        retry_after = response.headers.get("Retry-After", "")
                        if retry_after.isdigit() and int(retry_after) > 30:
                            raise ResourceLimit(f"ORNL solicitou esperar {retry_after}s; operação interrompida.")
                        if attempt < 2:
                            time.sleep(max(2 ** attempt, int(retry_after) if retry_after.isdigit() else 0))
                        continue
                    if 400 <= response.status_code < 500:
                        raise ValueError(f"ORNL recusou a consulta: HTTP {response.status_code}")
                    response.raise_for_status()
                    body = response.json()
                    if not isinstance(body, dict):
                        raise ValueError(f"ORNL retornou erro: {str(body)[:160]}")
                    envelope = {"source": ORNL_URL, "query": params, "retrieved_utc": utc_now(),
                                "body_sha256": hashlib.sha256(response.content).hexdigest(), "body": body}
                    _mask_summary(envelope, region)
                    write_json(path, envelope)
                    manifest["files"][key] = {"sha256": sha256_file(path), "body_sha256": envelope["body_sha256"],
                                              "region": region["name"], "start": query["start"], "end": query["end"]}
                    write_json(manifest_path, manifest)
                    break
                except (requests.RequestException, OSError) as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    if attempt < 2:
                        time.sleep(2 ** attempt)
            else:
                children = split_mask_query(query)
                if not children:
                    raise ConnectionError(f"Consulta ORNL falhou após subdivisão: {error}")
                manifest.setdefault("subdivided", {})[key] = {"reason": error, "start": query["start"], "end": query["end"]}
                write_json(manifest_path, manifest)
                plan[index:index + 1] = children
                run.logger.info("coverage_subdivided=%s/%s reason=transient_failure", query["start"], query["end"])
                continue
        all_rows.extend(_mask_summary(envelope, region))
        index += 1
        if index % 10 == 0 or index == len(plan):
            run.logger.info("coverage_progress=%d/%d requests=%d", index, len(plan), manifest["requests"])
            run.stage("coverage", "running", completed_queries=index, total_queries=len(plan), requests=manifest["requests"])
    session.close()
    frame = pd.DataFrame(all_rows)
    if frame.duplicated(["region_id", "period_start"]).any():
        raise ValueError("Composições duplicadas na cobertura.")
    frame["event"] = frame["event"].astype("Int64")
    frame.to_parquet(folder / "mask_summary.parquet", index=False)
    actual_periods = set(zip(frame.region_id, frame.period_start))
    missing_periods = []
    for region in run.config["regions"]:
        rid = region_id(region["latitude"], region["longitude"])
        for year in range(run.config["mask"]["start_year"], run.config["mask"]["end_year"] + 1):
            day, last = date(year, 1, 1), date(year, 12, 31)
            while day <= last:
                if (rid, day.isoformat()) not in actual_periods:
                    missing_periods.append({"region_id": rid, "period_start": day.isoformat(), "state": "missing_unknown"})
                day += timedelta(days=8)
    write_json(folder / "missing_periods.json", {"periods": missing_periods, "count": len(missing_periods)})
    summary = {"product": "MOD14A2", "collection": "não informada pela resposta de recorte ORNL",
        "source": ORNL_URL, "queries": len(plan),
        "requests": manifest["requests"], "periods": len(frame), "fully_observed": int(frame.fully_observed.sum()),
        "positive_fully_observed": int((frame.event == 1).sum()), "unknown": int(frame.event.isna().sum()),
        "missing_requested_periods_2018_2025": len(missing_periods),
        "output_sha256": sha256_file(folder / "mask_summary.parquet"),
        "limitation": "Composto MODIS/Terra de 8 dias; não informa cobertura de VIIRS nem garante fogo ausente entre passagens."}
    write_json(folder / "summary.json", summary)
    run.stage("coverage", "complete", **summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="NASA FIRMS e máscaras MODIS para o experimento v3")
    parser.add_argument("--config", type=Path, default=Path("config/nasa_complement_v3.json"))
    parser.add_argument("--run-id", default="nasa_complement_v3_20261111")
    parser.add_argument("stage", choices=("firms", "masks"))
    args = parser.parse_args()
    run = Run(args.config, args.run_id)
    output = download_firms(run) if args.stage == "firms" else download_masks(run)
    print(json.dumps(output if args.stage == "masks" else {"files": list(output["files"])}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
