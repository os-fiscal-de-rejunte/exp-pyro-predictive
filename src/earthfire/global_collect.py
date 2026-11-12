"""Coleta MODIS com concorrência pequena, cache verificado e orçamento compartilhado."""

from __future__ import annotations
import hashlib
import json
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
import pandas as pd
import requests
from .io import read_json, sha256_file, utc_now, write_json
from .nasa_complement import ORNL_URL, _mask_summary, modis_date
from .runtime import ResourceLimit, Run


def collect(run: Run) -> dict:
    folder = run.path / "coverage"
    cache = folder / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    mp = folder / "manifest.json"
    manifest = (
        read_json(mp) if mp.exists() else {"files": {}, "requests": 0, "failures": {}}
    )
    lock = threading.Lock()
    stopped = threading.Event()
    workers = int(run.config["operational"].get("network_concurrency", 1))
    if workers not in (1, 2):
        raise ValueError("Concorrência permitida: 1 ou 2.")

    def task(region, year, offset):
        if stopped.is_set():
            return []
        first = date(year, 1, 1) + timedelta(days=offset)
        last = min(date(year, 12, 31), first + timedelta(days=72))
        params = {
            "latitude": region["latitude"],
            "longitude": region["longitude"],
            "band": "FireMask",
            "startDate": modis_date(first),
            "endDate": modis_date(last),
            "kmAboveBelow": run.config["mask"]["radius_km"],
            "kmLeftRight": run.config["mask"]["radius_km"],
        }
        key = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()
        target = cache / (key + ".json")
        if not target.exists():
            for name in run.config.get("mask_cache_runs", []):
                old = run.root / "runs" / name / "coverage"
                path = old / "cache" / (key + ".json")
                if not path.exists() or not (old / "manifest.json").exists():
                    continue
                item = read_json(old / "manifest.json").get("files", {}).get(key)
                if item and sha256_file(path) == item["sha256"]:
                    shutil.copyfile(path, target)
                    with lock:
                        manifest["files"][key] = {
                            "sha256": sha256_file(target),
                            "region": region["name"],
                            "reused_from": name,
                        }
                        write_json(mp, manifest)
                    break
        if target.exists():
            with lock:
                item = manifest["files"].get(key)
            if not item or sha256_file(target) != item["sha256"]:
                raise ValueError("Cache MODIS incompatível.")
            return _mask_summary(read_json(target), region)
        with lock:
            if manifest["failures"].get(key, {}).get("attempts", 0) >= 3:
                return []
        with requests.Session() as session:
            for attempt in range(run.config["operational"]["network_retries"]):
                if stopped.is_set():
                    return []
                run.guard()
                with lock:
                    if manifest["requests"] >= run.config["mask"]["max_requests"]:
                        raise ResourceLimit("Limite ORNL atingido.")
                    time.sleep(
                        max(0, 1 - (time.time() - manifest.get("last_epoch", 0)))
                    )
                    manifest["requests"] += 1
                    manifest["last_epoch"] = time.time()
                    write_json(mp, manifest)
                try:
                    response = session.get(
                        ORNL_URL,
                        params=params,
                        timeout=run.config["operational"]["network_timeout_seconds"],
                    )
                    if response.status_code == 429:
                        stopped.set()
                        raise ResourceLimit(
                            "HTTP 429: coleta interrompida para respeitar o serviço."
                        )
                    response.raise_for_status()
                    envelope = {
                        "source": ORNL_URL,
                        "query": params,
                        "retrieved_utc": utc_now(),
                        "body_sha256": hashlib.sha256(response.content).hexdigest(),
                        "body": response.json(),
                    }
                    rows = _mask_summary(envelope, region)
                    if not rows:
                        raise ValueError("Nenhuma imagem retornada.")
                    write_json(target, envelope)
                    with lock:
                        manifest["files"][key] = {
                            "sha256": sha256_file(target),
                            "region": region["name"],
                            "start": first.isoformat(),
                            "end": last.isoformat(),
                        }
                        manifest["failures"].pop(key, None)
                        write_json(mp, manifest)
                    return rows
                except (requests.RequestException, ValueError) as exc:
                    with lock:
                        manifest["failures"][key] = {
                            "region": region["name"],
                            "start": first.isoformat(),
                            "error_type": type(exc).__name__,
                            "attempts": attempt + 1,
                        }
                        write_json(mp, manifest)
                    if attempt + 1 < run.config["operational"]["network_retries"]:
                        time.sleep(2**attempt)
        return []

    rows = []
    executor = ThreadPoolExecutor(max_workers=workers)
    # Anos intercalados entre continentes permitem salvar um piloto distribuído.
    tasks = [
        executor.submit(task, r, y, o)
        for y in range(
            run.config["mask"]["start_year"], run.config["mask"]["end_year"] + 1
        )
        for r in run.config["regions"]
        for o in range(0, 365, 80)
    ]
    try:
        for i, future in enumerate(as_completed(tasks), 1):
            rows.extend(future.result())
            if i % 10 == 0:
                run.stage(
                    "global_collect",
                    "running",
                    partitions=i,
                    total=len(tasks),
                    workers=workers,
                )
                print(
                    f'Partições {i}/{len(tasks)}; {len(rows)} área-períodos; {manifest["requests"]} novas consultas',
                    flush=True,
                )
    finally:
        stopped.set()
        executor.shutdown(wait=True, cancel_futures=True)
        if rows:
            frame = pd.DataFrame(rows).sort_values(["region_id", "period_start"]).reset_index(drop=True)
            if frame.duplicated(["region_id", "period_start"]).any():
                raise ValueError("Compostos duplicados.")
            frame["event"] = pd.array(frame.event, dtype="Int64")
            frame.to_parquet(folder / "mask_summary.parquet", index=False)
    if not rows:
        raise ValueError("Nenhuma máscara obtida.")
    result = {
        "rows": len(frame),
        "fully_observed": int(frame.fully_observed.sum()),
        "unknown": int(frame.event.isna().sum()),
        "positive": int(frame.event.sum()),
        "failed_queries": len(manifest["failures"]),
        "sha256": sha256_file(folder / "mask_summary.parquet"),
    }
    write_json(folder / "summary.json", result)
    run.stage("global_collect", "complete", **result)
    return result
