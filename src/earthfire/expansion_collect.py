"""Coleta blocos grandes e salva cada célula canônica em partições rastreáveis."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

import pandas as pd
import requests

from .expansion_grid import split_body
from .io import read_json, replace_with_retry, sha256_file, utc_now, write_json
from .nasa_complement import ORNL_URL, modis_date
from .runtime import ResourceLimit, Run


class BlockCollector:
    def __init__(self, run: Run):
        self.run = run
        assets = run.root / "runs" / run.config.get("mask_storage_run", run.run_id)
        self.base = assets / "blocks"
        self.cache = assets / "mask_cache"
        self.base.mkdir(exist_ok=True)
        self.cache.mkdir(exist_ok=True)
        self.path = assets / "mask_manifest.json"
        self.manifest = (
            read_json(self.path)
            if self.path.exists()
            else {
                "files": {},
                "pending": {},
                "requests": 0,
                "failures": {},
                "partitions": {},
            }
        )
        self.lock = threading.Lock()
        self.stop = threading.Event()

    def fetch(
        self, region: dict, year: int, offset: int, session: requests.Session
    ) -> list[dict]:
        first = date(year, 1, 1) + timedelta(days=offset)
        last = min(date(year, 12, 31), first + timedelta(days=72))
        query = {
            "latitude": region["latitude"],
            "longitude": region["longitude"],
            "band": "FireMask",
            "startDate": modis_date(first),
            "endDate": modis_date(last),
            "kmAboveBelow": 100,
            "kmLeftRight": 100,
        }
        key = hashlib.sha256(json.dumps(query, sort_keys=True).encode()).hexdigest()
        target = self.cache / (key + ".json")
        with self.lock:
            pending = self.manifest["pending"].get(key)
            if pending:
                tmp = self.cache / (key + ".part")
                candidate = target if target.exists() else tmp
                if candidate.exists() and sha256_file(candidate) == pending["sha256"]:
                    if candidate == tmp:
                        replace_with_retry(tmp, target)
                    self.manifest["files"][key] = pending
                    self.manifest["pending"].pop(key)
                    write_json(self.path, self.manifest)
            item = self.manifest["files"].get(key)
            attempts = self.manifest["failures"].get(key, {}).get("attempts", 0)
        if target.exists():
            if not item or sha256_file(target) != item["sha256"]:
                raise ValueError("Cache de bloco alterado.")
            envelope = read_json(target)
            return split_body(envelope["body"], region, envelope["body_sha256"])
        for attempt in range(attempts, 3):
            if self.stop.is_set():
                raise ResourceLimit("Coleta interrompida.")
            self.run.guard(require_training_seconds=1200)
            with self.lock:
                if self.manifest["requests"] >= self.run.config["mask"]["max_requests"]:
                    raise ResourceLimit("Limite ORNL atingido.")
                time.sleep(
                    max(0, 1 - (time.time() - self.manifest.get("last_epoch", 0)))
                )
                self.manifest["requests"] += 1
                self.manifest["last_epoch"] = time.time()
                write_json(self.path, self.manifest)
            started = time.monotonic()
            try:
                response = session.get(
                    ORNL_URL,
                    params=query,
                    timeout=self.run.config["operational"]["network_timeout_seconds"],
                )
                if response.status_code == 429:
                    self.stop.set()
                    raise ResourceLimit("NASA ORNL retornou HTTP 429; coleta parada.")
                response.raise_for_status()
                envelope = {
                    "query": query,
                    "source": ORNL_URL,
                    "retrieved_utc": utc_now(),
                    "body_sha256": hashlib.sha256(response.content).hexdigest(),
                    "body": response.json(),
                }
                rows = split_body(envelope["body"], region, envelope["body_sha256"])
                if not rows:
                    raise ValueError("Bloco sem células completas.")
                if any(
                    not first <= date.fromisoformat(row["period_start"]) <= last
                    for row in rows
                ):
                    raise ValueError("Data fora da consulta.")
                payload = (
                    json.dumps(envelope, ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                ).encode()
                item = {
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "query": query,
                    "body_sha256": envelope["body_sha256"],
                    "retrieved_utc": envelope["retrieved_utc"],
                    "seconds": time.monotonic() - started,
                }
                with self.lock:
                    self.manifest["pending"][key] = item
                    write_json(self.path, self.manifest)
                tmp = self.cache / (key + ".part")
                tmp.write_bytes(payload)
                replace_with_retry(tmp, target)
                with self.lock:
                    self.manifest["files"][key] = item
                    self.manifest["pending"].pop(key, None)
                    self.manifest["failures"].pop(key, None)
                    write_json(self.path, self.manifest)
                return rows
            except (requests.RequestException, ValueError) as exc:
                with self.lock:
                    self.manifest["failures"][key] = {
                        "attempts": attempt + 1,
                        "error_type": type(exc).__name__,
                        "query": query,
                    }
                    write_json(self.path, self.manifest)
                if attempt < 2:
                    time.sleep(2**attempt)
        return []

    def parent(self, parent: dict) -> dict:
        target = self.base / (parent["name"] + ".parquet")
        with self.lock:
            old = self.manifest["partitions"].get(parent["name"])
        if old:
            if not target.exists() or sha256_file(target) != old["sha256"]:
                raise ValueError("Partição de bloco alterada.")
            return old
        rows = []
        with requests.Session() as session:
            for year in (2020, 2021, 2022):
                for offset in range(0, 365, 80):
                    rows.extend(self.fetch(parent, year, offset, session))
            if not rows:
                return {"parent": parent["name"], "status": "no_data", "cells": 0}
            training = pd.DataFrame(rows)
            training["event"] = pd.array(training.event, dtype="Int64")
            grouped = training.groupby("region_id")
            support = grouped.event.agg(["count", "sum"])
            water = grouped.water_pixels.mean() / (41 * 41)
            eligible = support.index[
                (support["count"] >= 60)
                & (support["sum"] >= 3)
                & ((support["count"] - support["sum"]) >= 3)
                & (water <= 0.2)
            ].tolist()
            if eligible:
                for year in (2023, 2024, 2025):
                    for offset in range(0, 365, 80):
                        rows.extend(self.fetch(parent, year, offset, session))
        frame = (
            pd.DataFrame(rows)
            .sort_values(["region_id", "period_start"])
            .reset_index(drop=True)
        )
        frame["event"] = pd.array(frame.event, dtype="Int64")
        frame["train_support"] = frame.region_id.isin(eligible)
        frame["legacy_context"] = bool(parent["legacy_context"])
        if frame.duplicated(["region_id", "period_start"]).any():
            raise ValueError("Célula-data duplicada no bloco.")
        frame.to_parquet(target, index=False, row_group_size=276)
        item = {
            "parent": parent["name"],
            "status": "complete" if eligible else "no_train_support",
            "cells": int(frame.region_id.nunique()),
            "supported_cells": len(eligible),
            "rows": len(frame),
            "sha256": sha256_file(target),
            "finished_utc": utc_now(),
        }
        with self.lock:
            self.manifest["partitions"][parent["name"]] = item
            write_json(self.path, self.manifest)
        print(
            f"Bloco {parent['name']}: {item['cells']} células, {len(eligible)} com suporte de treino",
            flush=True,
        )
        return item

    def wave(self, parents: list[dict]) -> list[dict]:
        result = []
        executor = ThreadPoolExecutor(max_workers=2)
        futures = [executor.submit(self.parent, parent) for parent in parents]
        try:
            for future in as_completed(futures):
                result.append(future.result())
        finally:
            if any(not f.done() or f.exception() for f in futures):
                self.stop.set()
            executor.shutdown(wait=True, cancel_futures=True)
        return sorted(result, key=lambda r: r["parent"])
