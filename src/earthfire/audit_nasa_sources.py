"""Audita integralmente os CSVs públicos FIRMS baixados na atualização v3."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .ingest import parse_datetime
from .io import read_json, sha256_file, write_json
from .nasa_quality import firms_validation_masks
from .runtime import Run


def audit_file(path: Path, product: str, run: Run) -> dict[str, Any]:
    missing: Counter[str] = Counter()
    invalid: Counter[str] = Counter()
    categories: dict[str, Counter[str]] = {name: Counter() for name in ("satellite", "instrument", "confidence", "version", "daynight", "type")}
    hashes = []
    rows, columns = 0, []
    date_min = date_max = None
    lat_min = lon_min = float("inf")
    lat_max = lon_max = float("-inf")
    for frame in pd.read_csv(path, chunksize=50000, dtype=str, keep_default_na=False, encoding="utf-8-sig"):
        run.guard()
        if not columns:
            columns = frame.columns.tolist()
        rows += len(frame)
        for name in columns:
            missing[name] += int(frame[name].str.strip().eq("").sum())
        for name, mask in firms_validation_masks(frame, product).items():
            invalid[name] += int(mask.sum())
        for name in categories:
            if name in frame:
                categories[name].update(frame[name].value_counts().to_dict())
        _, acquired = parse_datetime(frame)
        if acquired.notna().any():
            a, b = acquired.min().date().isoformat(), acquired.max().date().isoformat()
            date_min, date_max = min(date_min, a) if date_min else a, max(date_max, b) if date_max else b
        for key in ("latitude", "longitude"):
            values = pd.to_numeric(frame[key], errors="coerce").dropna()
            if not values.empty:
                if key == "latitude": lat_min, lat_max = min(lat_min, float(values.min())), max(lat_max, float(values.max()))
                else: lon_min, lon_max = min(lon_min, float(values.min())), max(lon_max, float(values.max()))
        hashes.append(pd.util.hash_pandas_object(frame, index=False).to_numpy(dtype=np.uint64))
    combined = np.concatenate(hashes) if hashes else np.array([], dtype=np.uint64)
    combined.sort()
    candidates = np.unique(combined[1:][combined[1:] == combined[:-1]])
    duplicate_count = 0
    if len(candidates):
        actual: Counter[tuple[Any, ...]] = Counter()
        for frame in pd.read_csv(path, chunksize=50000, dtype=str, keep_default_na=False, encoding="utf-8-sig"):
            run.guard()
            selected = np.isin(pd.util.hash_pandas_object(frame, index=False).to_numpy(dtype=np.uint64), candidates)
            actual.update(tuple(row) for row in frame.loc[selected].itertuples(index=False, name=None))
        duplicate_count = sum(count - 1 for count in actual.values())
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size,
            "product": product, "rows": rows, "columns": columns, "physical_type": "CSV string",
            "date_min": date_min, "date_max": date_max,
            "bounds": [lon_min, lat_min, lon_max, lat_max] if rows else None,
            "missing": dict(missing), "invalid": dict(invalid),
            "categories": {k: dict(v) for k,v in categories.items() if v},
            "exact_duplicates_within_file": duplicate_count,
            "limitation": "Datas extremas e arquivo nominal não estabelecem cobertura nem completude."}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/nasa_complement_v3b.json"))
    parser.add_argument("--run-id", default="audit_nasa_v3_20261111")
    args = parser.parse_args()
    run = Run(args.config, args.run_id)
    source = run.root / "runs" / run.config["firms_run"] / "firms_raw"
    manifest = read_json(source / "manifest.json")
    out = run.path / "audit"
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    for name in manifest["files"]:
        path = source / name
        if sha256_file(path) != manifest["files"][name]["sha256"]:
            raise ValueError(f"Fonte alterada após download: {name}")
        product = "modis" if name.startswith("modis") else "viirs-jpss1"
        target = out / (name + ".json")
        if target.exists():
            report = read_json(target)
            if report["sha256"] != manifest["files"][name]["sha256"]:
                raise ValueError("Relatório de auditoria incompatível com fonte.")
        else:
            run.stage("audit", "running", current=name)
            report = audit_file(path, product, run)
            write_json(target, report)
        summary.append({"name": name, "rows": report["rows"], "date_min": report["date_min"],
                        "date_max": report["date_max"], "duplicates": report["exact_duplicates_within_file"]})
    write_json(out / "summary.json", {"files": summary, "complete": True})
    run.stage("audit", "complete", files=len(summary))
    print(summary)


if __name__ == "__main__":
    main()
