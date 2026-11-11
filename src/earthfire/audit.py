"""Auditoria completa, incremental por arquivo e com verificação exata de colisões."""
from __future__ import annotations

import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .ingest import archive_product, csv_chunks, parse_datetime, validation_masks
from .io import read_json, sha256_file, write_json
from .runtime import ResourceLimit, Run


def _hash_rows(frame: pd.DataFrame) -> np.ndarray:
    return pd.util.hash_pandas_object(frame, index=False, categorize=True).to_numpy(dtype=np.uint64)


def _profile_json(path: Path) -> dict[str, Any]:
    doc = json.loads(path.read_text(encoding="utf-8-sig"))
    data = doc.get("data", {})
    dates, places, scenarios = set(), set(), set()
    leaf_count = 0
    for variable, quantiles in data.items():
        if not isinstance(quantiles, dict):
            continue
        for _, ssps in quantiles.items():
            if not isinstance(ssps, dict): continue
            scenarios.update(ssps)
            for _, countries in ssps.items():
                if not isinstance(countries, dict): continue
                places.update(countries)
                for _, series in countries.items():
                    if isinstance(series, dict):
                        dates.update(series)
                        leaf_count += len(series)
    return {"path": str(path), "format": "json", "bytes": path.stat().st_size, "sha256": sha256_file(path),
            "root_keys": list(doc), "metadata": doc.get("metadata"), "data_keys": list(data),
            "leaf_values": leaf_count, "country_codes": len(places), "scenarios": sorted(scenarios),
            "time_min": min(dates) if dates else None, "time_max": max(dates) if dates else None,
            "units_and_variable": "não identificadas no arquivo", "is_daily_weather": False}


def audit_archive(path: Path, fingerprint: str, run: Run, full: bool = True) -> dict[str, Any]:
    product, nominal_year = archive_product(path)
    result: dict[str, Any] = {"path": str(path), "format": "zip/CSV", "bytes": path.stat().st_size,
        "sha256": fingerprint, "product_in_filename": product, "nominal_year": nominal_year,
        "members": [], "processed_all_csv_members": full, "exact_duplicates": None, "row_count": 0}
    hashes: list[np.ndarray] = []
    with zipfile.ZipFile(path) as z:
        infos = [info for info in z.infolist() if not info.is_dir() and info.filename.lower().endswith(".csv")]
        if not full:
            infos = [i for i in infos if i.filename.endswith("_Brazil.csv")]
        result["uncompressed_bytes"] = sum(x.file_size for x in z.infolist())
        result["total_csv_members"] = sum(1 for x in z.infolist() if x.filename.lower().endswith(".csv"))
    for member_index, info in enumerate(infos, start=1):
        run.guard()
        item: dict[str, Any] = {"member": info.filename, "bytes": info.file_size, "crc32": f"{info.CRC:08x}",
                "rows": 0, "columns": [], "physical_type": "CSV text", "missing": {}, "invalid": {},
                "categories": {}, "date_min": None, "date_max": None, "latitude_min": None,
                "latitude_max": None, "longitude_min": None, "longitude_max": None, "outside_nominal_year": 0}
        missing: Counter[str] = Counter(); invalid: Counter[str] = Counter(); categories: dict[str, Counter] = defaultdict(Counter)
        for frame in csv_chunks(path, info.filename):
            run.guard()
            item["columns"] = list(frame.columns)
            item["rows"] += len(frame)
            hashes.append(_hash_rows(frame))
            missing.update({c: int(frame[c].str.strip().eq("").sum()) for c in frame.columns})
            invalid.update({name: int(mask.sum()) for name, mask in validation_masks(frame, product).items()})
            for col in ("satellite", "instrument", "confidence", "version", "daynight", "type"):
                if col in frame:
                    categories[col].update(frame[col].value_counts(dropna=False).to_dict())
            dates = parse_datetime(frame)[0] if {"acq_date", "acq_time"}.issubset(frame.columns) else pd.Series(dtype="datetime64[ns]")
            good_dates = dates.dropna()
            item["outside_nominal_year"] += int((good_dates.dt.year != nominal_year).sum())
            if not good_dates.empty:
                lo, hi = good_dates.min().date().isoformat(), good_dates.max().date().isoformat()
                item["date_min"] = min(lo, item["date_min"] or lo)
                item["date_max"] = max(hi, item["date_max"] or hi)
            for col in ("latitude", "longitude"):
                if col not in frame:
                    continue
                values = pd.to_numeric(frame[col], errors="coerce").dropna()
                if len(values):
                    lo, hi = float(values.min()), float(values.max())
                    item[col+"_min"] = min(lo, item[col+"_min"] if item[col+"_min"] is not None else lo)
                    item[col+"_max"] = max(hi, item[col+"_max"] if item[col+"_max"] is not None else hi)
        item["missing"], item["invalid"] = dict(missing), dict(invalid)
        item["categories"] = {k: dict(v) for k, v in categories.items()}
        result["row_count"] += item["rows"]
        result["members"].append(item)
        if member_index % 20 == 0 or member_index == len(infos):
            run.logger.info("audit archive=%s members=%s/%s rows=%s", path.name, member_index, len(infos), result["row_count"])
    # A tabela de hashes é apenas um índice de candidatos. A contagem usa comparação
    # das linhas originais, portanto colisões não são declaradas duplicatas.
    if hashes:
        combined = np.concatenate(hashes)
        combined.sort()
        repeat = np.unique(combined[1:][combined[1:] == combined[:-1]])
        del combined, hashes
        candidates = set(int(x) for x in repeat)
        row_counts: Counter[tuple[str, ...]] = Counter()
        if candidates:
            for info in infos:
                for frame in csv_chunks(path, info.filename):
                    run.guard()
                    mask = np.isin(_hash_rows(frame), repeat)
                    row_counts.update((tuple(frame.columns), *row) for row in frame.loc[mask].itertuples(index=False, name=None))
        result["exact_duplicates"] = sum(n - 1 for n in row_counts.values())
        result["hash_candidates"] = len(candidates)
    return result


def audit(run: Run, full: bool = True) -> Path:
    output = run.path / ("audit_full" if full else "audit_demo")
    output.mkdir(parents=True, exist_ok=True)
    data = run.root / run.config["data_dir"]
    archives = sorted(data.glob("*.zip"), key=lambda x: (" (1)" in x.name, x.name))
    inventory = []
    by_hash: dict[str, str] = {}
    run.stage("audit_full" if full else "audit_demo", "running", archives=len(archives))
    try:
        for index, archive in enumerate(archives):
            run.guard()
            digest = sha256_file(archive)
            target = output / (archive.stem + ".json")
            if digest in by_hash:
                item = {"path": str(archive), "bytes": archive.stat().st_size, "format": "zip/CSV",
                        "sha256": digest, "duplicate_of": by_hash[digest]}
                write_json(target, item)
            elif target.exists():
                item = read_json(target)
                if item.get("sha256") != digest:
                    raise RuntimeError("Arquivo original mudou; escolha outro run_id.")
                by_hash[digest] = archive.name
            else:
                item = audit_archive(archive, digest, run, full=full)
                write_json(target, item)
                by_hash[digest] = archive.name
            inventory.append({"archive": archive.name, "report": str(target), "duplicate_of": item.get("duplicate_of"),
                              "rows": item.get("row_count"), "sha256": digest})
            write_json(output / "inventory.json", {"complete": False, "processed_archives": index + 1, "archives": inventory})
    except (ResourceLimit, KeyboardInterrupt) as exc:
        run.stage("audit_full" if full else "audit_demo", "interrupted", processed_archives=len(inventory), reason=str(exc))
        raise
    extra = [_profile_json(p) for p in data.glob("*.json")]
    write_json(output / "inventory.json", {"complete": True, "archives": inventory, "other_files": extra})
    run.stage("audit_full" if full else "audit_demo", "complete", inventory=str(output / "inventory.json"))
    return output / "inventory.json"
