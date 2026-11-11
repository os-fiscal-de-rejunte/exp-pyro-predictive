"""Complementa 2023 com datas fixas após diagnóstico de tamanho de validação."""
from __future__ import annotations

import argparse
import json
import os
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from .io import read_json, sha256_file, utc_now, write_json
from .runtime import Run
from .viirs_coverage import DATA_HOST, NAME, download_with_gaps

MONTHS = (2, 5, 8, 11)


def collect(run: Run, token: str) -> dict[str, Any]:
    if not token:
        raise ValueError("Token Earthdata não fornecido no ambiente.")
    base = run.path / "viirs_coverage"
    primary = read_json(base / "daily_progress.json")
    if primary["config_sha256"] != run.config_hash:
        raise ValueError("A coleta principal tem configuração diferente.")
    output = base / "extra_validation_progress.json"
    progress = read_json(output) if output.exists() else {"schema_version": "1.0", "config_sha256": run.config_hash,
        "months": list(MONTHS), "days_of_month": run.config["viirs_daily"]["days_of_month"],
        "selection_note": "Ampliação de 2023 após contagem agregada preliminar de classes, sem consultar o teste.",
        "partitions": {}, "created_utc": utc_now()}
    if progress["config_sha256"] != run.config_hash or progress["months"] != list(MONTHS):
        raise ValueError("Checkpoint complementar incompatível.")
    run.stage("viirs_validation_extra", "running", partitions_complete=len(progress["partitions"]))
    client = requests.Session()
    try:
        for region in run.config["regions"]:
            key = f"{region['name']}/2023"
            if key in progress["partitions"]:
                continue
            if key not in primary["partitions"]:
                raise ValueError(f"Partição principal ausente: {key}")
            run.guard()
            central = primary["partitions"][key]
            digest = central["cmr_sha256"]
            raw = base / f"cmr_{digest}.json"
            if sha256_file(raw) != digest:
                raise ValueError("Resposta CMR original não confere com o hash.")
            entries = read_json(raw)["feed"]["entry"]
            wanted = {date(2023, month, day).isoformat() for month in MONTHS
                      for day in run.config["viirs_daily"]["days_of_month"]}
            by_date: dict[str, list[dict[str, Any]]] = {}
            for item in entries:
                title, start = item.get("title", ""), item.get("time_start", "")
                if f".{central['tile']}." not in title or start[:10] not in wanted:
                    continue
                url = next((x.get("href", "") for x in item.get("links", []) if x.get("href", "").endswith(".h5")), "")
                parsed = urlparse(url)
                filename = Path(parsed.path).name
                if parsed.scheme != "https" or parsed.netloc != DATA_HOST or not NAME.fullmatch(filename) or filename != title + ".h5":
                    raise ValueError("Endereço CMR complementar inválido.")
                by_date.setdefault(start[:10], []).append({"title": title, "filename": filename, "url": url,
                    "start_utc": start, "end_utc": item.get("time_end"), "size_mb_reported": item.get("granule_size")})
            ambiguous = sorted(day for day, group in by_date.items() if len(group) != 1)
            selected = [group[0] for day, group in by_date.items() if day not in ambiguous]
            unavailable = []
            if selected:
                result = download_with_gaps(run, {"product": "VJ114A1.002", "granules": selected}, token, session=client)
                unavailable = result["unavailable_files"]
            unavailable_days = sorted(x["start_utc"][:10] for x in selected if x["filename"] in unavailable)
            progress["partitions"][key] = {"tile": central["tile"], "expected_days": sorted(wanted),
                "filenames": [x["filename"] for x in selected if x["filename"] not in unavailable],
                "missing_days": sorted((wanted - set(by_date)) | set(unavailable_days)),
                "download_failed_days": unavailable_days, "ambiguous_days": ambiguous,
                "cmr_sha256": digest, "completed_utc": utc_now()}
            write_json(output, progress)
            run.stage("viirs_validation_extra", "running", partitions_complete=len(progress["partitions"]), latest=key)
            print(f"Validação complementar: {key}", flush=True)
    except BaseException:
        run.stage("viirs_validation_extra", "interrupted", partitions_complete=len(progress["partitions"]))
        raise
    finally:
        client.close()
    result = {"status": "ok", "partitions": len(progress["partitions"]),
              "files": sum(len(x["filenames"]) for x in progress["partitions"].values()),
              "missing_days": sum(len(x["missing_days"]) for x in progress["partitions"].values())}
    run.stage("viirs_validation_extra", "complete", **result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/viirs_daily_v4b.json"))
    parser.add_argument("--run-id", default="viirs_daily_v4b_20261111")
    args = parser.parse_args()
    print(json.dumps(collect(Run(args.config, args.run_id), os.environ.get("EARTHDATA_TOKEN", "")),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
