"""Audita grânulos VIIRS recentes em uma área já georreferenciada."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .io import read_json, sha256_file, write_json
from .runtime import Run
from .viirs_daily_data import read_mask


def audit(run: Run, region_name: str, start: str, end: str) -> dict[str, Any]:
    base = run.path / "viirs_coverage"
    manifest = read_json(base / "manifest.json")
    ref_path = run.root / "runs/nasa_complement_v3b_20261111/coverage/mask_summary.parquet"
    ref = pd.read_parquet(ref_path)
    selected = ref.loc[ref.region_name == region_name]
    if selected.empty:
        raise ValueError("Área não está nos recortes georreferenciados.")
    geometry = selected.iloc[0].to_dict()
    dates = pd.date_range(start, end, freq="D")
    if not len(dates) or len(dates) > 32:
        raise ValueError("Intervalo vazio ou superior a 32 dias.")
    rows = []
    for filename, item in manifest["files"].items():
        day = item["start_utc"][:10]
        if not start <= day <= end:
            continue
        run.guard()
        path = base / filename
        if sha256_file(path) != item["sha256"]:
            raise ValueError("Grânulo recente não corresponde ao manifesto.")
        rows.append({"region_name": region_name, "region_id": geometry["region_id"],
                     "date": pd.Timestamp(day), **read_mask(path, run.root, geometry, day)})
    frame = pd.DataFrame(rows)
    if frame.empty or frame.date.duplicated().any():
        raise ValueError("Não há grânulos únicos para esse intervalo.")
    out = run.path / "recent_coverage"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "daily_coverage.parquet"
    frame.to_parquet(path, index=False)
    present = set(frame.date.dt.strftime("%Y-%m-%d"))
    missing = [d.strftime("%Y-%m-%d") for d in dates if d.strftime("%Y-%m-%d") not in present]
    result = {"product": "VJ114A1.002", "region_name": region_name, "region_id": geometry["region_id"],
              "start": start, "end": end, "expected_days": len(dates), "granule_days": len(frame),
              "missing_granule_days": missing, "fully_classified_days": int(frame.fully_observed.sum()),
              "unknown_coverage_days": int(frame.event.isna().sum()),
              "days_with_fire_when_fully_classified": int((frame.event == 1).sum()),
              "source_manifest_sha256": sha256_file(base / "manifest.json"), "output_sha256": sha256_file(path),
              "limitation": "Máscara diária VIIRS de 1 km para um recorte; não demonstra cobertura de 375 m ou ausência de fogo entre passagens."}
    write_json(out / "summary.json", result)
    run.stage("viirs_recent_audit", "complete", region=region_name, granule_days=len(frame), missing=len(missing))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/nasa_complement_v3b.json"))
    parser.add_argument("--run-id", default="viirs_oct2026_20261111")
    parser.add_argument("--region", default="centro_norte")
    parser.add_argument("--start", default="2026-10-01")
    parser.add_argument("--end", default="2026-10-22")
    args = parser.parse_args()
    print(json.dumps(audit(Run(args.config, args.run_id), args.region, args.start, args.end),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
