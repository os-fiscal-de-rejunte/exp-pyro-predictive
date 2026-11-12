"""Demonstração offline: dados NASA reais, previsão principal e candidato geográfico."""

# ruff: noqa: E402
from pathlib import Path
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from earthfire.geographic_predict import forecast_with_geography
from earthfire.io import read_json, sha256_file, write_json


def main() -> None:
    source = ROOT / "artifacts/geography_demo"
    for name, digest in read_json(source / "manifest.json")["files"].items():
        if sha256_file(source / name) != digest:
            raise ValueError("Dados offline de demonstração alterados.")
    output = ROOT / "runs/geography_demo_offline"
    cache = output / "cache"
    downloads = cache / "downloads/offline_demo"
    downloads.mkdir(parents=True, exist_ok=True)
    for path in (source / "source_csv").iterdir():
        shutil.copy2(path, downloads / path.name)
    for path in (source / "metadata").iterdir():
        shutil.copy2(path, cache / path.name)
    result = forecast_with_geography(
        -21.229167,
        -54.341825,
        ROOT / "artifacts/maxcoverage_v6",
        "2026-11-12T12:00:00Z",
        cache_dir=cache,
        fetch=False,
        geography_model_dir=ROOT / "artifacts/geography_annual_v1",
        view_dir=output / "view",
    )
    write_json(output / "forecast.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
