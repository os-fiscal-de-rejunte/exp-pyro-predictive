"""Cria pacote local verificável; não publica nem envia dados brutos."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthfire.expansion_predict import load
from earthfire.io import read_json, sha256_file, utc_now, write_json


def package(source: Path, target: Path) -> dict:
    source, target = source.resolve(), target.resolve()
    if not source.is_relative_to(ROOT / "runs") or not target.is_relative_to(
        ROOT / "artifacts"
    ):
        raise ValueError("Use origem em runs/ e destino em artifacts/ deste projeto.")
    registry, _, _ = load(source)
    verification = read_json(source / "verification.json")
    if verification["status"] != "passed":
        raise ValueError("A verificação real precisa passar antes do pacote.")
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(
            "Destino já contém arquivos; preserve a versão e escolha outro diretório."
        )
    files = set(registry["artifacts"]) | set(registry["features"])
    files.update(
        [
            "model_registry.json",
            "training_sample.json",
            "processed/mask_examples.parquet",
            "verification.json",
            "evaluation/metrics.json",
            "environment_versions.json",
            "selection.json",
        ]
    )
    for path in (source / "evaluation").rglob("*"):
        if path.is_file() and path.suffix in (".json", ".parquet", ".png", ".svg"):
            files.add(path.relative_to(source).as_posix())
    if (source / "resources.json").exists():
        files.add("resources.json")
    total = sum((source / name).stat().st_size for name in files)
    budget_path = ROOT / "runs/budget_maxcoverage_v6_20261111.json"
    budget = read_json(budget_path)
    snapshot = read_json(source / "resources.json")
    if (
        snapshot["tracked_run_bytes"] + 2 * total + 10_000_000
        > budget["new_files_limit_bytes"]
    ):
        raise RuntimeError(
            "Sem reserva de disco para pacote e checkout de verificação."
        )
    target.mkdir(parents=True, exist_ok=True)
    for name in sorted(files):
        src, dst = source / name, target / name
        if not src.resolve().is_relative_to(source) or not dst.resolve().is_relative_to(
            target
        ):
            raise ValueError("Caminho de artefato fora do pacote.")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        if sha256_file(src) != sha256_file(dst):
            raise OSError("Cópia não confere: " + name)
    write_json(target / "config.json", read_json(source / "run.json")["config"])
    provenance = target / "catalog"
    provenance.mkdir(exist_ok=True)
    shutil.copyfile(
        ROOT / "artifacts/global_v5/catalog/local_inventory.json",
        provenance / "local_inventory.json",
    )
    mask_root = ROOT / "runs/maxcoverage_v6_20261111"
    for name in ("mask_manifest.json", "alignment_verification.json"):
        shutil.copyfile(mask_root / name, provenance / name)
    weather_root = source / "weather"
    weather_cells = {
        path.stem: read_json(path) for path in (weather_root / "cells").glob("*.json")
    }
    write_json(provenance / "weather_cells.json", weather_cells)
    weather_queries = {}
    for path in (weather_root / "daily_api").glob("*/*.json"):
        envelope = read_json(path)
        if "response" not in envelope:
            continue
        weather_queries[path.parent.name] = {
            "request": envelope["request"],
            "retrieved_utc": envelope["retrieved_utc"],
            "header": envelope["response"]["header"],
            "parameters": envelope["response"]["parameters"],
            "source_snapshot_sha256": sha256_file(path),
        }
    write_json(provenance / "weather_queries.json", weather_queries)
    daily_budget = read_json(weather_root / "daily_api/request_budget.json")
    weather_provider = read_json(weather_root / "sparse_provider.json")
    zarr_manifest = weather_root / "cache/manifest.json"
    zarr_requests = (
        read_json(zarr_manifest)["requests"] if zarr_manifest.exists() else 0
    )
    write_json(
        provenance / "provenance.json",
        {
            "created_utc": utc_now(),
            "source_run": source.name,
            "mask_source": "https://modis.ornl.gov/rst/api/v1/MOD14A2/subset",
            "weather_source": "https://power.larc.nasa.gov/api/temporal/daily/point",
            "weather_units": read_json(weather_root / "active_source.json")["units"],
            "weather_time_standard": "UTC",
            "weather_period": ["2019-12-01", "2025-12-31"],
            "weather_daily_requests": daily_budget["requests"],
            "weather_zarr_requests": zarr_requests,
            "weather_failures": weather_provider["failures"],
            "cached_weather_cells": len(weather_cells),
            "raw_cache": "Mantido localmente em runs/; não incluído no Git. Features e hashes estão no pacote.",
        },
    )
    (target / "README.md").write_text(
        """# Artefatos v6 — expansão em células nativas MODIS

Pacote offline de dados derivados e modelos reais. O resultado não é um modelo mundial validado. Consulte o relatório de avaliação e `coverage.geojson` para a área efetivamente sustentada.

O evento é ao menos um pixel MOD14A2 com foco no composto de até oito dias, condicionado a todos os pixels 41×41 terem máscara válida. Não é a probabilidade de ignição nem de todo fogo físico. A projeção futura disponível usa referência sazonal/estática validada; não incorpora uma previsão meteorológica atual.

Na raiz do repositório, após instalar as dependências em ambiente isolado:

```powershell
.\\.venv\\Scripts\\python.exe scripts/testar_expansao_real.py
.\\.venv\\Scripts\\python.exe scripts/testar_expansao_real.py --historical
.\\.venv\\Scripts\\python.exe scripts/verificar_expansao_v6.py --run-dir artifacts/maxcoverage_v6 --skip-originals
```

Sem `DATA/`, a verificação usa `--skip-originals`. Isso não substitui a conferência dos originais, já registrada na execução de origem. As consultas não usam rede ou retreinamento. Coordenadas externas à cobertura retornam `insufficient_data` com probabilidade nula.

`model_registry.json` verifica modelos, features e código. `package_manifest.json` cobre os demais arquivos distribuídos. O calendário preserva ausências e não calcula anomalias com imagens cuja produção é posterior à emissão histórica.
""",
        encoding="utf-8",
    )
    checksums = {
        p.relative_to(target).as_posix(): sha256_file(p)
        for p in sorted(target.rglob("*"))
        if p.is_file()
    }
    manifest = {
        "schema_version": "native41-package-1.0",
        "created_utc": utc_now(),
        "source_run": source.name,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "files": checksums,
        "bytes": sum(p.stat().st_size for p in target.rglob("*") if p.is_file()),
    }
    write_json(target / "package_manifest.json", manifest)
    load(target)
    return {
        "files": len(checksums) + 1,
        "bytes": manifest["bytes"],
        "target": str(target),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir", type=Path, default=ROOT / "runs/maxcoverage_v6b_20261111"
    )
    parser.add_argument(
        "--target", type=Path, default=ROOT / "artifacts/maxcoverage_v6"
    )
    args = parser.parse_args()
    print(package(args.run_dir, args.target))
