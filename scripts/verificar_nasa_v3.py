"""Verifica hashes originais e contratos dos artefatos reais da execução v3."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthfire.io import read_json, sha256_file, utc_now, write_json  # noqa: E402


def main() -> None:
    old = read_json(ROOT / "runs/full_20261111/audit_full/inventory.json")
    added = read_json(ROOT / "runs/audit_zarr_v2_20261111/audit_added/inventory.json")
    originals = {x["archive"]: x["sha256"] for x in old["archives"] + added["archives"]}
    originals["global_countries.json"] = old["other_files"][0]["sha256"]
    for name, expected in originals.items():
        if sha256_file(ROOT / "DATA" / name) != expected:
            raise ValueError(f"Arquivo original mudou desde a auditoria: {name}")

    source = ROOT / "runs/nasa_complement_v3_20261111/firms_raw"
    files = read_json(source / "manifest.json")["files"]
    for name, info in files.items():
        if sha256_file(source / name) != info["sha256"]:
            raise ValueError(f"Arquivo NASA mudou desde o download: {name}")

    run = ROOT / "runs/nasa_complement_v3b_20261111"
    summary = read_json(run / "processed/summary.json")
    examples_path = run / "processed/mask_examples.parquet"
    if sha256_file(examples_path) != summary["examples_sha256"]:
        raise ValueError("Exemplos mudaram desde a preparação.")
    examples = pd.read_parquet(examples_path)
    known = examples.loc[examples.event.notna()]
    if not ((known.valid_pixels == known.total_pixels)
            & (known.event.astype(int) == (known.fire_pixels > 0).astype(int))).all():
        raise ValueError("Rótulo de máscara é incompatível com pixels válidos ou focos.")
    if not (examples.loc[examples.event.isna(), "split"] == "unknown_coverage").all():
        raise ValueError("Ausência desconhecida não foi preservada.")
    if not (pd.to_datetime(examples.weather_last_date) < pd.to_datetime(examples.period_start)).all():
        raise ValueError("Clima não é estritamente anterior ao alvo.")
    cuts = {"train": (None, "2023-01-01"), "validation": ("2023-01-01", "2024-01-01"),
            "test": ("2024-01-01", "2026-01-01")}
    for split, (first, end) in cuts.items():
        part = examples.loc[examples.split == split]
        if (first and not (part.period_start >= pd.Timestamp(first)).all()) or not (
                part.period_end_exclusive <= pd.Timestamp(end)).all():
            raise ValueError(f"Separação temporal inválida: {split}")

    selection = read_json(run / "evaluation/selection.json")
    variant = selection["variant"]
    if not selection["locked_before_test"]:
        raise ValueError("Seleção de modelo não foi registrada antes do teste.")
    predictions = pd.read_parquet(run / "evaluation" / f"predictions_test_{variant}.parquet")
    if not predictions.probability_given_full_mask.between(0, 1).all():
        raise ValueError("Probabilidade condicional inválida.")
    if not ((predictions.parameter_lower_80 >= 0) & (predictions.parameter_upper_80 <= 1)
            & (predictions.parameter_lower_80 <= predictions.parameter_upper_80)).all():
        raise ValueError("Intervalo de parâmetros inválido.")
    result = {"status": "passed", "checked_utc": utc_now(), "original_files_rehashed": len(originals),
              "nasa_firms_files_rehashed": len(files), "examples": len(examples),
              "unknown_coverage": int(examples.event.isna().sum()), "test_predictions": len(predictions),
              "selected_variant": variant, "limitations": "Os hashes comparam os arquivos atuais aos manifestos de auditoria."}
    write_json(run / "verification_v3.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
