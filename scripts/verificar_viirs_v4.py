"""Verifica artefatos locais reais e integridade espaço-temporal do experimento VIIRS."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthfire.io import read_json, sha256_file, utc_now, write_json  # noqa: E402


def main() -> None:
    run = ROOT / "runs/viirs_daily_v4b_20261111"
    source = run / "viirs_coverage"
    manifest = read_json(source / "manifest.json")
    for filename, item in manifest["files"].items():
        if sha256_file(source / filename) != item["sha256"]:
            raise ValueError(f"Grânulo modificado: {filename}")
    summary = read_json(run / "processed/viirs_daily_summary.json")
    examples_path = run / "processed/mask_examples.parquet"
    if sha256_file(examples_path) != summary["examples_sha256"]:
        raise ValueError("Exemplos não conferem com seu manifesto.")
    data = pd.read_parquet(examples_path)
    known = data.loc[data.event.notna()]
    if not ((known.valid_pixels == known.total_pixels)
            & (known.event.astype(int) == (known.fire_pixels > 0).astype(int))).all():
        raise ValueError("Evento não confere com máscara e classes de foco.")
    if not (data.loc[data.event.isna(), "split"] == "unknown_coverage").all():
        raise ValueError("Cobertura desconhecida foi usada como rótulo.")
    if not ((pd.to_datetime(data.weather_last_date) <= data.period_start - pd.Timedelta(days=3)) &
            (pd.to_datetime(data.viirs_history_end) <= data.period_start - pd.Timedelta(days=2))).all():
        raise ValueError("Clima ou registros VIIRS posteriores ao limite retrospectivo.")
    for col in ("past_modis_produced_at", "past_viirs_mask_produced_at"):
        present = data[col].notna()
        if not (pd.to_datetime(data.loc[present, col]) < data.loc[present, "period_start"]).all():
            raise ValueError(f"Histórico produzido após a emissão: {col}")
    for split, begin, end in (("train", None, "2022-12-31"), ("validation", "2023-01-01", "2023-12-31"),
                              ("test", "2024-01-01", "2025-12-31")):
        part = data.loc[data.split == split]
        if (begin and not (part.period_start >= pd.Timestamp(begin)).all()) or not (part.period_start <= pd.Timestamp(end)).all():
            raise ValueError(f"Divisão temporal inválida: {split}")
    model = run / "models/mask_weather.pt"
    predictions_path = run / "evaluation/viirs_predictions_test.parquet"
    predictions_count = None
    if model.exists() and predictions_path.exists():
        predictions = pd.read_parquet(predictions_path)
        if not (predictions.model_probability.between(0, 1).all() and
                predictions.parameter_lower_80.between(0, 1).all() and
                predictions.parameter_upper_80.between(0, 1).all() and
                (predictions.parameter_lower_80 <= predictions.parameter_upper_80).all()):
            raise ValueError("Probabilidade ou intervalo inválidos.")
        predictions_count = len(predictions)
    result = {"status": "passed", "checked_utc": utc_now(), "granules_rehashed": len(manifest["files"]),
              "examples": len(data), "unknown_coverage": int(data.event.isna().sum()),
              "test_predictions": predictions_count}
    write_json(run / "verification_v4.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
