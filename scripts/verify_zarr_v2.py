"""Verifica os artefatos efetivamente produzidos pelo experimento Zarr v2."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def strict_json(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"Constante JSON não permitida: {value} em {path}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(root: Path) -> dict[str, Any]:
    run = root / "runs/zarr_brasil_v2_20261111"
    export = run / "exports_v2"
    old = strict_json(root / "runs/full_20261111/audit_full/inventory.json")
    added = strict_json(root / "runs/audit_zarr_v2_20261111/audit_added/inventory.json")
    preparation = strict_json(root / "runs/sat_zarr_v2_20261111/processed/preparation.json")
    audited = {x["archive"]: x["sha256"] for x in old["archives"] + added["archives"]}
    prepared = {x["archive"]: x["sha256"] for x in preparation["archive_inventory"]}
    if audited != prepared:
        raise ValueError("Hashes registrados na auditoria e preparação não conferem.")

    cache = run / "weather_zarr/cache"
    manifest = strict_json(cache / "manifest.json")
    for key, info in manifest["objects"].items():
        path = cache / "objects" / key
        if path.stat().st_size != info["bytes"] or sha256(path) != info["sha256"]:
            raise ValueError(f"Objeto Zarr alterado ou incompleto: {key}")

    for path in export.glob("*.json"):
        strict_json(path)
    regions = strict_json(export / "regions.geojson")
    if regions["type"] != "FeatureCollection" or len(regions["features"]) != 16:
        raise ValueError("GeoJSON das regiões incompatível com o recorte.")
    for feature in regions["features"]:
        ring = feature["geometry"]["coordinates"][0]
        if feature["geometry"]["type"] != "Polygon" or len(ring) != 5 or ring[0] != ring[-1]:
            raise ValueError("Polígono de região inválido.")

    calendar = strict_json(export / "calendar.json")["records"]
    for row in calendar:
        if row["observation_state"] == "missing_unknown":
            if row["detections"] is not None:
                raise ValueError("Uma ausência desconhecida foi preenchida como detecção.")
        elif row["observation_state"] != "positive" or row["detections"] <= 0:
            raise ValueError("Registro inesperado no calendário.")

    forecasts = strict_json(export / "retrospective_forecasts.json")["forecasts"]
    if len(forecasts) != 403:
        raise ValueError("Quantidade inesperada de previsões de teste.")
    for row in forecasts:
        lo80, hi80 = row["predictive_interval_80"]
        lo95, hi95 = row["predictive_interval_95"]
        if not (0 < lo95 <= lo80 <= hi80 <= hi95 and row["observed_detections"] > 0):
            raise ValueError("Contagem ou intervalo preditivo inválido.")
        if row["event_probability"] is not None or row["sensor_version_support"] != "unseen_version":
            raise ValueError("Estado de ocorrência ou versão incompatível.")

    examples = pd.read_parquet(run / "processed_v2/examples.parquet")
    future = [column for column in examples.columns if column.startswith("future_")]
    protected = examples.loc[(examples["split"] != "train") | (examples["spatial_group"] == "holdout")]
    if not protected[future].isna().all().all():
        raise ValueError("Clima futuro observado aparece em linhas de teste ou células reservadas.")
    scenario = strict_json(export / "scenario_20251228.json")
    current = strict_json(export / "predict_20261108.json")
    if scenario["event_probability"] is not None or scenario["emission_mode"] != "retrospective_simulation":
        raise ValueError("Simulação retrospectiva incompatível.")
    if current["status"] != "insufficient_data" or current["event_probability"] is not None:
        raise ValueError("Consulta recente deve indicar dados insuficientes.")
    return {
        "status": "passed",
        "archive_hashes_compared_between_audit_and_preparation": len(audited),
        "note": "Comparação de manifestos; os ZIPs não foram novamente lidos nesta verificação.",
        "cache_objects_rehashed": len(manifest["objects"]),
        "cache_bytes_rehashed": sum(v["bytes"] for v in manifest["objects"].values()),
        "json_forecasts": len(forecasts),
        "calendar_rows": len(calendar),
        "unknown_calendar_rows": sum(x["observation_state"] == "missing_unknown" for x in calendar),
        "regions": len(regions["features"]),
        "examples": len(examples),
        "future_weather_protected_rows": len(protected),
        "scenario_and_missing_current_data": "passed",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    result = verify(args.root.resolve())
    destination = args.root / "runs/zarr_brasil_v2_20261111/verification_v2.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
