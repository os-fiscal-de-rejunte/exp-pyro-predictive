"""Contratos para a aplicação web do experimento v2."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .ingest import region_id
from .io import read_json, write_json
from .runtime import Run


def export(run: Run) -> Path:
    out = run.path / "exports_v2"; out.mkdir(parents=True, exist_ok=True)
    features = []
    for r in run.config["regions"]:
        lat, lon = r["latitude"], r["longitude"]
        rid = region_id(lat, lon)
        ring = [[lon-.3125,lat-.25],[lon+.3125,lat-.25],[lon+.3125,lat+.25],[lon-.3125,lat+.25],[lon-.3125,lat-.25]]
        features.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring]},
            "properties": {"region_id": rid, "name": r["name"],
                           "spatial_group": "holdout" if rid in run.config["spatial_holdout_regions"] else "train_regions",
                           "scope": "registros Brazil dentro do retângulo; fronteira efetiva não fornecida"}})
    write_json(out / "regions.geojson", {"type": "FeatureCollection", "features": features})
    calendar = pd.read_parquet(run.path / "processed_v2/calendar.parquet")
    calendar["week_start"] = calendar.week_start.dt.strftime("%Y-%m-%d")
    write_json(out / "calendar.json", {"schema_version": "2.0", "run_id": run.run_id,
        "missing_semantics": "missing_unknown; ausência não convertida a zero",
        "records": json.loads(calendar.to_json(orient="records", date_format="iso"))})
    selected = read_json(run.path / "evaluation_v2/selection.json")["variant"]
    predicted = pd.read_parquet(run.path / f"evaluation_v2/predictions_test_{selected}.parquet")
    items = []
    for row in predicted.itertuples():
        items.append({"region_id": row.region_id, "spatial_group": row.spatial_group,
            "emission_utc": row.origin.isoformat() + "Z", "week_start": row.week_start.date().isoformat(),
            "horizon_days": 7, "model_version": f"dual_nb_positive_{selected}_v2",
            "status": "retrospective_conditional", "event_probability": None,
            "sensor_version_support": row.sensor_version_support,
            "modis_terra_versions": None if pd.isna(row.modis_terra_versions) else str(row.modis_terra_versions),
            "viirs_n20_versions": None if pd.isna(row.viirs_n20_versions) else str(row.viirs_n20_versions),
            "observed_detections": int(row.detections), "median": float(row.pyro_median), "mean": float(row.pyro_mean),
            "predictive_interval_80": [float(row.lower_80), float(row.upper_80)],
            "predictive_interval_95": [float(row.lower_95), float(row.upper_95)],
            "unit": "detecções VIIRS/NOAA-20 por semana, condicionadas à seleção positiva"})
    write_json(out / "retrospective_forecasts.json", {"schema_version": "2.0", "run_id": run.run_id,
        "warning": "contagens condicionais retrospectivas; não são probabilidades de ocorrência", "forecasts": items})
    write_json(out / "harmonization.json", read_json(run.path / "harmonization_v2/report.json"))
    harmonic_metrics = run.path / "harmonization_v2/evaluation.json"
    if harmonic_metrics.exists():
        write_json(out / "harmonization_metrics.json", read_json(harmonic_metrics))
    run.stage("export_v2", "complete", directory=str(out), forecasts=len(items))
    return out
