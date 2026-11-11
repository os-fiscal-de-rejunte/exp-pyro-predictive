"""Contratos pequenos em JSON/GeoJSON para uso por aplicações."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .ingest import region_id
from .io import read_json, write_json
from .runtime import Run


def plot_scenario(result: dict[str, Any], target: Path) -> bool:
    a, b = result.get("reference_scenario"), result.get("modified_scenario")
    if not a or not b or a.get("status") != "ok" or b.get("status") != "ok":
        return False
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, scenario in enumerate((a, b)):
        lower, upper = scenario["predictive_interval_80"]
        ax.vlines(i, lower, upper, color="#264653", linewidth=2)
        ax.scatter([i], [scenario["expected_count_conditional"]], color="#d66b3b", s=45, zorder=3)
    ax.set_xticks([0, 1], ["Referência sazonal", "Cenário modificado"])
    ax.set_ylabel("Detecções / semana, condicionais")
    prefix = "DEMONSTRATIVO — " if str(result.get("model_version", "")).endswith("_DEMO") else ""
    ax.set_title(prefix + "Simulação condicional: média e intervalo preditivo 80%")
    ax.grid(axis="y", alpha=.2); fig.tight_layout()
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=160); plt.close(fig)
    return True


def export(run: Run) -> None:
    target = run.path / "exports"; target.mkdir(parents=True, exist_ok=True)
    features: list[dict[str, Any]] = []
    for r in run.config["regions"]:
        lat, lon = r["latitude"], r["longitude"]
        polygon = [[lon-.3125, lat-.25], [lon+.3125, lat-.25], [lon+.3125, lat+.25],
                   [lon-.3125, lat+.25], [lon-.3125, lat-.25]]
        features.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [polygon]},
                         "properties": {"region_id": region_id(lat, lon), "name": r["name"],
                                        "data_scope": "somente registros das partições " + ", ".join(run.config.get("source_countries", ["Brazil"])) + " dentro do retângulo"}})
    write_json(target / "regions.geojson", {"type": "FeatureCollection", "features": features})
    calendar = pd.read_parquet(run.path / "processed/calendar.parquet")
    calendar["week_start"] = calendar.week_start.dt.strftime("%Y-%m-%d")
    write_json(target / "calendar.json", {"schema_version": "1.0", "run_id": run.run_id,
        "units": "detecções por semana", "missing_semantics": "missing_unknown; nunca zero por ausência",
        "records": json.loads(calendar.to_json(orient="records", date_format="iso"))})
    selection_path = run.path / "evaluation/selection.json"
    if selection_path.exists():
        selected = read_json(selection_path)["variant"]
        predictions = pd.read_parquet(run.path / f"evaluation/predictions_test_{selected}.parquet")
        forecasts = []
        for r in predictions.itertuples():
            forecasts.append({"region_id": r.region_id, "emission_utc": r.origin.isoformat() + "Z",
                "week_start": r.week_start.date().isoformat(), "horizon_days": 7,
                "model_version": "nb_positive_" + selected + "_v1" + ("_DEMO" if run.run_id.startswith("quick_") else ""),
                "state": "retrospective_conditional", "observed_detections": int(r.detections),
                "median": float(r.pyro_median), "mean": float(r.pyro_mean),
                "predictive_interval_80": [float(r.lower_80), float(r.upper_80)],
                "predictive_interval_95": [float(r.lower_95), float(r.upper_95)],
                "unit": "detecções/semana, condicionadas à semana alvo e à anterior positivas"})
        write_json(target / "retrospective_forecasts.json", {"schema_version": "1.0", "run_id": run.run_id,
            "warning": "estimativas retrospectivas condicionais, não probabilidades de ocorrência", "forecasts": forecasts})
    run.stage("export", "complete", directory=str(target))
