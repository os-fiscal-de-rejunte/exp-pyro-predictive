"""Inferência do modelo v2 com duas séries de sensor e clima NASA POWER."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .dual_features import FUTURE, candidates, with_climatology
from .dual_model import model_columns, posterior_counts
from .ingest import region_id
from .io import read_json, sha256_file
from .power import PARAMETERS, grid_center
from .scenario import _forecast, _modify
from .sensor_versions import annotate_versions, build_version_index


def predict_week(latitude: float, longitude: float, reference_date: str, run_dir: str | Path,
                 scenario_overrides: dict[str, float] | None = None, forecast_path: str | Path | None = None,
                 variant: str = "selected", n_samples: int = 600) -> dict[str, Any]:
    lat, lon = grid_center(float(latitude), float(longitude))
    rid, ref = region_id(lat, lon), date.fromisoformat(reference_date)
    week = pd.Timestamp(ref) + pd.Timedelta(days=1)
    origin = pd.Timestamp(ref) + pd.Timedelta(hours=23, minutes=59, seconds=59)
    root = Path(run_dir)
    result: dict[str, Any] = {"schema_version": "2.0", "status": "insufficient_data", "conditional_count_status": "insufficient_data",
        "location": {"requested_latitude": latitude, "requested_longitude": longitude, "region_id": rid,
                     "center_latitude": lat, "center_longitude": lon,
                     "bounds": [lon-.3125, lat-.25, lon+.3125, lat+.25], "scope": "registros das partições Brazil no retângulo"},
        "emission_utc": origin.isoformat() + "Z", "target_week_start": week.date().isoformat(), "horizon_days": 7,
        "event_definition": "pelo menos uma detecção VIIRS/NOAA-20 na célula durante a semana",
        "event_probability": None, "probability_difference_percentage_points": None,
        "reference_scenario": None, "modified_scenario": None,
        "limitations": ["a chance de ocorrência não é identificável sem cobertura e rótulos negativos verificáveis",
            "as contagens são condicionadas a semana alvo positiva e à semana anterior positiva em VIIRS e MODIS/Terra",
            "associação meteorológica é condicional e não demonstra causalidade",
            "os arquivos históricos podem ter sido revisados e não comprovam disponibilidade na data simulada"]}
    if ref.weekday() != 6:
        result["reason"] = "A emissão deve ser um domingo UTC; o modelo prevê a semana seguinte."
        return result
    run_manifest = read_json(root / "run.json")
    config = run_manifest["config"]
    if rid not in {region_id(r["latitude"], r["longitude"]) for r in config["regions"]}:
        result["reason"] = "Célula fora do recorte configurado; a generalização não está estabelecida."
        return result
    if ref <= date.fromisoformat(config["split"]["train_end"]):
        result["reason"] = "A simulação deve ser posterior ao último dia de treino."
        return result
    selection = read_json(root / "evaluation_v2/selection.json")
    chosen = selection["variant"] if variant == "selected" else variant
    if scenario_overrides or forecast_path:
        if variant not in ("selected", "weather"):
            raise ValueError("Alterações de clima exigem a variante weather.")
        chosen = "weather"
    if chosen not in ("weather", "no_weather"):
        raise ValueError("Variante de modelo inválida.")
    model_path = root / "models_v2" / f"{chosen}.pt"
    saved = torch.load(model_path, map_location="cpu", weights_only=False)
    examples = root / "processed_v2/examples.parquet"
    if not saved.get("complete") or sha256_file(examples) != saved["data_sha256"]:
        result["reason"] = "Checkpoint incompleto ou dados de treino incompatíveis."
        return result
    result.update(model_version=saved["model_version"], model_sha256=sha256_file(model_path),
                  training_examples_sha256=saved["data_sha256"], training_end=saved["training_end"],
                  model_trained_utc=saved.get("trained_utc"),
                  emission_mode=("retrospective_simulation" if saved.get("trained_utc") and
                                 pd.Timestamp(saved["trained_utc"]) > origin.tz_localize("UTC") else "inference"),
                  selected_forecast_variant=selection["variant"], model_variant_used=chosen,
                  spatial_group="holdout" if rid in config["spatial_holdout_regions"] else "train_regions")
    prep = read_json(root / "processed_v2/preparation.json")
    available = {(x["archive"].split("_")[0], int(x["nominal_year"])) for x in prep["inputs"]}
    previous = week - pd.Timedelta(days=7)
    years = {int((previous + pd.Timedelta(days=i)).year) for i in range(7)}
    if not all((product, y) in available for y in years for product in ("modis", "viirs-jpss1")):
        result["reason"] = "Faltam arquivos nominais para ambos os sensores na semana anterior."
        return result
    fire = pd.read_parquet(root / "processed_v2/weekly_positive.parquet")
    weather = pd.read_parquet(root / "weather_zarr/weekly.parquet")
    rows = candidates(fire, weather, config)
    rows = rows.loc[(rows.region_id == rid) & (rows.week_start == week)].copy()
    if len(rows) != 1:
        result["reason"] = "Histórico anterior de ambos os sensores ou meteorologia defasada insuficiente."
        return result
    training_examples = pd.read_parquet(examples)
    training_examples = training_examples.loc[(training_examples.split == "train") &
        (training_examples.spatial_group == "train_regions")].dropna(subset=model_columns(True))
    raw_records = pd.read_parquet(root / "processed_v2/records.parquet")
    rows, known_versions = annotate_versions(rows, training_examples, build_version_index(raw_records))
    support = rows.iloc[0].sensor_version_support
    def version_text(value: Any) -> str | None:
        return None if pd.isna(value) else str(value)
    result["data_quality"] = {"fire_observation_coverage": "unknown", "sensor_version_support": support,
        "previous_week_modis_terra_versions": version_text(rows.iloc[0].modis_terra_versions),
        "previous_week_viirs_n20_versions": version_text(rows.iloc[0].viirs_n20_versions),
        "training_sensor_versions": known_versions,
        "version_warning": ("Há versão de sensor não observada no treino; os intervalos não incluem incerteza de processamento."
                            if support == "unseen_version" else "Versão da observação indisponível." if support == "unknown" else None)}
    climate = pd.read_parquet(root / "processed_v2/climatology.parquet")
    baseline = with_climatology(rows, climate)
    if baseline[FUTURE].isna().any(axis=None):
        result["status"] = "weather_unavailable"; result["reason"] = "Climatologia do treino indisponível."
        return result
    future_weather = {name: float(baseline.iloc[0]["future_" + name]) for name in PARAMETERS}
    forecast_metadata = None
    if forecast_path is not None:
        future_weather, forecast_metadata = _forecast(Path(forecast_path), origin, week, lat, lon)
    future_weather = _modify(future_weather, {})
    modified_weather = _modify(future_weather, scenario_overrides or {})
    changed = baseline.copy()
    for name in PARAMETERS:
        baseline["future_" + name] = future_weather[name]
        changed["future_" + name] = modified_weather[name]

    def estimate(frame: pd.DataFrame) -> dict[str, Any]:
        raw = frame[saved["features"]].to_numpy(dtype=float)[0]
        low, high, scale = np.array(saved["feature_min"]), np.array(saved["feature_max"]), np.array(saved["feature_scale"])
        indices = np.where((raw < low-scale) | (raw > high+scale))[0]
        bad = [saved["features"][i] for i in indices]
        if bad:
            return {"status": "out_of_distribution", "outside_training_features": bad, "event_probability": None,
                    "expected_count_conditional": None, "median_count_conditional": None,
                    "predictive_interval_80": None, "predictive_interval_95": None}
        draws, _ = posterior_counts(model_path, frame, n_samples=n_samples, seed=1234)
        return {"status": "ok", "outside_training_features": [], "event_probability": None,
                "expected_count_conditional": float(draws.mean()), "median_count_conditional": float(np.quantile(draws, .5)),
                "predictive_interval_80": np.quantile(draws, [.1, .9]).tolist(),
                "predictive_interval_95": np.quantile(draws, [.025, .975]).tolist(),
                "unit": "detecções VIIRS/NOAA-20 por semana, condicionadas à seleção positiva"}
    a, b = estimate(baseline), estimate(changed)
    a["weather"], b["weather"] = future_weather, modified_weather
    both = a["status"] == b["status"] == "ok"
    result.update({"conditional_count_status": "ok" if both else "out_of_distribution", "reference_scenario": a, "modified_scenario": b,
        "expected_count_difference_conditional": b["expected_count_conditional"] - a["expected_count_conditional"] if both else None,
        "scenario_overrides": scenario_overrides or {},
        "weather_trajectory": {"source": "previsão externa fornecida" if forecast_path else "climatologia ajustada somente no treino",
             "forecast_metadata": forecast_metadata, "soil_moisture": "climatologia do treino; mantida fixa nos dois cenários",
             "historical_source": "NASA POWER Zarr UTC, MERRA-2", "past_last_week_start": (week-pd.Timedelta(days=14)).date().isoformat()},
        "previous_week": {"viirs_n20_detections": int(rows.iloc[0].viirs_lag1),
                          "modis_terra_detections": int(rows.iloc[0].modis_terra_lag1)},
        "uncertainty": "intervalos preditivos condicionais para contagens; clima futuro fixo, não uma probabilidade de ocorrência",
        "reason": "Contagem condicional estimada; ocorrência não identificada." if both else "Cenário ou célula fora da faixa de suporte observada no treino."})
    if not both: result["status"] = "out_of_distribution"
    return result
