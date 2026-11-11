"""Simulação condicional reutilizável, sem retreinar o modelo a cada consulta."""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from . import features as feature_module, model as model_module
from .features import WEATHER, temporal_features, weekly_weather
from .ingest import region_id
from .model import posterior_counts
from .power import EXPECTED_UNITS, NasaPowerClient, WeatherUnavailable, grid_center, power_frame
from .io import read_json, sha256_file


def _forecast(path: Path, origin: pd.Timestamp, week: pd.Timestamp, lat: float, lon: float) -> tuple[dict[str, float], dict[str, Any]]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    issue = pd.Timestamp(doc["issued_at"])
    if issue.tzinfo is None or issue.tz_convert("UTC") > origin.tz_localize("UTC"):
        raise ValueError("Previsão meteorológica precisa de emissão UTC anterior ao instante da simulação.")
    if not doc.get("source") or doc.get("units") != EXPECTED_UNITS:
        raise ValueError("Previsão meteorológica sem fonte ou unidades compatíveis.")
    location = doc.get("location", {})
    if "latitude" not in location or "longitude" not in location or grid_center(float(location["latitude"]), float(location["longitude"])) != (lat, lon):
        raise ValueError("Local da previsão meteorológica não corresponde à célula analisada.")
    frame = pd.DataFrame(doc["daily"])
    frame["date"] = pd.to_datetime(frame.date, format="%Y-%m-%d", errors="raise")
    expected = set(pd.date_range(week, periods=7))
    if len(frame) != 7 or set(frame.date) != expected:
        raise ValueError("Previsão deve conter exatamente os sete dias do horizonte.")
    for column in WEATHER:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    for _, daily in frame.iterrows():
        _modify({k: float(daily[k]) for k in WEATHER}, {})
    summarized = weekly_weather(frame)
    if len(summarized) != 1:
        raise ValueError("Previsão contém valores meteorológicos ausentes.")
    return ({x: float(summarized.iloc[0][x]) for x in WEATHER},
            {"source": doc["source"], "issued_at": issue.isoformat(), "sha256": sha256_file(path), "location": location})


def _modify(values: dict[str, float], overrides: dict[str, float]) -> dict[str, float]:
    allowed = {"temperature_delta_c", "relative_humidity_delta_percentage_points", "precipitation_multiplier", "wind_multiplier"}
    if set(overrides) - allowed:
        raise ValueError("Modificação meteorológica desconhecida.")
    result = values.copy()
    delta = float(overrides.get("temperature_delta_c", 0))
    for name in ("T2M", "T2M_MAX", "T2M_MIN"):
        result[name] += delta
    result["RH2M"] += float(overrides.get("relative_humidity_delta_percentage_points", 0))
    result["PRECTOTCORR"] *= float(overrides.get("precipitation_multiplier", 1))
    result["WS10M"] *= float(overrides.get("wind_multiplier", 1))
    if (not all(np.isfinite(list(result.values()))) or
        not (-80 <= result["T2M_MIN"] <= result["T2M"] <= result["T2M_MAX"] <= 60) or
        not 0 <= result["RH2M"] <= 100 or result["PRECTOTCORR"] < 0 or result["WS10M"] < 0):
        raise ValueError("Cenário meteorológico viola limites físicos ou consistência entre temperaturas.")
    return result


def simulate_fire_scenario(latitude: float, longitude: float, reference_date: str, horizon_days: int,
                           spatial_config: dict[str, Any], scenario_overrides: dict[str, float] | None = None,
                           forecast_path: str | Path | None = None, model_version: str | None = None) -> dict[str, Any]:
    """Simula contagem semanal condicional; probabilidade incondicional pode ser indisponível.

    `spatial_config` requer `run_dir` e `power_cache`; a data é um domingo UTC.
    As alterações meteorológicas são aplicadas às entradas, nunca à saída do modelo.
    """
    lat, lon = grid_center(float(latitude), float(longitude))
    rid = region_id(lat, lon)
    ref = date.fromisoformat(reference_date)
    origin = pd.Timestamp(ref) + pd.Timedelta(hours=23, minutes=59, seconds=59)
    week = pd.Timestamp(ref) + pd.Timedelta(days=1)
    result: dict[str, Any] = {
        "schema_version": "1.0", "status": "insufficient_data", "conditional_count_status": "insufficient_data",
        "requested_location": {"latitude": latitude, "longitude": longitude},
        "analysis_area": {"region_id": rid, "center_latitude": lat, "center_longitude": lon,
            "bounds": [lon - .3125, lat - .25, lon + .3125, lat + .25],
            "source_scope": "interseção com registros do arquivo Brazil; fronteira efetiva não fornecida"},
        "emission_utc": origin.isoformat() + "Z", "horizon_days": horizon_days,
        "event_definition": "ao menos uma detecção VIIRS/NOAA-20 na área e semana, caso a completude fosse verificável",
        "event_probability": None, "probability_difference_percentage_points": None,
        "reference_scenario": None, "modified_scenario": None,
        "limitations": ["probabilidade de ocorrência não identificável a partir apenas de registros positivos",
                        "uma detecção não é uma ignição, área queimada nem incêndio independente",
                        "associação condicional do modelo não é efeito causal"],
        "reanalysis_vintage": "dados históricos POWER podem ter sido revisados após a data simulada",
    }
    if horizon_days != 7 or ref.weekday() != 6:
        result["reason"] = "O modelo exige horizonte de 7 dias e emissão no domingo UTC."
        return result
    run = Path(spatial_config["run_dir"])
    model_file = run / "models" / "weather.pt"
    if not model_file.exists():
        result["reason"] = "Modelo meteorológico treinado não encontrado."
        return result
    model = torch.load(model_file, map_location="cpu", weights_only=False)
    run_config = read_json(run / "run.json")["config"]
    countries = run_config.get("source_countries", ["Brazil"])
    result["analysis_area"]["source_scope"] = "registros das partições " + ", ".join(countries) + "; fronteira efetiva não fornecida"
    result["model_version"] = model["model_version"]
    result["preprocessing_version"] = model["preprocessing_version"]
    result["model_sha256"] = sha256_file(model_file)
    result["model_code_sha256"] = model.get("model_code_sha256")
    result["features_code_sha256"] = model.get("features_code_sha256")
    result["training_examples_sha256"] = model.get("data_sha256")
    result["data_quality"] = {"fire_observation_coverage": "unknown", "weather_missing_days_used": None,
                              "training_scope": "células e partições configuradas; não é modelo global"}
    metrics_path = run / "evaluation/metrics.json"
    if metrics_path.exists():
        metrics = read_json(metrics_path)
        validation = metrics.get("validation", {}).get("weather", {}).get("overall")
        result["validation_quality"] = ({"conditional_n": validation["n"], "conditional_mae": validation["pyro_median"]["mae"],
            "conditional_interval_80_coverage": validation["interval_80"]["coverage"],
            "conditional_interval_95_coverage": validation["interval_95"]["coverage"],
            "selected_forecast_variant": metrics.get("selection", {}).get("variant")} if validation else None)
    else:
        result["validation_quality"] = None
    if model_version and model_version != model["model_version"]:
        result["reason"] = "Versão solicitada não corresponde ao modelo encontrado."
        return result
    if not model.get("complete") or rid not in model["regions"] or ref <= date.fromisoformat(model["training_end"]):
        result["reason"] = "Modelo incompleto, célula não treinada ou emissão anterior ao fim do treino."
        return result
    if model.get("data_sha256") and sha256_file(run / "processed/examples.parquet") != model["data_sha256"]:
        result["reason"] = "Exemplos de treinamento não correspondem ao checkpoint."
        return result
    result["training_history_until"] = model["training_end"]
    if ((model.get("model_code_sha256") and model["model_code_sha256"] != sha256_file(Path(model_module.__file__))) or
        (model.get("features_code_sha256") and model["features_code_sha256"] != sha256_file(Path(feature_module.__file__)))):
        result["reason"] = "O código atual não corresponde às transformações e ao modelo salvos."
        return result
    weekly = pd.read_parquet(run / "processed" / "weekly_positive.parquet")
    previous = week - pd.Timedelta(days=7)
    recent = weekly.loc[(weekly.region_id == rid) & (weekly["product"] == "viirs-jpss1") &
                        (weekly.satellite == "N20") & (weekly.week_start == previous)]
    if recent.empty:
        result["reason"] = "Semana anterior sem registro positivo; ausência não é interpretada como zero."
        return result
    lag = int(recent.detections.sum())
    local_history = weekly.loc[(weekly.region_id == rid) & (weekly["product"] == "viirs-jpss1") &
                               (weekly.satellite == "N20") & (weekly.week_start <= previous)]
    result["local_history"] = {"first_week_with_record": local_history.week_start.min().date().isoformat(),
                               "last_week_used": previous.date().isoformat(), "positive_weeks": len(local_history)}
    try:
        ops = run_config["operational"]
        client = NasaPowerClient(Path(spatial_config["power_cache"]), max_requests=ops["nasa_power_requests"],
                                 timeout=ops["network_timeout_seconds"], retries=ops["network_retries"])
        weather_start = week - pd.Timedelta(days=14)
        weather_end = weather_start + pd.Timedelta(days=6)
        envelope = client.fetch(lat, lon, weather_start.date().isoformat(), weather_end.date().isoformat())
        units = {k: v.get("units") for k, v in envelope["response"].get("parameters", {}).items()}
        if any(units.get(k) != EXPECTED_UNITS[k] for k in WEATHER):
            raise WeatherUnavailable("Unidades meteorológicas incompatíveis com o treino.")
        weather = power_frame(envelope)
        past = weather.loc[(weather.date >= weather_start) & (weather.date <= weather_end)]
        past = weekly_weather(past)
        if len(past) != 1:
            raise WeatherUnavailable("Meteorologia retrospectiva incompleta para a semana anterior.")
        result["data_quality"]["weather_missing_days_used"] = 0
        result["weather_history"] = {"source": "NASA POWER Daily UTC", "start": weather_start.date().isoformat(),
                                     "end": weather_end.date().isoformat(), "retrieved_utc": envelope["retrieved_utc"],
                                     "native_resolution": "0.5° latitude × 0.625° longitude", "units": units,
                                     "response_api": envelope["response"].get("header", {}).get("api"),
                                     "cached_source_request": envelope["request"]}
    except (WeatherUnavailable, OSError) as exc:
        result.update(status="weather_unavailable", reason=str(exc)); return result
    except (KeyError, TypeError, ValueError) as exc:
        result.update(status="weather_unavailable", reason=f"Resposta ou configuração meteorológica inválida: {exc}"); return result
    climate = pd.read_parquet(run / "processed" / "climatology.parquet")
    climate_row = climate.loc[(climate.region_id == rid) & (climate.month == week.month)]
    forecast_metadata = None
    if forecast_path:
        future, forecast_metadata = _forecast(Path(forecast_path), origin, week, lat, lon)
        future_source = "previsão meteorológica externa fornecida, com emissão validada"
    elif len(climate_row) == 1:
        future = {name: float(climate_row.iloc[0][name]) for name in WEATHER}
        future_source = "climatologia sazonal estimada no treino; trajetória hipotética fixa"
    else:
        result["reason"] = "Climatologia do treino indisponível para região e mês."
        return result
    future = _modify(future, {})
    modified = _modify(future, scenario_overrides or {})

    def row_for(values: dict[str, float]) -> pd.DataFrame:
        log_lag1, season_sin, season_cos = temporal_features(lag, week.dayofyear)
        row: dict[str, Any] = {"log_lag1": log_lag1, "season_sin": season_sin, "season_cos": season_cos}
        row.update({"past_" + x: float(past.iloc[0][x]) for x in WEATHER})
        row.update({"future_" + x: values[x] for x in WEATHER})
        row.update({"region=" + x: float(x == rid) for x in model["regions"][1:]})
        return pd.DataFrame([row])

    def in_range(frame: pd.DataFrame) -> tuple[bool, list[str]]:
        raw = frame[model["features"]].to_numpy()[0]
        low, high = np.array(model["feature_min"]), np.array(model["feature_max"])
        scale = np.array(model["feature_scale"])
        bad = [model["features"][i] for i in np.where((raw < low - scale) | (raw > high + scale))[0]]
        return not bad, bad

    def estimate(frame: pd.DataFrame, values: dict[str, float]) -> dict[str, Any]:
        valid, bad = in_range(frame)
        if not valid:
            return {"status": "out_of_distribution", "outside_training_features": bad,
                    "weather": values, "event_probability": None, "expected_count_conditional": None,
                    "median_count_conditional": None, "predictive_interval_80": None, "predictive_interval_95": None,
                    "unit": "detecções/semana, condicionadas à semana alvo e à anterior positivas"}
        draws, _ = posterior_counts(model_file, frame, n_samples=600, seed=1234)
        return {"status": "ok", "weather": values, "event_probability": None, "expected_count_conditional": float(draws.mean()),
                "median_count_conditional": float(np.quantile(draws, .5)),
                "predictive_interval_80": np.quantile(draws, [.1, .9]).tolist(),
                "predictive_interval_95": np.quantile(draws, [.025, .975]).tolist(),
                "unit": "detecções/semana, condicionadas à semana alvo e à anterior positivas"}

    baseline, altered = estimate(row_for(future), future), estimate(row_for(modified), modified)
    result.update({"conditional_count_status": "ok" if baseline["status"] == altered["status"] == "ok" else "out_of_distribution",
                   "weather_trajectory": {"source": future_source, "valid_from": week.date().isoformat(),
                                          "valid_to": (week + pd.Timedelta(days=6)).date().isoformat(),
                                          "temperature": "°C", "humidity": "%", "precipitation": "mm/semana", "wind": "m/s",
                                          "forecast_metadata": forecast_metadata},
                   "previous_week_detections": lag, "reference_scenario": baseline, "modified_scenario": altered,
                   "expected_count_difference_conditional": (altered["expected_count_conditional"] - baseline["expected_count_conditional"]
                       if baseline["status"] == altered["status"] == "ok" else None),
                   "scenario_overrides": scenario_overrides or {},
                   "uncertainty": "intervalos preditivos incluem dispersão observacional e amostras de parâmetros, condicionados à trajetória fixa; não são intervalos de probabilidade de ocorrência"})
    if result["conditional_count_status"] == "out_of_distribution":
        result["status"] = "out_of_distribution"
    else:
        result["reason"] = "Contagem condicional disponível; probabilidade de ocorrência sem identificação ou calibração."
    return result
