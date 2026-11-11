"""Reconhece coordenadas e avalia suporte para inferência MODIS de oito dias."""
from __future__ import annotations

import hashlib
import json
import math
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import torch

from .io import read_json, sha256_file, utc_now, write_json
from .mask_features import WEATHER
from .mask_model import posterior_probabilities
from .nasa_complement import ORNL_URL, _mask_summary, modis_date
from .power import NasaPowerClient, WeatherUnavailable, grid_center, power_frame
from .runtime import ResourceLimit, Run


def modis_period_start(day: date) -> date:
    return date(day.year, 1, 1) + timedelta(days=((day.timetuple().tm_yday - 1) // 8) * 8)


def next_period(day: date) -> date:
    start = modis_period_start(day)
    return min(start + timedelta(days=8), date(day.year + 1, 1, 1))


def _mask_for_location(run: Run, latitude: float, longitude: float, start: date) -> dict[str, Any] | None:
    # Recortes já coletados para as regiões exatas podem ser usados sem rede.
    coverage = run.path / "coverage"
    summary_path, data_path = coverage / "summary.json", coverage / "mask_summary.parquet"
    if summary_path.exists() and data_path.exists():
        summary = read_json(summary_path)
        if sha256_file(data_path) != summary["output_sha256"]:
            raise ValueError("Resumo local das máscaras não confere com o manifesto.")
        saved = pd.read_parquet(data_path)
        selected = saved.loc[(saved.latitude == latitude) & (saved.longitude == longitude)
                             & (saved.period_start == start.isoformat())]
        if len(selected) > 1:
            raise ValueError("Mais de uma máscara local para o mesmo recorte e período.")
        if len(selected) == 1:
            row = selected.iloc[0].to_dict()
            if pd.isna(row["event"]):
                row["event"] = None
            return row
    base = run.path / "location_cache"
    base.mkdir(parents=True, exist_ok=True)
    params = {"latitude": latitude, "longitude": longitude, "band": "FireMask", "startDate": modis_date(start),
              "endDate": modis_date(start), "kmAboveBelow": run.config["mask"]["radius_km"],
              "kmLeftRight": run.config["mask"]["radius_km"]}
    key = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()
    target = base / (key + ".json")
    manifest_path = base / "manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {"requests": 0, "files": {}}
    if target.exists():
        item = manifest["files"].get(key)
        if not item or sha256_file(target) != item["sha256"]:
            raise ValueError("Cache de local incompatível.")
        envelope = read_json(target)
    else:
        last = "sem resposta"
        for attempt in range(min(3, run.config["operational"]["network_retries"])):
            run.guard()
            if manifest["requests"] >= 50:
                raise ResourceLimit("Limite de consultas dinâmicas de máscara atingido.")
            delay = 1. - (time.time() - manifest.get("last_request_epoch", 0))
            if delay > 0:
                time.sleep(delay)
            manifest["requests"] += 1
            manifest["last_request_epoch"] = time.time()
            write_json(manifest_path, manifest)
            try:
                response = requests.get(ORNL_URL, params=params, timeout=run.config["operational"]["network_timeout_seconds"])
                if response.status_code == 429 or response.status_code >= 500:
                    last = f"HTTP {response.status_code}"
                    retry_after = response.headers.get("Retry-After", "")
                    if retry_after.isdigit() and int(retry_after) > 30:
                        raise ResourceLimit(f"ORNL solicitou esperar {retry_after}s; operação interrompida.")
                    if attempt < 2: time.sleep(max(2 ** attempt, int(retry_after) if retry_after.isdigit() else 0))
                    continue
                if 400 <= response.status_code < 500:
                    raise ValueError(f"ORNL recusou a consulta: HTTP {response.status_code}")
                response.raise_for_status()
                body = response.json()
                if not isinstance(body, dict):
                    raise ValueError(f"ORNL retornou {str(body)[:150]}")
                envelope = {"source": ORNL_URL, "query": params, "retrieved_utc": utc_now(),
                            "body_sha256": hashlib.sha256(response.content).hexdigest(), "body": body}
                region = {"latitude": latitude, "longitude": longitude, "name": "consulta", "group": "query"}
                _mask_summary(envelope, region)
                write_json(target, envelope)
                manifest["files"][key] = {"sha256": sha256_file(target), "retrieved_utc": envelope["retrieved_utc"]}
                write_json(manifest_path, manifest)
                break
            except (requests.RequestException, OSError) as exc:
                last = f"{type(exc).__name__}: {exc}"
                if attempt < 2: time.sleep(2 ** attempt)
        else:
            raise ConnectionError(f"Máscara indisponível: {last}")
    region = {"latitude": latitude, "longitude": longitude, "name": "consulta", "group": "query"}
    rows = _mask_summary(envelope, region)
    return rows[0] if rows else None


def _validation_support(metrics: dict[str, Any], variant: str, group: str) -> tuple[bool, dict[str, Any]]:
    data = metrics["validation"][variant]["spatial_groups"].get(group, {})
    model, seasonal = data.get("model", {}), data.get("seasonal", {})
    supported = (model.get("n", 0) >= 30 and model.get("positive", 0) >= 5 and
                 model.get("brier", float("inf")) <= seasonal.get("brier", float("-inf")))
    return supported, {"group": group, "n": model.get("n"), "positive": model.get("positive"),
                       "model_brier": model.get("brier"), "seasonal_brier": seasonal.get("brier")}


def predict_location(run: Run, latitude: float, longitude: float, reference_date: str,
                     n_samples: int = 600) -> dict[str, Any]:
    lat_grid, lon_grid = grid_center(float(latitude), float(longitude))
    reference = date.fromisoformat(reference_date)
    target = next_period(reference)
    target_end = min(target + timedelta(days=8), date(target.year + 1, 1, 1))
    emission = pd.Timestamp(reference.isoformat(), tz="UTC") + pd.Timedelta(hours=23, minutes=59, seconds=59)
    result: dict[str, Any] = {"schema_version": "3.0", "status": "insufficient_data",
        "location": {"latitude": latitude, "longitude": longitude, "recognized": True,
                     "power_grid_center": {"latitude": lat_grid, "longitude": lon_grid},
                     "mask_radius_km": run.config["mask"]["radius_km"]},
        "emission_date_utc": reference.isoformat(), "target_start": target.isoformat(),
        "target_end_exclusive": target_end.isoformat(),
        "event": "ao menos um pixel MODIS/Terra MOD14A2 com foco no composto, condicionado à máscara válida em todos os pixels",
        "event_probability": None, "probability_given_full_mask": None,
        "limitations": ["A observabilidade futura não é conhecida; ausência em pixels processados não prova ausência de fogo.",
                        "A máscara MODIS não estabelece cobertura VIIRS.",
                        "Reanálises e produtos históricos podem ter sido revisados."]}
    if emission > pd.Timestamp.now(tz="UTC"):
        result["reason"] = "A emissão ocorre às 23:59:59 UTC da data informada e ainda não aconteceu."
        return result
    if reference != target - timedelta(days=1):
        result["reason"] = "O modelo emite no dia anterior ao próximo composto de oito dias."
        result["next_eligible_emission"] = (target - timedelta(days=1)).isoformat()
        return result
    selection_path = run.path / "evaluation/selection.json"
    metrics_path = run.path / "evaluation/metrics.json"
    if not selection_path.exists() or not metrics_path.exists():
        result["reason"] = "O modelo de máscara ainda não foi treinado e avaliado."
        return result
    variant = read_json(selection_path)["variant"]
    model_path = run.path / "models" / f"mask_{variant}.pt"
    if not model_path.exists():
        result["reason"] = "Checkpoint do modelo não encontrado."
        return result
    saved = torch.load(model_path, map_location="cpu", weights_only=False)
    examples_path = run.path / "processed/mask_examples.parquet"
    if not saved.get("complete") or not examples_path.exists() or sha256_file(examples_path) != saved["data_sha256"]:
        result["reason"] = "Modelo e exemplos de treinamento são incompatíveis."
        return result
    region_match = next((x for x in run.config["regions"] if
                         abs(latitude - x["latitude"]) < 1e-8 and abs(longitude - x["longitude"]) < 1e-8), None)
    group = "train" if region_match and region_match["group"] == "train" else "holdout"
    supported, evidence = _validation_support(read_json(metrics_path), variant, group)
    result.update({"model_version": saved["model_version"], "model_sha256": sha256_file(model_path),
                   "model_trained_utc": saved["trained_utc"], "validation_support": evidence,
                   "spatial_support": "região de treino" if group == "train" else "fora das regiões de treino",
                   "emission_mode": "retrospective_simulation" if pd.Timestamp(saved["trained_utc"]) > emission else "inference"})
    if not supported:
        result["reason"] = "A validação temporal ou espacial não satisfez o critério mínimo de suporte definido antes do teste."
        return result
    previous = modis_period_start(target - timedelta(days=1))
    prior = modis_period_start(previous - timedelta(days=1))
    try:
        mask = _mask_for_location(run, latitude, longitude, prior)
    except (ConnectionError, requests.RequestException, ValueError) as exc:
        result["reason"] = f"Máscara NASA indisponível: {exc}"
        return result
    event = None
    if mask:
        production = pd.to_datetime(mask.get("proc_date"), format="%Y%j%H%M%S", errors="coerce")
        if pd.notna(production) and production < pd.Timestamp(reference) + pd.Timedelta(days=1):
            event = mask["event"]
    result["previous_mask"] = {"period_start": prior.isoformat(), "result": event,
        "valid_fraction": mask["valid_fraction"] if mask else None, "source": "NASA MOD14A2/ORNL",
        "proc_date": mask.get("proc_date") if mask else None}
    result["location"]["area"] = ({"projection": "MODIS sinusoidal", "xllcorner_m": mask["xllcorner"],
        "yllcorner_m": mask["yllcorner"], "pixel_size_m": mask["cellsize_m"], "rows": mask["nrows"],
        "columns": mask["ncols"]} if mask else None)
    last = target - timedelta(days=3)
    first = last - timedelta(days=7)
    row: dict[str, Any] = {"latitude": latitude, "longitude": longitude,
        "season_sin": math.sin(2 * math.pi * target.timetuple().tm_yday / 365.2425),
        "season_cos": math.cos(2 * math.pi * target.timetuple().tm_yday / 365.2425),
        "past_modis_event": float(event) if event is not None else np.nan,
        "past_modis_missing": int(event is None), "past_viirs_log_records": np.nan, "past_viirs_missing": 1}
    if region_match:
        counts = pd.read_parquet(run.path / "processed/viirs_positive_8day.parquet")
        rid = mask["region_id"] if mask else None
        match = counts.loc[(counts.region_id == rid) & (pd.to_datetime(counts.period_start) == pd.Timestamp(prior))]
        if len(match) == 1:
            row["past_viirs_log_records"] = float(np.log1p(match.iloc[0].records))
            row["past_viirs_missing"] = 0
    if variant == "weather":
        try:
            run.guard()
            client = NasaPowerClient(run.root / "runs/power_location_cache_v3", max_requests=run.config["operational"]["nasa_power_requests"])
            envelope = client.fetch(latitude, longitude, first.isoformat(), last.isoformat(), parameters=WEATHER)
            frame = power_frame(envelope)
        except (WeatherUnavailable, ConnectionError) as exc:
            result["status"] = "weather_unavailable"
            result["reason"] = str(exc)
            return result
        expected_units = {"T2M": {"C"}, "RH2M": {"%"}, "PRECTOTCORR": {"mm/day"}, "WS10M": {"m/s"},
                          "GWETROOT": {"1", "dimensionless"}}
        parameters = envelope["response"].get("parameters", {})
        if any(parameters.get(k, {}).get("units") not in acceptable for k,acceptable in expected_units.items()):
            result["status"] = "weather_unavailable"
            result["reason"] = "Unidades POWER não conferem com o treinamento."
            return result
        if len(frame) != 8 or frame[list(WEATHER)].isna().any(axis=None):
            result["status"] = "weather_unavailable"
            result["reason"] = "Oito dias meteorológicos completos não estão disponíveis."
            return result
        for key in WEATHER:
            row["weather_" + key] = float(frame[key].sum() if key == "PRECTOTCORR" else frame[key].mean())
        result["weather"] = {"source": "NASA POWER UTC", "start": first.isoformat(), "end": last.isoformat(),
                             "retrieved_utc": envelope["retrieved_utc"]}
    raw = np.array([row[name] for name in saved["features"]], dtype=float)
    low, high, scale = np.array(saved["feature_min"]), np.array(saved["feature_max"]), np.array(saved["scale"])
    present = np.isfinite(raw)
    ood = np.where(present & ((raw < low - scale) | (raw > high + scale)))[0]
    if len(ood):
        result["status"] = "out_of_distribution"
        result["outside_training_features"] = [saved["features"][i] for i in ood]
        result["reason"] = "Entradas fora do suporte aproximado de treino."
        return result
    probabilities = posterior_probabilities(model_path, pd.DataFrame([row]), n_samples=n_samples, seed=run.config["seed"])[:, 0]
    result.update({"status": "experimental_conditional", "probability_given_full_mask": float(probabilities.mean()),
        "parameter_interval_80": np.quantile(probabilities, [.1, .9]).tolist(),
        "reason": "Estimativa condicional experimental; disponibilidade da futura máscara é desconhecida."})
    return result
