"""Adaptador exploratório espacial; preserva o modelo e a cobertura v6 congelados."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import bernoulli

from .expansion_predict import base_result, forecast as validated_forecast, load
from .io import read_json, sha256_file


def forecast(
    latitude: float,
    longitude: float,
    run_dir: str | Path,
    reference_date: str | None = None,
    *,
    allow_exploratory: bool = True,
) -> dict:
    """Usa a rota validada; fora dela aplica o Pyro espacial sem histórico local.

    Não consulta clima nem retreina. A estimativa exploratória é uma saída
    Bernoulli do modelo existente, cuja calibração no local não foi demonstrada.
    """
    root = Path(run_dir)
    result = validated_forecast(latitude, longitude, root, reference_date)
    if not (root / "model_registry.json").exists():
        return result
    registry, catalog, baselines = load(root)
    rid = result["cell"]["id"]
    info = catalog.get(rid)
    result.update(
        is_predictive=True,
        spatial_adapter_version="spatial-exploratory-1.0",
        spatial_adapter_sha256=sha256_file(Path(__file__)),
        location_seen_in_training=rid in baselines["region_month"],
        local_history_available=bool(info and info["training_n"] > 0),
        within_validated_coverage=bool(info and info["release"].get("eligible")),
        locally_validated_for_this_query=result["status"] == "ok",
    )
    if result["status"] == "ok":
        result.update(prediction_type="preditivo_validado", exploratory=False)
        return result
    if not allow_exploratory:
        return result
    issued = (
        pd.Timestamp(reference_date) if reference_date else pd.Timestamp.now(tz="UTC")
    )
    issued = (
        issued.tz_localize("UTC") if issued.tz is None else issued.tz_convert("UTC")
    )
    if issued < pd.Timestamp("2023-01-01", tz="UTC") or issued.year > 2027:
        result.update(
            status="out_of_distribution",
            reason="Emissão fora da janela temporal suportada pelo modelo (2023–2027).",
        )
        return result
    checkpoint = root / "models/mask_no_weather.pt"
    if not checkpoint.exists():
        result["reason"] = (
            "Checkpoint espacial Pyro não disponível; nenhuma estimativa foi inventada."
        )
        return result
    sample_path = root / "processed/mask_examples.parquet"
    if (
        sha256_file(sample_path)
        != read_json(root / "training_sample.json")["sample_sha256"]
    ):
        raise ValueError("Amostra de treinamento incompatível com o modelo.")
    training = pd.read_parquet(
        sample_path, columns=["region_id", "latitude", "longitude"]
    )
    locations = training.drop_duplicates("region_id")
    if locations.empty:
        result["reason"] = "Não há suporte de treinamento espacial registrado."
        return result
    day = issued.date()
    current = day.replace(month=1, day=1) + timedelta(
        days=((day.timetuple().tm_yday - 1) // 8) * 8
    )
    start = min(
        current + timedelta(days=8), day.replace(year=day.year + 1, month=1, day=1)
    )
    end = min(
        start + timedelta(days=8), start.replace(year=start.year + 1, month=1, day=1)
    )
    # Usa a mesma unidade espacial e a representação de ausência do treinamento.
    geometry = base_result(latitude, longitude)["cell"]
    from .expansion_grid import point_cell

    cell = point_cell(float(latitude), float(longitude))
    phase = 2 * np.pi * start.timetuple().tm_yday / 365.2425
    frame = pd.DataFrame(
        [
            {
                "latitude": cell["latitude"],
                "longitude": cell["longitude"],
                "season_sin": np.sin(phase),
                "season_cos": np.cos(phase),
                "past_modis_event": np.nan,
                "past_modis_missing": 1,
                "past_viirs_log_records": np.nan,
                "past_viirs_missing": 1,
            }
        ]
    )
    from .mask_model import posterior_probabilities

    samples = posterior_probabilities(checkpoint, frame, n_samples=400, seed=42)[:, 0]
    if not np.isfinite(samples).all() or ((samples < 0) | (samples > 1)).any():
        raise ValueError("Saída probabilística inválida.")
    lat = np.deg2rad(locations.latitude.to_numpy(float))
    lon = np.deg2rad(locations.longitude.to_numpy(float))
    query_lat, query_lon = np.deg2rad([cell["latitude"], cell["longitude"]])
    hav = (
        np.sin((lat - query_lat) / 2) ** 2
        + np.cos(lat) * np.cos(query_lat) * np.sin((lon - query_lon) / 2) ** 2
    )
    nearest_km = float((2 * 6371.007181 * np.arcsin(np.sqrt(np.clip(hav, 0, 1)))).min())
    outside = [
        name
        for name in ("latitude", "longitude")
        if not float(locations[name].min())
        <= cell[name]
        <= float(locations[name].max())
    ]
    result.pop("reason", None)
    result.update(
        schema_version="spatial-prediction-1.0",
        status="exploratory",
        prediction_type="preditivo_exploratorio",
        exploratory=True,
        in_global_model_training_sample=rid in set(locations.region_id),
        probability=float(samples.mean()),
        probability_interval_95=np.quantile(samples, [0.025, 0.975]).tolist(),
        predictive_event_interval_95=bernoulli.ppf(
            [0.025, 0.975], float(samples.mean())
        )
        .astype(int)
        .tolist(),
        probability_locally_calibrated=False,
        cell=geometry,
        selected_model="pyro_no_weather",
        mode="spatial_extrapolation",
        issued_at=issued.isoformat(),
        valid_from=start.isoformat() + "T00:00:00Z",
        valid_until=end.isoformat() + "T00:00:00Z",
        horizon_days=(end - start).days,
        lead_days=(start - day).days,
        future_observability="unknown",
        weather_context="not_used; no current or future weather inputs",
        local_history_used=False,
        missing_features=["past_modis_event", "past_viirs_log_records"],
        history_missing_representation="Imputação e indicadores de ausência salvos no treinamento; não são zeros observados.",
        nearest_global_training_cell_km=nearest_km,
        outside_training_coordinate_range=outside,
        land_cover_checked=False,
        uncertainty_method="Posterior variacional Pyro; intervalo de parâmetros não cobre todo erro de transferência geográfica.",
        warning="Estimativa preditiva exploratória: a calibração local não foi demonstrada. Não é chance validada de ignição; não houve consulta de clima, vegetação ou cobertura atual.",
        data_assembled_at=registry.get("created_utc"),
        historical_projection=bool(issued < pd.Timestamp(registry["created_utc"]))
        if registry.get("created_utc")
        else True,
    )
    return result
