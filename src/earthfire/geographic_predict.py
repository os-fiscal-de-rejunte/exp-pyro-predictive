"""Consulta preditiva com contexto AppEEARS e modelo geográfico opcional."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import bernoulli

from .geography_view import describe_geography, export_view
from .io import read_json, sha256_file
from .spatial_predict import forecast


def forecast_with_geography(
    latitude: float,
    longitude: float,
    run_dir: Path,
    reference_date: str | None = None,
    *,
    cache_dir: Path = Path("runs/geography_cache"),
    view_dir: Path | None = None,
    geography_model_dir: Path | None = None,
    fetch: bool = True,
    allow_exploratory: bool = True,
) -> dict:
    issued = reference_date or pd.Timestamp.now(tz="UTC").isoformat()
    result = forecast(
        latitude, longitude, run_dir, issued, allow_exploratory=allow_exploratory
    )
    if result["status"] == "out_of_distribution" and result.get("probability") is None:
        result.update(
            geography_model_used=False,
            primary_prediction_uses_geography=False,
            geographic_context={
                "status": "not_requested",
                "reason": "Emissão fora da janela do modelo; nenhuma extração geográfica futura foi solicitada.",
            },
        )
        return result
    context = describe_geography(
        latitude, longitude, issued, cache_dir=cache_dir, fetch=fetch
    )
    if view_dir:
        result["geography_view"] = str(export_view(context, view_dir))
    result["geographic_context"] = context
    result["geography_model_used"] = False
    result["primary_prediction_uses_geography"] = False
    if geography_model_dir is None or context["status"] not in ("ok", "partial"):
        return result
    root = Path(geography_model_dir)
    if not (root / "registry.json").exists():
        result["geography_model_status"] = "model_unavailable"
        return result
    if not allow_exploratory or not result.get("valid_from"):
        result["geography_model_status"] = "exploratory_disabled_or_unsupported_date"
        return result
    registry = read_json(root / "registry.json")
    if sha256_file(root / "examples.parquet") != registry["examples_sha256"]:
        raise ValueError("Exemplos geográficos alterados.")
    for filename, digest in registry["artifacts"].items():
        if sha256_file(root / filename) != digest:
            raise ValueError("Artefato geográfico alterado.")
    for module, digest in registry["code"].items():
        if sha256_file(Path(__file__).with_name(module + ".py")) != digest:
            raise ValueError("Código geográfico incompatível com o treinamento.")
    from .geo_model import BASE_FEATURES, GEO_FEATURES, posterior

    features = context["features"]
    if (
        max(
            features.get("geo_ndvi_sample_n", 0),
            features.get("geo_landcover_sample_n", 0),
        )
        < 5
    ):
        result["geography_model_status"] = "insufficient_geographic_samples"
        return result
    start = pd.Timestamp(result["valid_from"])
    phase = 2 * np.pi * start.dayofyear / 365.2425
    from .expansion_grid import point_cell

    cell = point_cell(latitude, longitude)
    data = {
        "latitude": cell["latitude"],
        "longitude": cell["longitude"],
        "season_sin": np.sin(phase),
        "season_cos": np.cos(phase),
        **{k: features.get(k) for k in GEO_FEATURES},
    }
    data.update(
        latitude_season_sin=data["latitude"] * data["season_sin"],
        latitude_season_cos=data["latitude"] * data["season_cos"],
    )
    selected = registry["selected"]
    values = posterior(root / (selected + ".pt"), pd.DataFrame([data]), samples=400)[
        :, 0
    ]
    saved_features = pd.read_parquet(root / "examples.parquet")
    training = saved_features.loc[
        (saved_features.split == "train") & (saved_features.spatial_group == "train")
    ]
    ood = [
        name
        for name in (
            BASE_FEATURES + GEO_FEATURES
            if selected == "with_geography"
            else BASE_FEATURES
        )
        if data.get(name) is not None
        and training[name].notna().any()
        and not training[name].min() <= data[name] <= training[name].max()
    ]
    result["geography_candidate"] = {
        "status": "exploratory",
        "prediction_type": "preditivo_exploratorio_geografico",
        "selected_model": selected,
        "probability": float(values.mean()),
        "probability_interval_95": np.quantile(values, [0.025, 0.975]).tolist(),
        "predictive_event_interval_95": bernoulli.ppf(
            [0.025, 0.975], float(values.mean())
        )
        .astype(int)
        .tolist(),
        "uncertainty_method": "Posterior variacional Pyro; não inclui toda a incerteza de transferência para regiões novas.",
        "uses_geography": selected == "with_geography",
        "locally_calibrated": False,
        "outside_training_feature_range": ood,
        "ignored_features_not_observed_in_training": [
            name
            for name in GEO_FEATURES
            if data.get(name) is not None and not training[name].notna().any()
        ],
        "registry_sha256": sha256_file(root / "registry.json"),
        "interpretation": "Candidato comparável; não substitui silenciosamente a previsão v6 validada. Veja métricas e limites do experimento.",
    }
    result["geography_model_used"] = selected == "with_geography"
    result["primary_prediction_uses_geography"] = False
    context["model_uses_geography"] = selected == "with_geography"
    context["model_output"] = "geography_candidate"
    return result
