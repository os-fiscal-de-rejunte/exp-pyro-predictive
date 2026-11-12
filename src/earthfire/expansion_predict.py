"""Consulta histórica e projeção sazonal explícita para células validadas, sem retreino."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta, bernoulli

from .expansion_grid import point_cell
from .global_grid import footprint
from .io import read_json, sha256_file

EVENT = "ao menos um pixel MOD14A2 marcado como foco no composto, condicionado a todos os 41×41 pixels terem máscara válida"


def load(root: Path):
    registry = read_json(root / "model_registry.json")
    for filename, digest in registry["artifacts"].items():
        if sha256_file(root / filename) != digest:
            raise ValueError("Artefato de modelo alterado: " + filename)
    for module, digest in registry["code"].items():
        if sha256_file(Path(__file__).with_name(module + ".py")) != digest:
            raise ValueError("Código incompatível com o modelo: " + module)
    return (
        registry,
        read_json(root / "cell_catalog.json"),
        read_json(root / "baseline_model.json"),
    )


def baseline_parameters(model, method, rid, month, past=None):
    if method == "seasonal":
        return (
            model["region_month"]
            .get(rid, {})
            .get(str(month), model["month"].get(str(month), model["global"]))
        )
    if method == "persistence" and past is not None and pd.notna(past):
        return model["transition"].get(str(float(past)), model["global"])
    return model["global"]


def base_result(latitude, longitude):
    cell = point_cell(float(latitude), float(longitude))
    return {
        "schema_version": "native41-1.0",
        "status": "insufficient_data",
        "requested_location": {"latitude": latitude, "longitude": longitude},
        "cell": {
            "id": cell["region_id"],
            "geometry": footprint(cell),
            "area_km2": cell["area_km2"],
        },
        "event": EVENT,
        "probability": None,
        "probability_interval_95": None,
        "model_version": "modis_native41_v6",
    }


def predict(
    latitude: float, longitude: float, reference_date: str, run_dir: str | Path
) -> dict:
    root = Path(run_dir)
    result = base_result(latitude, longitude)
    if not (root / "model_registry.json").exists():
        result["reason"] = "Modelo ainda não disponível."
        return result
    registry, catalog, model = load(root)
    rid = result["cell"]["id"]
    info = catalog.get(rid)
    if info is None:
        result["reason"] = "Localização reconhecida; célula ainda não foi avaliada."
        return result
    result["history"] = {
        "period_n": info["training_n"],
        "positive_periods": info["training_positive"],
        "observed_rate": info["training_positive"] / info["training_n"]
        if info["training_n"]
        else None,
        "interpretation": "frequência observada no período de treino, não uma previsão",
    }
    entry = info["release"]
    if not entry.get("eligible"):
        result["reason"] = "A validação não sustentou uma previsão para esta célula."
        return result
    path = root / info["feature_file"]
    if sha256_file(path) != registry["features"][info["feature_file"]]:
        raise ValueError("Histórico alterado.")
    frame = pd.read_parquet(path, filters=[("region_id", "==", rid)])
    issued = pd.Timestamp(reference_date)
    if issued.tz is not None:
        issued = issued.tz_convert("UTC").tz_localize(None)
    row = frame.loc[frame.period_start == issued]
    if row.empty:
        result["reason"] = (
            "Data não preparada. Use forecast para projeção sazonal do próximo composto."
        )
        return result
    method = entry["selected_model"]
    values = None
    needs_weather = method in ("pyro_weather", "logistic_weather")
    bounds = read_json(root / "feature_bounds.json")
    ood = [
        name
        for name, (low, high) in bounds.items()
        if pd.notna(row.iloc[0][name]) and not low <= row.iloc[0][name] <= high
    ]
    if needs_weather and (not row.iloc[0].weather_complete or ood):
        method = entry.get("static_projection_model")
        if method is None:
            result.update(
                status="out_of_distribution" if ood else "weather_unavailable",
                reason="Clima indisponível ou fora do treino.",
            )
            return result
        result["fallback_reason"] = (
            "Meteorologia indisponível ou extrapolada; referência estática validada."
        )
    if method in ("global_frequency", "seasonal", "persistence"):
        ab = baseline_parameters(
            model, method, rid, issued.month, row.iloc[0].past_modis_event
        )
        central = float(ab[0] / sum(ab))
        interval = beta.ppf([0.025, 0.975], *ab).tolist()
        uncertainty = "posterior Beta com compartilhamento empírico global"
    elif method.startswith("pyro_"):
        from .mask_model import posterior_probabilities

        path = (
            root
            / "models"
            / ("mask_weather.pt" if method == "pyro_weather" else "mask_no_weather.pt")
        )
        values = posterior_probabilities(path, row, 400, 42)[:, 0]
        central = float(values.mean())
        interval = np.quantile(values, [0.025, 0.975]).tolist()
        uncertainty = "posterior variacional Pyro"
    else:
        from scipy.special import expit

        saved = read_json(root / "statistical_models.json")[method]
        raw = row[saved["features"]].to_numpy(float)
        clean = np.where(np.isnan(raw), np.array(saved["fill"]), raw)
        x = np.r_[
            1.0, ((clean - np.array(saved["mean"])) / np.array(saved["scale"]))[0]
        ]
        central = float(expit(x @ np.asarray(saved["coef"])))
        coefficients = np.random.default_rng(42).multivariate_normal(
            saved["coef"], saved["covariance"], 1000
        )
        interval = np.quantile(expit(coefficients @ x), [0.025, 0.975]).tolist()
        uncertainty = "aproximação de Laplace"
    result.update(
        status="ok",
        probability=central,
        probability_interval_95=interval,
        selected_model=method,
        issued_at=issued.isoformat() + "Z",
        valid_from=issued.isoformat() + "Z",
        valid_until=pd.Timestamp(row.iloc[0].period_end_exclusive).isoformat() + "Z",
        horizon_days=int((row.iloc[0].period_end_exclusive - issued).days),
        mode="historical_replay",
        weather_context="available" if row.iloc[0].weather_complete else "unavailable",
        uncertainty_method=uncertainty,
        future_observability="unknown",
        climate_out_of_distribution=ood,
    )
    result["predictive_event_interval_95"] = (
        bernoulli.ppf([0.025, 0.975], central).astype(int).tolist()
    )
    return result


def forecast(
    latitude: float,
    longitude: float,
    run_dir: str | Path,
    reference_date: str | None = None,
) -> dict:
    """Projeção marginal histórica, não condicionada a uma previsão de meteorologia futura."""
    root = Path(run_dir)
    result = base_result(latitude, longitude)
    if not (root / "model_registry.json").exists():
        result["reason"] = "Modelo ainda não disponível."
        return result
    registry, catalog, model = load(root)
    rid = result["cell"]["id"]
    info = catalog.get(rid)
    if info is None:
        result["reason"] = "Localização reconhecida; ainda não há suporte nesta célula."
        return result
    method = info["release"].get("static_projection_model")
    result["history"] = {
        "period_n": info["training_n"],
        "positive_periods": info["training_positive"],
        "observed_rate": info["training_positive"] / info["training_n"]
        if info["training_n"]
        else None,
    }
    if method is None:
        result["reason"] = (
            "Referência sazonal/estática não passou na validação desta célula."
        )
        return result
    issued = (
        pd.Timestamp(reference_date) if reference_date else pd.Timestamp.now(tz="UTC")
    )
    if issued.tz is None:
        issued = issued.tz_localize("UTC")
    else:
        issued = issued.tz_convert("UTC")
    if issued < pd.Timestamp("2023-01-01", tz="UTC"):
        result["reason"] = "A emissão precede o encerramento do treino."
        return result
    if issued.year > 2027:
        result.update(
            status="out_of_distribution",
            reason="Modelo desatualizado para essa emissão.",
        )
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
    ab = baseline_parameters(model, method, rid, start.month)
    if start.month not in info["release"].get("static_projection_months", []):
        result["reason"] = (
            "Mês sem suporte observacional suficiente no treino e na validação."
        )
        return result
    probability = float(ab[0] / sum(ab))
    result.update(
        status="ok",
        probability=probability,
        probability_interval_95=beta.ppf([0.025, 0.975], *ab).tolist(),
        issued_at=issued.isoformat(),
        valid_from=start.isoformat() + "T00:00:00Z",
        valid_until=end.isoformat() + "T00:00:00Z",
        horizon_days=(end - start).days,
        lead_days=(start - day).days,
        mode="seasonal_projection",
        selected_model=method,
        weather_context="historically_marginalized; no current or future weather inputs",
        future_observability="unknown",
        assumption="Estabilidade sazonal da relação histórica; ainda sem validação prospectiva.",
        uncertainty_method="posterior Beta; intervalo sobre a probabilidade, não sobre quantidade de incêndios",
        training_period=["2020-01-01", "2022-12-31"],
        validation_year=2023,
        data_assembled_at=registry.get("created_utc"),
        historical_projection=issued < pd.Timestamp(registry["created_utc"])
        if registry.get("created_utc")
        else True,
    )
    result["predictive_event_interval_95"] = (
        bernoulli.ppf([0.025, 0.975], probability).astype(int).tolist()
    )
    return result
