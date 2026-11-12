"""Piloto multicontinental particionado; liberação definida na validação."""

from __future__ import annotations
import argparse
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .global_grid import contains, footprint, locate
from .ingest import region_id
from .io import read_json, sha256_file, write_json
from .mask_features import weather_history
from .power import NasaPowerClient, WeatherUnavailable, power_frame
from .runtime import ResourceLimit, Run

EVENT = "ao menos um pixel MOD14A2 com foco no composto de até oito dias, condicionado à máscara integralmente válida"


def collect(run: Run) -> dict[str, Any]:
    from .global_collect import collect as collect_partitions

    return collect_partitions(run)


def prepare(run: Run) -> dict[str, Any]:
    masks = run.path / "coverage/mask_summary.parquet"
    if sha256_file(masks) != read_json(run.path / "coverage/summary.json")["sha256"]:
        raise ValueError("Máscaras alteradas.")
    frame = pd.read_parquet(masks)
    geometry_fields = ["xllcorner", "yllcorner", "cellsize_m", "nrows", "ncols"]
    if (
        frame.groupby("region_id")[geometry_fields]
        .nunique(dropna=False)
        .to_numpy()
        .max()
        != 1
    ):
        raise ValueError(
            "Geometria do recorte mudou entre datas; não é seguro combinar os alvos."
        )
    expected = []
    for region in run.config["regions"]:
        for year in range(
            run.config["mask"]["start_year"], run.config["mask"]["end_year"] + 1
        ):
            day = date(year, 1, 1)
            while day.year == year:
                expected.append(
                    {
                        "region_id": region_id(region["latitude"], region["longitude"]),
                        "period_start": day.isoformat(),
                    }
                )
                day += timedelta(days=8)
    gaps = pd.DataFrame(expected).merge(
        frame[["region_id", "period_start"]], how="left", indicator=True
    )
    gaps = gaps.loc[gaps["_merge"] == "left_only", ["region_id", "period_start"]]
    gaps["observation_state"] = "missing_unknown"
    (run.path / "processed").mkdir(exist_ok=True)
    gaps.to_parquet(run.path / "processed/coverage_gaps.parquet", index=False)
    for col in ["period_start", "period_end_exclusive"]:
        frame[col] = pd.to_datetime(frame[col])
    frame = frame.sort_values(["region_id", "period_start"]).reset_index(drop=True)
    frame["produced_at"] = pd.to_datetime(
        frame.proc_date.astype(str), format="%Y%j%H%M%S", errors="coerce"
    )
    # Cada previsão usa apenas a última imagem cuja produção precede a emissão.
    history = []
    for _, group in frame.groupby("region_id"):
        for idx, row in group.iterrows():
            past = group.loc[
                (group.period_end_exclusive <= row.period_start)
                & (group.produced_at < row.period_start)
                & (group.period_start >= row.period_start - pd.Timedelta(days=40))
            ]
            previous = past.iloc[-1] if len(past) else None
            event = previous.event if previous is not None else pd.NA
            history.append(
                {
                    "index": idx,
                    "past_modis_event": float(event) if pd.notna(event) else np.nan,
                    "past_modis_missing": int(pd.isna(event)),
                    "history_last_produced_at": previous.produced_at
                    if previous is not None
                    else pd.NaT,
                }
            )
    frame = frame.join(pd.DataFrame(history).set_index("index"))
    from .mask_features import _inside_mask

    satellite = (
        run.root
        / "runs"
        / run.config.get("global_records_run", run.run_id + "_records")
        / "satellite"
    )
    if not (satellite / "manifest.json").exists():
        satellite = run.root / "runs" / (run.run_id + "_records") / "satellite"
    satellite_manifest = read_json(satellite / "manifest.json")
    raw = []
    for name, item in satellite_manifest["files"].items():
        from .ingest import archive_product

        product, year = archive_product(Path(name))
        part = satellite / (product + "_" + str(year) + ".parquet")
        if sha256_file(part) != item["output_sha256"]:
            raise ValueError("Histórico satélite alterado.")
        raw.append(pd.read_parquet(part))
    records = pd.concat(raw, ignore_index=True)
    # Diagnóstico descritivo por sensor: combinações sem registros permanecem ausentes.
    from .mask_features import period_start

    records["acquired_utc"] = pd.to_datetime(records.acquired_utc, utc=True)
    records["sensor_period"] = period_start(records.acquired_utc)
    sensor_groups = ["region_id", "product", "sensor_period"] + [
        name
        for name in ("satellite", "instrument", "version", "daynight", "confidence")
        if name in records
    ]
    sensor_counts = (
        records.groupby(sensor_groups, dropna=False)
        .size()
        .rename("positive_record_count")
        .reset_index()
    )
    sensor_counts["interpretation"] = (
        "contagem em célula climática; ausência de linha não é zero; não mede incêndios independentes"
    )
    sensor_counts.to_parquet(run.path / "processed/sensor_counts.parquet", index=False)
    records = records.drop(columns="sensor_period")
    records = records.loc[
        (records["product"] == "viirs-jpss1") & (records.satellite == "N20")
    ].copy()
    records["acquired_utc"] = pd.to_datetime(
        records.acquired_utc, utc=True
    ).dt.tz_localize(None)
    frame["past_viirs_log_records"] = np.nan
    frame["past_viirs_missing"] = 1
    for rid, group in frame.groupby("region_id"):
        one = records.loc[records.region_id == rid]
        one = one.loc[_inside_mask(one, group.iloc[0].to_dict())]
        for idx, row in group.iterrows():
            cutoff = row.period_start - pd.Timedelta(days=2)
            count = int(
                one.acquired_utc.between(
                    cutoff - pd.Timedelta(days=8), cutoff, inclusive="left"
                ).sum()
            )
            if count:
                frame.loc[idx, "past_viirs_log_records"] = np.log1p(count)
                frame.loc[idx, "past_viirs_missing"] = 0
    frame["season_sin"] = np.sin(2 * np.pi * frame.period_start.dt.dayofyear / 365.2425)
    frame["season_cos"] = np.cos(2 * np.pi * frame.period_start.dt.dayofyear / 365.2425)
    weather = []
    errors = {}
    client = NasaPowerClient(
        run.path / "weather/cache",
        max_requests=run.config["operational"]["nasa_power_requests"],
        timeout=run.config["operational"]["network_timeout_seconds"],
        retries=run.config["operational"]["network_retries"],
    )
    for region in run.config["regions"]:
        run.guard()
        try:
            envelope = client.fetch(
                region["latitude"],
                region["longitude"],
                run.config["weather_start"],
                run.config["weather_end"],
                parameters=("T2M", "RH2M", "PRECTOTCORR", "WS10M", "GWETROOT"),
            )
            expected = {
                "T2M": "C",
                "RH2M": "%",
                "PRECTOTCORR": "mm/day",
                "WS10M": "m/s",
                "GWETROOT": "1",
            }
            units = {
                k: envelope["response"]["parameters"][k]["units"] for k in expected
            }
            if units != expected:
                raise ValueError(f"Unidades POWER incompatíveis: {units}")
            part = power_frame(envelope)
            part["region_id"] = region_id(region["latitude"], region["longitude"])
            weather.append(part)
        except WeatherUnavailable as exc:
            errors[region["name"]] = type(exc).__name__
        print(
            f"Clima {region['name']}: {'ok' if region['name'] not in errors else 'indisponível'}",
            flush=True,
        )
    if not weather:
        raise WeatherUnavailable("Nenhuma série meteorológica disponível.")
    daily = pd.concat(weather, ignore_index=True)
    (run.path / "weather").mkdir(exist_ok=True)
    daily.to_parquet(run.path / "weather/daily.parquet", index=False)
    frame = frame.merge(
        weather_history(daily, frame),
        on=["region_id", "period_start"],
        validate="one_to_one",
    )
    split = run.config["split"]
    frame["split"] = "excluded"
    frame.loc[
        frame.period_end_exclusive
        <= pd.Timestamp(split["train_end"]) + pd.Timedelta(days=1),
        "split",
    ] = "train"
    frame.loc[
        (frame.period_start > split["train_end"])
        & (
            frame.period_end_exclusive
            <= pd.Timestamp(split["validation_end"]) + pd.Timedelta(days=1)
        ),
        "split",
    ] = "validation"
    frame.loc[
        (frame.period_start >= split["test_start"])
        & (
            frame.period_end_exclusive
            <= pd.Timestamp(split["test_end"]) + pd.Timedelta(days=1)
        ),
        "split",
    ] = "test"
    continents = {r["name"]: r["continent"] for r in run.config["regions"]}
    frame["continent"] = frame.region_name.map(continents)
    frame["observation_state"] = np.where(
        frame.event.isna(),
        np.where(frame.cloud_pixels > 0, "cloud_obscured", "missing_unknown"),
        np.where(
            (frame.event == 1).fillna(False),
            "positive_detection",
            "observed_no_detection",
        ),
    )
    (run.path / "processed").mkdir(exist_ok=True)
    path = run.path / "processed/mask_examples.parquet"
    frame.to_parquet(path, index=False)
    features = []
    for _, row in frame.groupby("region_id").first().iterrows():
        features.append(
            {
                "type": "Feature",
                "geometry": footprint(row.to_dict()),
                "properties": {
                    "cell": locate(row.latitude, row.longitude),
                    "region": row.region_name,
                    "continent": row.continent,
                    "spatial_group": row.spatial_group,
                },
            }
        )
    write_json(
        run.path / "processed/areas.geojson",
        {"type": "FeatureCollection", "features": features},
    )
    result = {
        "rows": len(frame),
        "unknown": int(frame.event.isna().sum()),
        "weather_errors": errors,
        "examples_sha256": sha256_file(path),
        "masks_sha256": sha256_file(masks),
        "satellite_manifest_sha256": sha256_file(satellite / "manifest.json"),
        "code_sha256": sha256_file(Path(__file__)),
        "groups": frame.groupby(["continent", "split", "spatial_group"])
        .event.agg(["count", "sum"])
        .reset_index()
        .to_dict("records"),
    }
    write_json(run.path / "processed/manifest.json", result)
    run.stage("global_prepare", "complete", rows=len(frame))
    return result


def model(run: Run) -> dict[str, Any]:
    from .mask_model import eligible, train, posterior_probabilities
    from .mask_features import FEATURES_BASE, FEATURES_WEATHER
    from .mask_evaluate import _baseline_fit, _baselines, _score, _calibration
    from .viirs_daily_model import fit_logistic, logistic_predict

    path = run.path / "processed/mask_examples.parquet"
    manifest = read_json(run.path / "processed/manifest.json")
    if sha256_file(path) != manifest["examples_sha256"] or manifest[
        "code_sha256"
    ] != sha256_file(Path(__file__)):
        raise ValueError("Preparação incompatível.")
    metrics_path = run.path / "evaluation/metrics.json"
    if metrics_path.exists():
        previous = read_json(metrics_path)
        if previous["data_sha256"] != manifest["examples_sha256"] or previous[
            "code_sha256"
        ] != sha256_file(Path(__file__)):
            raise ValueError("Avaliação concluída incompatível; use outro run_id.")
        return previous
    frame = eligible(pd.read_parquet(path))
    fit = frame.loc[(frame.split == "train") & (frame.spatial_group == "train")]
    valid = frame.loc[(frame.split == "validation") & (frame.spatial_group == "train")]
    test = frame.loc[frame.split == "test"]
    if len(valid) < 30 or valid.event.sum() < 5 or len(test) < 30:
        raise ValueError("Validação/teste insuficientes.")
    baseline = _baseline_fit(fit)
    models = {
        name: train(
            run,
            frame,
            weather=name == "pyro_weather",
            steps=run.config["global_model"]["steps"],
        )
        for name in ("pyro_no_weather", "pyro_weather")
    }
    logistic = {
        name: fit_logistic(
            fit,
            list(FEATURES_BASE)
            + (list(FEATURES_WEATHER) if name == "logistic_weather" else []),
        )
        for name in ("logistic_no_weather", "logistic_weather")
    }
    for item in logistic.values():
        raw = fit[item["features"]].to_numpy(float)
        clean = np.where(np.isnan(raw), np.array(item["fill"]), raw)
        x = np.column_stack(
            [
                np.ones(len(fit)),
                (clean - np.array(item["mean"])) / np.array(item["scale"]),
            ]
        )
        p = logistic_predict(item, fit)
        precision = x.T @ (x * (p * (1 - p))[:, None]) + np.diag(
            [1e-8] + [item["penalty"]] * (x.shape[1] - 1)
        )
        item["laplace_covariance"] = np.linalg.pinv(precision).tolist()
    write_json(run.path / "statistical_models.json", logistic)
    serialized = {
        "global": baseline["global"],
        "month": {str(k): v for k, v in baseline["month"].items()},
        "month_region": [
            [str(r), int(m), float(p)] for (r, m), p in baseline["month_region"].items()
        ],
        "transition": {str(k): v for k, v in baseline["transition"].items()},
    }
    write_json(run.path / "baseline_model.json", serialized)

    def beta_parameters(group):
        return [
            float(group.event.sum() + 2 * baseline["global"]),
            float(len(group) - group.event.sum() + 2 * (1 - baseline["global"])),
        ]

    beta_counts = {
        "global": [
            float(fit.event.sum() + 0.5),
            float(len(fit) - fit.event.sum() + 0.5),
        ],
        "month": {
            str(m): beta_parameters(g)
            for m, g in fit.groupby(fit.period_start.dt.month)
        },
        "month_region": [
            [str(r), int(m), *beta_parameters(g)]
            for (r, m), g in fit.groupby(["region_id", fit.period_start.dt.month])
        ],
        "transition": {
            str(float(v)): beta_parameters(g)
            for v, g in fit.dropna(subset=["past_modis_event"]).groupby(
                "past_modis_event"
            )
        },
    }
    write_json(run.path / "baseline_uncertainty.json", beta_counts)
    bounds = {
        name: [float(fit[name].min()), float(fit[name].max())]
        for name in FEATURES_WEATHER
    }
    write_json(run.path / "feature_bounds.json", bounds)

    def predictions(data):
        pred = _baselines(baseline, data)
        for name, checkpoint in models.items():
            pred[name] = posterior_probabilities(
                checkpoint, data, 400, run.config["seed"]
            ).mean(axis=0)
        for name, item in logistic.items():
            pred[name] = logistic_predict(item, data)
        return pred

    vp = predictions(valid)
    scores = {n: _score(valid.event.to_numpy(int), p) for n, p in vp.items()}
    champion = min(scores, key=lambda n: scores[n]["brier"])
    policy = {
        "selected_on": "validation/train only",
        "champion": champion,
        "validation": scores,
        "groups": {},
    }
    # Contagem, taxa de eventos e erro de calibração são critérios prévios da configuração.
    allvalid = frame.loc[frame.split == "validation"]
    ap = predictions(allvalid)
    for name, g in allvalid.groupby("region_name"):
        index = allvalid.index.get_indexer(g.index)
        p = ap[champion][index]
        y = g.event.to_numpy(int)
        s = _score(y, p)
        seasonal_score = _score(y, ap["seasonal"][index])
        selected_model = champion
        if seasonal_score["brier"] < s["brier"]:
            selected_model = "seasonal"
            p = ap["seasonal"][index]
            s = seasonal_score
        cfg = run.config["release"]
        passed = (
            len(y) >= cfg["min_n"]
            and min(int(y.sum()), int(len(y) - y.sum())) >= cfg["min_class_n"]
            and abs(p.mean() - y.mean()) <= cfg["max_mean_calibration_error"]
            and s["brier"]
            <= _score(y, ap["seasonal"][index])["brier"] * cfg["max_brier_ratio"]
        )
        policy["groups"][name] = {
            "eligible": bool(passed),
            "validation": s,
            "selected_model": selected_model,
        }
    write_json(run.path / "release_policy.json", policy)
    registry_files = [
        "release_policy.json",
        "statistical_models.json",
        "baseline_model.json",
        "baseline_uncertainty.json",
        "feature_bounds.json",
        "models/mask_weather.pt",
        "models/mask_no_weather.pt",
    ]
    write_json(
        run.path / "model_registry.json",
        {
            "version": "global_mod14a2_v5",
            "config_sha256": run.config_hash,
            "data_sha256": manifest["examples_sha256"],
            "pipeline_sha256": sha256_file(Path(__file__)),
            "artifacts": {p: sha256_file(run.path / p) for p in registry_files},
        },
    )
    # A política está persistida antes de abrir previsões do teste.
    tp = predictions(test)
    routed = np.array(
        [
            tp[policy["groups"].get(region, {}).get("selected_model", champion)][i]
            for i, region in enumerate(test.region_name)
        ]
    )
    result = {
        "event": EVENT,
        "horizon": "MODIS composto de até oito dias",
        "training_n": len(fit),
        "validation": scores,
        "champion": champion,
        "test": {n: _score(test.event.to_numpy(int), p) for n, p in tp.items()},
        "by_region": {},
        "by_continent": {},
        "code_sha256": sha256_file(Path(__file__)),
        "data_sha256": manifest["examples_sha256"],
    }
    export = test[
        [
            "region_id",
            "region_name",
            "continent",
            "spatial_group",
            "period_start",
            "event",
        ]
    ].copy()
    for n, p in tp.items():
        export[n] = p
    for group, key in [("region_name", "by_region"), ("continent", "by_continent")]:
        for name, g in test.groupby(group):
            idx = test.index.get_indexer(g.index)
            result[key][name] = {
                n: _score(g.event.to_numpy(int), p[idx]) for n, p in tp.items()
            }
    result["calibration"] = _calibration(test.event.to_numpy(int), tp[champion])
    supported = test.region_name.map(
        lambda name: policy["groups"].get(name, {}).get("eligible", False)
    ).to_numpy(bool)
    for name, (low, high) in bounds.items():
        supported &= test[name].between(low, high).to_numpy()
    result["routed_test"] = _score(test.event.to_numpy(int), routed)
    result["release_test"] = {
        "eligible_n": int(supported.sum()),
        "total_n": len(test),
        "metrics": _score(test.event.to_numpy(int)[supported], routed[supported])
        if supported.any()
        else None,
    }
    export["routed_probability"] = routed
    export["release_eligible"] = supported
    result["accuracy_at_threshold_0_5"] = {
        name: float(np.mean((p >= 0.5) == test.event.to_numpy(int)))
        for name, p in tp.items()
    }
    # Reamostragem em blocos de região conserva a dependência dentro de cada área.
    rng = np.random.default_rng(run.config["seed"])
    groups = [test.index.get_indexer(g.index) for _, g in test.groupby("region_name")]
    draws = []
    for _ in range(500):
        idx = np.concatenate(
            [groups[k] for k in rng.integers(0, len(groups), len(groups))]
        )
        y = test.event.to_numpy(int)[idx]
        draws.append(
            [
                np.mean((tp[champion][idx] - y) ** 2),
                np.mean((tp[champion][idx] - y) ** 2 - (tp["seasonal"][idx] - y) ** 2),
            ]
        )
    result["region_block_bootstrap_95"] = {
        "brier": np.quantile(np.array(draws)[:, 0], [0.025, 0.975]).tolist(),
        "brier_difference_vs_seasonal": np.quantile(
            np.array(draws)[:, 1], [0.025, 0.975]
        ).tolist(),
        "limitation": "12 áreas intencionais; não mede incerteza para todo o planeta.",
    }
    (run.path / "evaluation").mkdir(exist_ok=True)
    export.to_parquet(run.path / "evaluation/predictions.parquet", index=False)
    # Calendário descritivo; anomalia depende apenas das observações anteriores.
    calendar = pd.read_parquet(path).sort_values(["region_id", "period_start"])
    calendar["historical_rate"] = np.nan
    for _, g in calendar.groupby("region_id"):
        for idx, row in g.iterrows():
            past = g.loc[
                (g.period_end_exclusive <= row.period_start)
                & (g.produced_at < row.period_start)
                & (g.period_start.dt.month == row.period_start.month)
            ].event.dropna()
            if len(past) >= 10:
                calendar.loc[idx, "historical_rate"] = float(past.mean())
    calendar["observed_anomaly"] = (
        calendar.event.astype(float) - calendar.historical_rate
    )
    calendar.to_parquet(run.path / "evaluation/calendar.parquet", index=False)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    names = list(result["test"])
    ax.bar(names, [result["test"][n]["brier"] for n in names])
    ax.tick_params(axis="x", rotation=30)
    ax.set_ylabel("Brier no teste (menor é melhor)")
    fig.tight_layout()
    fig.savefig(run.path / "evaluation/comparison.png")
    plt.close(fig)
    write_json(run.path / "evaluation/metrics.json", result)
    run.stage(
        "global_model",
        "complete",
        champion=champion,
        training_n=len(fit),
        test_n=len(test),
    )
    return result


def predict(
    latitude: float, longitude: float, reference_date: str, run_dir: str | Path
) -> dict[str, Any]:
    """Consulta sem retreino; apenas recortes, datas e política avaliados são elegíveis."""
    root = Path(run_dir)
    cell = locate(latitude, longitude)
    origin = pd.Timestamp(reference_date)
    if origin.tz is not None:
        origin = origin.tz_convert("UTC").tz_localize(None)
    result = {
        "schema_version": "global-1.0",
        "status": "insufficient_data",
        "requested_location": {"latitude": latitude, "longitude": longitude},
        "cell": cell,
        "issued_at": origin.isoformat()+"Z",
        "horizon_days": None,
        "event": EVENT,
        "probability": None,
        "probability_interval_95": None,
        "model_version": "global_mod14a2_v5",
        "reason": "Local ou data sem suporte validado.",
    }
    if not (root / "processed/manifest.json").exists():
        result["reason"] = (
            "Modelo e histórico preparado não estão disponíveis neste diretório."
        )
        return result
    countries = root / "catalog/countries.geojson"
    if countries.exists():
        from .global_catalog import country_names

        if (
            sha256_file(countries)
            != read_json(root / "catalog/countries_manifest.json")["sha256"]
        ):
            raise ValueError("Catálogo administrativo alterado.")
        result["administrative_context"] = {
            "countries": country_names(latitude, longitude, read_json(countries)),
            "source": "Natural Earth 1:110m; associação aproximada",
            "state": None,
        }
    manifest = read_json(root / "processed/manifest.json")
    if (
        sha256_file(root / "processed/mask_examples.parquet")
        != manifest["examples_sha256"]
    ):
        raise ValueError("Exemplos alterados.")
    frame = pd.read_parquet(root / "processed/mask_examples.parquet")
    areas = frame.loc[frame.region_id == cell["id"]]
    if areas.empty or not contains(areas.iloc[0].to_dict(), latitude, longitude):
        return result
    registry = read_json(root / "model_registry.json")
    if registry["data_sha256"] != manifest["examples_sha256"] or registry[
        "pipeline_sha256"
    ] != sha256_file(Path(__file__)):
        raise ValueError("Modelo incompatível com os dados ou com o pipeline.")
    if any(
        sha256_file(root / p) != digest for p, digest in registry["artifacts"].items()
    ):
        raise ValueError("Artefato de inferência alterado.")
    result["analyzed_area"] = footprint(areas.iloc[0].to_dict())
    policy = read_json(root / "release_policy.json")
    region = areas.iloc[0].region_name
    if not policy["groups"].get(region, {}).get("eligible", False):
        result["reason"] = "Região não passou nos critérios de validação."
        return result
    rows = areas.loc[areas.period_start == origin]
    if rows.empty:
        return result
    result["valid_from"] = origin.isoformat()+"Z"
    result["valid_until"] = pd.Timestamp(rows.iloc[0].period_end_exclusive).isoformat()+"Z"
    result["horizon_days"] = int((rows.iloc[0].period_end_exclusive-origin).days)
    result["preprocessing_version"] = registry["pipeline_sha256"][:12]
    result["quality"] = {"future_observability":"unknown", "historical_availability":"acquisition_lags_assumed",
                          "weather_lag_days":3, "viirs_lag_days":2}
    from .mask_model import posterior_probabilities
    from .mask_features import FEATURES_WEATHER

    row = rows.loc[
        np.isfinite(rows[list(FEATURES_WEATHER)].to_numpy(float)).all(axis=1)
    ]
    if row.empty:
        result["reason"] = "Meteorologia retrospectiva indisponível."
        return result
    bounds = read_json(root / "feature_bounds.json")
    outside = [
        name
        for name, (low, high) in bounds.items()
        if not low <= float(row.iloc[0][name]) <= high
    ]
    if outside:
        result.update(
            status="out_of_distribution",
            reason="Clima fora dos limites do treino.",
            extrapolated_features=outside,
        )
        return result
    # O cliente público retorna o campeão validado, inclusive quando é uma referência.
    from .mask_evaluate import _baselines
    from .viirs_daily_model import logistic_predict

    champion = policy["groups"][region].get("selected_model", policy["champion"])
    if champion.startswith("pyro_"):
        checkpoint = (
            root
            / "models"
            / (
                "mask_weather.pt"
                if champion == "pyro_weather"
                else "mask_no_weather.pt"
            )
        )
        values = posterior_probabilities(checkpoint, row, 400, 42)[:, 0]
        result["probability_interval_95"] = np.quantile(values, [0.025, 0.975]).tolist()
        result["uncertainty_method"] = "posterior variacional Pyro sobre os parâmetros"
        probability = float(values.mean())
    elif champion.startswith("logistic_"):
        # Artefato estatístico salvo no treino; consulta nunca ajusta parâmetros.
        item = read_json(root / "statistical_models.json")[champion]
        probability = float(logistic_predict(item, row)[0])
        raw = row[item["features"]].to_numpy(float)
        clean = np.where(np.isnan(raw), np.array(item["fill"]), raw)
        x = np.r_[1.0, ((clean - np.array(item["mean"])) / np.array(item["scale"]))[0]]
        from scipy.special import expit

        rng = np.random.default_rng(42)
        coefficients = rng.multivariate_normal(
            item["coef"], item["laplace_covariance"], size=1000
        )
        result["probability_interval_95"] = np.quantile(
            expit(coefficients @ x), [0.025, 0.975]
        ).tolist()
        result["uncertainty_method"] = (
            "aproximação de Laplace para os coeficientes regularizados"
        )
    else:
        stored = read_json(root / "baseline_model.json")
        baseline = {
            "global": stored["global"],
            "month": {int(k): v for k, v in stored["month"].items()},
            "month_region": {(r, int(m)): p for r, m, p in stored["month_region"]},
            "transition": {float(k): v for k, v in stored["transition"].items()},
        }
        probability = float(_baselines(baseline, row)[champion][0])
        counts = read_json(root / "baseline_uncertainty.json")
        rr = row.iloc[0]
        month = int(rr.period_start.month)
        ab = counts["global"]
        if champion == "seasonal":
            mapping = {(r, int(m)): [a, b] for r, m, a, b in counts["month_region"]}
            ab = mapping.get((rr.region_id, month), counts["month"].get(str(month), ab))
        elif champion == "persistence" and pd.notna(rr.past_modis_event):
            ab = counts["transition"].get(str(float(rr.past_modis_event)), ab)
        from scipy.stats import beta

        result["probability_interval_95"] = beta.ppf([0.025, 0.975], *ab).tolist()
        result["uncertainty_method"] = (
            "posterior Beta com compartilhamento do valor médio global"
        )
    result.update(
        status="ok",
        probability=probability,
        selected_model=champion,
        reason=None,
        weather_last_date=str(row.iloc[0].weather_last_date),
        historical_replay=True,
        predictive_count_support=[0, 1],
        unit="probabilidade condicional do evento binário no recorte MODIS",
    )
    result["interval_interpretation"] = (
        "incerteza sobre a probabilidade; a observação futura é binária, com suporte {0,1}"
    )
    from scipy.stats import bernoulli

    result["predictive_event_intervals"] = {
        "80": bernoulli.ppf([0.1, 0.9], probability).astype(int).tolist(),
        "95": bernoulli.ppf([0.025, 0.975], probability).astype(int).tolist(),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=["collect", "records", "prepare", "train", "all", "predict"]
    )
    parser.add_argument("--config", type=Path, default=Path("config/global_v5b.json"))
    parser.add_argument("--run-id", default="global_v5b_20261111")
    parser.add_argument("--latitude", type=float)
    parser.add_argument("--longitude", type=float)
    parser.add_argument("--date")
    args = parser.parse_args()
    import os

    os.environ.setdefault("EARTHFIRE_BUDGET_ID", args.run_id)
    run = Run(
        args.config,
        args.run_id + "_records" if args.stage == "records" else args.run_id,
    )
    try:
        if args.stage == "predict":
            output = predict(args.latitude, args.longitude, args.date, run.path)
        elif args.stage == "records":
            from .global_records import extract

            output = extract(run)
        else:
            output = {}
            if args.stage == "all":
                from .global_records import extract

                extract(Run(args.config, args.run_id + "_records"))
            for name, fn in [
                ("collect", collect),
                ("prepare", prepare),
                ("train", model),
            ]:
                if args.stage in (name, "all"):
                    output[name] = fn(run)
        print(json.dumps(output, ensure_ascii=False, default=str), flush=True)
    except (ResourceLimit, WeatherUnavailable, ValueError, KeyboardInterrupt) as exc:
        run.stage(
            "global", "interrupted", error_type=type(exc).__name__, reason=str(exc)
        )
        raise


if __name__ == "__main__":
    main()
