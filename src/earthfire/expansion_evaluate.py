"""Ajuste limitado em RAM, seleção na validação e teste final em células novas."""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd

from .io import read_json, sha256_file, utc_now, write_json
from .runtime import Run, ResourceLimit

BASES = ("global_frequency", "seasonal", "regional_seasonal", "persistence")


def partitions(run):
    for meta in sorted((run.path / "features").glob("*.json")):
        info = read_json(meta)
        path = meta.with_suffix(".parquet")
        if sha256_file(path) != info["sha256"]:
            raise ValueError("Partição de features alterada.")
        yield path


def baseline_probabilities(model, frame):
    global_rate = model["global"][0] / sum(model["global"])
    result = {key: [] for key in BASES}
    for row in frame.itertuples():
        month = str(row.period_start.month)
        ab = (
            model["region_month"]
            .get(row.region_id, {})
            .get(month, model["month"].get(month, model["global"]))
        )
        past = (
            str(float(row.past_modis_event)) if pd.notna(row.past_modis_event) else ""
        )
        transition = model["transition"].get(past, model["global"])
        result["global_frequency"].append(global_rate)
        result["seasonal"].append(ab[0] / sum(ab))
        regional = (
            model.get("continent_month", {})
            .get(str(row.continent), {})
            .get(month, model["month"].get(month, model["global"]))
        )
        result["regional_seasonal"].append(regional[0] / sum(regional))
        result["persistence"].append(transition[0] / sum(transition))
    return {key: np.asarray(value) for key, value in result.items()}


def summarize(y, p):
    from scipy.stats import rankdata

    if len(y) == 0:
        return None
    y = np.asarray(y, int)
    p = np.clip(np.asarray(p, float), 1e-8, 1 - 1e-8)
    if not np.isfinite(p).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("Predições ou alvos inválidos.")
    n1, n0 = int(y.sum()), int(len(y) - y.sum())
    auc = (
        float((rankdata(p)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
        if n1 and n0
        else None
    )
    order = np.argsort(-p, kind="stable")
    ranked = y[order]
    sorted_p = p[order]
    boundaries = np.r_[np.flatnonzero(np.diff(sorted_p) != 0), len(y) - 1]
    positives = np.cumsum(ranked)[boundaries]
    precision = positives / (boundaries + 1)
    recall = positives / n1 if n1 else np.zeros(len(boundaries))
    ap = float((np.diff(np.r_[0.0, recall]) * precision).sum()) if n1 else None
    result = {
        "n": len(y),
        "positive": n1,
        "observed_rate": float(y.mean()),
        "mean_probability": float(p.mean()),
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": float(np.mean(-y * np.log(p) - (1 - y) * np.log1p(-p))),
        "roc_auc": auc,
        "average_precision": ap,
    }
    positive = p >= 0.5
    tp = int((positive & (y == 1)).sum())
    fp = int((positive & (y == 0)).sum())
    fn = int((~positive & (y == 1)).sum())
    result.update(
        accuracy=float((positive == y).mean()),
        recall=tp / (tp + fn) if tp + fn else None,
        precision=tp / (tp + fp) if tp + fp else None,
        true_positive=tp,
        false_positive=fp,
        false_negative=fn,
    )
    return result


def finalize(run: Run) -> dict:
    from .mask_features import FEATURES_BASE, FEATURES_WEATHER
    from .global_grid import footprint

    files = list(partitions(run))
    if not files:
        raise ValueError(
            "Nenhuma partição preparada; resultados brutos foram preservados."
        )
    directory = run.path / "evaluation"
    directory.mkdir(exist_ok=True)
    if (directory / "metrics.json").exists():
        registry = read_json(run.path / "model_registry.json")
        current = {str(p.relative_to(run.path)): sha256_file(p) for p in files}
        if current != registry["features"]:
            raise ResourceLimit(
                "Avaliação concluída está congelada; novas partições exigem outro run_id."
            )
        from .expansion_predict import load

        load(run.path)
        return read_json(directory / "metrics.json")
    meta = {}
    assets = run.root / "runs" / run.config.get("mask_storage_run", run.run_id)
    for path in sorted((assets / "blocks").glob("*.parquet")):
        raw = pd.read_parquet(path)
        for rid, g in raw.groupby("region_id"):
            if rid in meta:
                continue
            item = g.iloc[0]
            tr = g.loc[(g.period_start < "2023-01-01") & g.event.notna()]
            meta[rid] = {
                name: item[name].item()
                if isinstance(item[name], np.generic)
                else item[name]
                for name in (
                    "parent_id",
                    "continent",
                    "spatial_group",
                    "latitude",
                    "longitude",
                    "xllcorner",
                    "yllcorner",
                    "cellsize_m",
                    "nrows",
                    "ncols",
                    "native_row",
                    "native_column",
                    "area_km2",
                    "legacy_context",
                )
            }
            meta[rid].update(
                feature_file=None,
                training_n=len(tr),
                training_positive=int(tr.event.sum()),
                release={"eligible": False, "static_projection_model": None},
                history_only=True,
                observed_training_months={
                    str(int(m)): len(part)
                    for m, part in tr.groupby(pd.to_datetime(tr.period_start).dt.month)
                },
                observed_validation_months={},
            )
    for path in files:
        frame = pd.read_parquet(path)
        for rid, g in frame.groupby("region_id"):
            item = g.iloc[0]
            meta[rid] = {
                name: item[name].item()
                if isinstance(item[name], np.generic)
                else item[name]
                for name in (
                    "parent_id",
                    "continent",
                    "spatial_group",
                    "latitude",
                    "longitude",
                    "xllcorner",
                    "yllcorner",
                    "cellsize_m",
                    "nrows",
                    "ncols",
                    "native_row",
                    "native_column",
                    "area_km2",
                    "legacy_context",
                )
            }
            tr = g.loc[(g.split == "train") & g.event.notna()]
            meta[rid].update(
                feature_file=str(path.relative_to(run.path)),
                training_n=len(tr),
                training_positive=int(tr.event.sum()),
                history_only=False,
                observed_training_months={
                    str(int(m)): len(part)
                    for m, part in tr.groupby(tr.period_start.dt.month)
                },
                observed_validation_months={
                    str(int(m)): len(part)
                    for m, part in g.loc[
                        (g.split == "validation") & g.event.notna()
                    ].groupby(
                        g.loc[
                            (g.split == "validation") & g.event.notna()
                        ].period_start.dt.month
                    )
                },
            )
    holdout = {
        (v["native_row"], v["native_column"])
        for v in meta.values()
        if v["spatial_group"] == "holdout"
    }
    buffer = {
        rid
        for rid, v in meta.items()
        if v["spatial_group"] == "train"
        and any(
            (v["native_row"] + dr, v["native_column"] + dc) in holdout
            for dr in (-41, 0, 41)
            for dc in (-41, 0, 41)
        )
    }
    rng = np.random.default_rng(run.config["seed"])
    sample = pd.DataFrame()
    cap = run.config["expansion"]["max_training_examples"]
    counts = {}
    transitions = {}
    months = {}
    continent_months = {}
    total = [0, 0]
    bounds = {c: [float("inf"), float("-inf")] for c in FEATURES_WEATHER}
    for path in files:
        frame = pd.read_parquet(path)
        fit = frame.loc[
            (frame.split == "train")
            & (frame.spatial_group == "train")
            & ~frame.region_id.isin(buffer)
            & frame.event.notna()
        ]
        total[0] += int(fit.event.sum())
        total[1] += len(fit)
        for (rid, m), g in fit.groupby(["region_id", fit.period_start.dt.month]):
            counts[(str(rid), int(m))] = [int(g.event.sum()), len(g)]
        for m, g in fit.groupby(fit.period_start.dt.month):
            old = months.setdefault(int(m), [0, 0])
            old[0] += int(g.event.sum())
            old[1] += len(g)
        for (continent, month), g in fit.groupby(
            ["continent", fit.period_start.dt.month]
        ):
            old = continent_months.setdefault((str(continent), int(month)), [0, 0])
            old[0] += int(g.event.sum())
            old[1] += len(g)
        for past, g in fit.dropna(subset=["past_modis_event"]).groupby(
            "past_modis_event"
        ):
            old = transitions.setdefault(str(float(past)), [0, 0])
            old[0] += int(g.event.sum())
            old[1] += len(g)
        complete = fit.loc[fit.weather_complete].copy()
        for c in FEATURES_WEATHER:
            if len(complete):
                bounds[c] = [
                    min(bounds[c][0], float(complete[c].min())),
                    max(bounds[c][1], float(complete[c].max())),
                ]
        if len(complete):
            complete["_priority"] = rng.random(len(complete))
            sample = pd.concat([sample, complete], ignore_index=True)
            sample = sample.nsmallest(cap, "_priority")
        del frame, fit, complete
    if total[1] < 100 or total[0] < 10:
        raise ValueError("Treino observacional insuficiente.")
    rate = (total[0] + 0.5) / (total[1] + 1)

    def beta(pair):
        return [float(pair[0] + 2 * rate), float(pair[1] - pair[0] + 2 * (1 - rate))]

    model = {
        "global": [float(total[0] + 0.5), float(total[1] - total[0] + 0.5)],
        "month": {str(m): beta(v) for m, v in months.items()},
        "region_month": {},
        "continent_month": {},
        "transition": {str(p): beta(v) for p, v in transitions.items()},
    }
    for (rid, m), pair in counts.items():
        model["region_month"].setdefault(rid, {})[str(m)] = beta(pair)
    for (continent, month), pair in continent_months.items():
        model["continent_month"].setdefault(continent, {})[str(month)] = beta(pair)
    write_json(run.path / "baseline_model.json", model)
    write_json(
        run.path / "feature_bounds.json",
        {k: v for k, v in bounds.items() if np.isfinite(v).all()},
    )
    if sample.empty:
        sample = pd.read_parquet(files[0]).iloc[:0]
    sample = sample.drop(columns="_priority", errors="ignore").sort_values(
        ["region_id", "period_start"]
    )
    (run.path / "processed").mkdir(exist_ok=True)
    sample.to_parquet(run.path / "processed/mask_examples.parquet", index=False)
    write_json(
        run.path / "training_sample.json",
        {
            "all_training_n": total[1],
            "sample_n": len(sample),
            "sampling": "prioridade pseudoaleatória uniforme com semente fixa; somente treino; no máximo 12.000 exemplos",
            "sample_sha256": sha256_file(run.path / "processed/mask_examples.parquet"),
            "buffer_cells": sorted(buffer),
        },
    )
    models = {}
    statistical = {}
    training_error = None
    if len(sample) >= 100 and 10 <= sample.event.sum() <= len(sample) - 10:
        try:
            from .mask_model import train
            from .viirs_daily_model import fit_logistic

            for name, weather in [("pyro_no_weather", False), ("pyro_weather", True)]:
                models[name] = train(
                    run,
                    sample,
                    weather=weather,
                    steps=run.config["global_model"]["steps"],
                )
            for name, weather in [
                ("logistic_no_weather", False),
                ("logistic_weather", True),
            ]:
                cols = list(FEATURES_BASE) + (list(FEATURES_WEATHER) if weather else [])
                item = fit_logistic(sample, cols)
                raw = sample[cols].to_numpy(float)
                clean = np.where(np.isnan(raw), np.array(item["fill"]), raw)
                x = np.column_stack(
                    [
                        np.ones(len(sample)),
                        (clean - np.array(item["mean"])) / np.array(item["scale"]),
                    ]
                )
                from .viirs_daily_model import logistic_predict

                p = logistic_predict(item, sample)
                item["covariance"] = np.linalg.pinv(
                    x.T @ (x * (p * (1 - p))[:, None])
                    + np.diag([1e-8] + [1.0] * (len(cols)))
                ).tolist()
                statistical[name] = item
        except (ResourceLimit, FloatingPointError, RuntimeError) as exc:
            training_error = type(exc).__name__ + ": " + str(exc)
    write_json(run.path / "statistical_models.json", statistical)
    del sample
    gc.collect()

    def probabilities(frame):
        result = baseline_probabilities(model, frame)
        if models or statistical:
            from .mask_model import posterior_probabilities
            from .viirs_daily_model import logistic_predict

            for name, path in models.items():
                ok = (
                    frame.weather_complete.to_numpy(bool)
                    if name == "pyro_weather"
                    else np.ones(len(frame), bool)
                )
                values = np.full(len(frame), np.nan)
                idx = np.flatnonzero(ok)
                for start in range(0, len(idx), 512):
                    batch = idx[start : start + 512]
                    values[batch] = posterior_probabilities(
                        path, frame.iloc[batch], 200, run.config["seed"]
                    ).mean(axis=0)
                result[name] = values
            for name, item in statistical.items():
                ok = (
                    frame.weather_complete.to_numpy(bool)
                    if name == "logistic_weather"
                    else np.ones(len(frame), bool)
                )
                values = np.full(len(frame), np.nan)
                values[ok] = logistic_predict(item, frame.loc[ok])
                result[name] = values
        return result

    def predict_split(split):
        for path in files:
            frame = pd.read_parquet(path)
            frame = (
                frame.loc[(frame.split == split) & frame.event.notna()]
                .copy()
                .reset_index(drop=True)
            )
            if frame.empty:
                continue
            output = frame[
                [
                    "region_id",
                    "parent_id",
                    "continent",
                    "spatial_group",
                    "legacy_context",
                    "period_start",
                    "event",
                    "weather_complete",
                ]
            ].copy()
            output.loc[output.region_id.isin(buffer), "spatial_group"] = "buffer"
            for name, p in probabilities(frame).items():
                output[name] = p
            yield frame, output

    validations = [out for _, out in predict_split("validation")]
    validation = pd.concat(validations, ignore_index=True)
    common = validation.loc[
        (validation.spatial_group == "train") & validation.weather_complete
    ]
    names = list(BASES) + list(models) + list(statistical)
    scores = {
        name: summarize(common.event.to_numpy(int), common[name].to_numpy(float))
        for name in names
        if common[name].notna().all() and len(common)
    }
    champion = min(scores, key=lambda k: scores[k]["brier"]) if scores else "seasonal"
    policy = {
        "version": "modis_native41_v6",
        "champion_common_weather": champion,
        "selection": "validation only; test unopened until this file is persisted",
        "validation_common_weather": scores,
        "criteria": run.config["release"],
        "groups": {},
    }
    for rid, g in validation.groupby("region_id"):
        y = g.event.to_numpy(int)
        candidates = list(BASES)
        if champion not in candidates and (
            champion not in ("pyro_weather", "logistic_weather")
            or g.weather_complete.all()
        ):
            candidates.append(champion)
        values = {name: summarize(y, g[name].to_numpy(float)) for name in candidates}
        cfg = run.config["release"]
        n = len(g)
        pos = int(y.sum())
        enough = (
            n >= cfg["min_n"] and cfg["min_class_n"] <= pos <= n - cfg["min_class_n"]
        )
        valid = {
            name: s
            for name, s in values.items()
            if enough
            and abs(s["mean_probability"] - s["observed_rate"])
            <= cfg["max_mean_calibration_error"]
            and s["brier"] <= values["seasonal"]["brier"] * cfg["max_brier_ratio"]
        }
        ring = footprint(meta[str(rid)])["coordinates"][0]
        if any(abs(x) > 180 or abs(y) > 90 for x, y in ring):
            valid = {}
        chosen = min(valid, key=lambda k: valid[k]["brier"]) if valid else "seasonal"
        static = {
            name: s
            for name, s in valid.items()
            if name in ("seasonal", "regional_seasonal", "global_frequency")
        }
        projection = min(static, key=lambda k: static[k]["brier"]) if static else None
        supported_months = [
            int(month)
            for month, n in meta[str(rid)]["observed_training_months"].items()
            if n >= 3
            and meta[str(rid)]["observed_validation_months"].get(month, 0) >= 1
        ]
        if not supported_months:
            projection = None
        policy["groups"][str(rid)] = {
            "eligible": bool(valid),
            "selected_model": chosen,
            "validation": values[chosen],
            "seasonal_eligible": "seasonal" in valid,
            "static_projection_model": projection,
            "static_projection_months": sorted(supported_months),
            "weather_available": bool(g.weather_complete.all()),
        }
    write_json(run.path / "release_policy.json", policy)
    for rid, info in meta.items():
        info["release"] = policy["groups"].get(
            rid, {"eligible": False, "static_projection_model": None}
        )
    write_json(run.path / "cell_catalog.json", meta)
    features = []
    for rid, info in meta.items():
        shape = footprint(info)
        if any(abs(x) > 180 or abs(y) > 90 for x, y in shape["coordinates"][0]):
            continue
        features.append(
            {
                "type": "Feature",
                "geometry": shape,
                "properties": {
                    "cell_id": rid,
                    "parent_id": info["parent_id"],
                    "continent": info["continent"],
                    "area_km2": info["area_km2"],
                    "forecast_supported": info["release"].get("eligible", False),
                    "static_projection_supported": info["release"].get(
                        "static_projection_model"
                    )
                    is not None,
                    "historical_observed_periods": info["training_n"],
                    "historical_positive_periods": info["training_positive"],
                    "historical_information_supported": info["training_n"] > 0,
                    "value_kind": "conditional_forecast"
                    if info["release"].get("eligible")
                    else "observed_history_only"
                    if info["training_n"] > 0
                    else "insufficient_data",
                },
            }
        )
    write_json(
        run.path / "coverage.geojson",
        {"type": "FeatureCollection", "features": features},
    )
    # Todas as escolhas estão congeladas antes de abrir o teste.
    test_parts = directory / "test_parts"
    test_parts.mkdir(exist_ok=True)
    tests = []
    for frame, output in predict_split("test"):
        ood = np.zeros(len(frame), bool)
        for column, (low, high) in bounds.items():
            if np.isfinite([low, high]).all():
                ood |= (
                    frame[column].notna() & ~frame[column].between(low, high)
                ).to_numpy()
        output["weather_out_of_distribution"] = ood
        routed = []
        supported = []
        for i, row in output.iterrows():
            entry = policy["groups"].get(row.region_id, {})
            choice = entry.get("selected_model", "seasonal")
            value = row.get(choice, np.nan)
            if (
                choice in ("pyro_weather", "logistic_weather")
                and row.weather_out_of_distribution
            ):
                value = np.nan
            if not np.isfinite(value):
                fallback = entry.get("static_projection_model")
                value = row.get(fallback, np.nan) if fallback else np.nan
            routed.append(value)
            supported.append(bool(entry.get("eligible", False) and np.isfinite(value)))
        output["routed_probability"] = routed
        output["release_eligible"] = supported
        name = str(output.parent_id.iloc[0])
        output.to_parquet(test_parts / (name + ".parquet"), index=False)
        tests.append(output)
    test = pd.concat(tests, ignore_index=True)
    result = {
        "version": "modis_native41_v6",
        "finished_utc": utc_now(),
        "event": "ao menos um pixel MOD14A2 com foco, condicionado a todos os 41×41 pixels terem máscara válida",
        "trained_rows_available": total[1],
        "trained_positive_available": total[0],
        "sampled_training": read_json(run.path / "training_sample.json"),
        "training_error": training_error,
        "champion_common_weather": champion,
        "validation_common_weather": scores,
        "test_all_baselines": {
            name: summarize(test.event, test[name]) for name in BASES
        },
        "test_common_weather": {},
        "released_cells": sum(v["eligible"] for v in policy["groups"].values()),
        "static_projection_cells": sum(
            v.get("static_projection_model") is not None
            for v in policy["groups"].values()
        ),
        "studied_feature_cells": sum(
            not v.get("history_only", False) for v in meta.values()
        ),
        "catalog_cells": len(meta),
        "historical_cells": sum(
            v["training_n"] > 0
            and all(
                abs(x) <= 180 and abs(y) <= 90
                for x, y in footprint(v)["coordinates"][0]
            )
            for v in meta.values()
        ),
        "historical_area_km2": sum(
            v["area_km2"]
            for v in meta.values()
            if v["training_n"] > 0
            and all(
                abs(x) <= 180 and abs(y) <= 90
                for x, y in footprint(v)["coordinates"][0]
            )
        ),
        "test_n": len(test),
        "test_positive": int(test.event.sum()),
        "by_continent": {},
        "by_group": {},
    }
    common_test = test.loc[test.weather_complete]
    for name in names:
        rows = common_test.loc[common_test[name].notna()]
        result["test_common_weather"][name] = summarize(rows.event, rows[name])
    eligible = test.loc[test.release_eligible]
    result["released_test"] = summarize(eligible.event, eligible.routed_probability)
    fresh = eligible.loc[~eligible.legacy_context]
    result["released_test_new_context"] = summarize(
        fresh.event, fresh.routed_probability
    )
    heldout = fresh.loc[fresh.spatial_group == "holdout"]
    result["released_test_spatial_holdout"] = summarize(
        heldout.event, heldout.routed_probability
    )
    spatial_all = test.loc[(test.spatial_group == "holdout") & ~test.legacy_context]
    spatial_weather = spatial_all.loc[spatial_all.weather_complete]
    result["spatial_holdout_unfiltered"] = {
        "interpretation": "Regiões novas; sem filtrar pela liberação local. Modelos globais escolhidos na validação das regiões de treino. A liberação local, separadamente, utiliza a validação 2023 da própria célula.",
        "all_baselines": {
            name: summarize(spatial_all.event, spatial_all[name]) for name in BASES
        },
        "common_weather": {
            name: summarize(spatial_weather.event, spatial_weather[name])
            for name in names
            if spatial_weather[name].notna().all()
        },
    }
    for column, key in [("continent", "by_continent"), ("spatial_group", "by_group")]:
        for name, g in eligible.groupby(column):
            result[key][str(name)] = summarize(g.event, g.routed_probability)
    result["released_area_km2"] = sum(
        meta[rid]["area_km2"] for rid, p in policy["groups"].items() if p["eligible"]
    )
    result["static_projection_area_km2"] = sum(
        meta[rid]["area_km2"]
        for rid, p in policy["groups"].items()
        if p.get("static_projection_model")
    )
    result["global_surface_percent"] = (
        result["released_area_km2"] / (4 * np.pi * 6371.007181**2) * 100
    )
    artifacts = [
        "baseline_model.json",
        "statistical_models.json",
        "feature_bounds.json",
        "release_policy.json",
        "cell_catalog.json",
        "training_sample.json",
        "coverage.geojson",
    ]
    artifacts += [str(p.relative_to(run.path)) for p in models.values()]
    registry = {
        "version": "modis_native41_v6",
        "config_sha256": run.config_hash,
        "created_utc": utc_now(),
        "artifacts": {p: sha256_file(run.path / p) for p in artifacts},
        "features": {str(p.relative_to(run.path)): sha256_file(p) for p in files},
        "code": {
            name: sha256_file(Path(__file__).with_name(name + ".py"))
            for name in (
                "expansion_grid",
                "expansion_features",
                "expansion_sparse_weather",
                "expansion_evaluate",
                "expansion_predict",
            )
        },
    }
    write_json(run.path / "model_registry.json", registry)
    write_json(directory / "metrics.json", result)
    run.stage(
        "expansion_evaluation",
        "complete",
        released_cells=result["released_cells"],
        test_n=len(test),
    )
    return result
