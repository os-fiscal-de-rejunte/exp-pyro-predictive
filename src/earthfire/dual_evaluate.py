"""Avaliação em tempo e em células espaciais mantidas fora do ajuste."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import logsumexp

from .dual_features import BASE, SENSOR, with_climatology
from .dual_model import posterior_counts
from .evaluate import _crps, _point_metrics, _summary
from .io import write_json
from .runtime import Run
from .sensor_versions import annotate_versions, build_version_index


def _fit_ridge(train: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
    x = train[columns].to_numpy(dtype=float)
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-8] = 1.
    x = np.column_stack([np.ones(len(x)), (x - mean) / scale])
    penalty = np.eye(x.shape[1]); penalty[0, 0] = 0
    coef = np.linalg.solve(x.T @ x + penalty, x.T @ np.log1p(train.detections.to_numpy(dtype=float)))
    return {"columns": columns, "mean": mean, "scale": scale, "coef": coef}


def _ridge(model: dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    x = (frame[model["columns"]].to_numpy(dtype=float) - model["mean"]) / model["scale"]
    x = np.column_stack([np.ones(len(x)), x])
    return np.maximum(1., np.expm1(np.clip(x @ model["coef"], -10, 12)))


def _baseline_fit(train: pd.DataFrame) -> dict[str, Any]:
    frame = train.copy(); frame["month"] = frame.week_start.dt.month
    return {"seasonal": frame.groupby(["region_id", "month"]).detections.mean().to_dict(),
            "seasonal_pooled": frame.groupby("month").detections.mean().to_dict(), "overall": float(frame.detections.mean()),
            "with_modis": _fit_ridge(frame, BASE + SENSOR), "without_modis": _fit_ridge(frame, BASE)}


def _baseline_predict(model: dict[str, Any], frame: pd.DataFrame) -> dict[str, np.ndarray]:
    seasonal = np.array([model["seasonal"].get((rid, int(month)), model["seasonal_pooled"].get(int(month), model["overall"]))
                         for rid, month in zip(frame.region_id, frame.week_start.dt.month)])
    return {"persistence": frame.viirs_lag1.to_numpy(dtype=float), "seasonal": seasonal,
            "log_linear_ridge": _ridge(model["with_modis"], frame),
            "ridge_without_modis": _ridge(model["without_modis"], frame)}


def _metrics(frame: pd.DataFrame) -> dict[str, Any]:
    summary = _summary(frame)
    summary["ridge_without_modis"] = _point_metrics(frame.detections.to_numpy(dtype=float),
                                                     frame.ridge_without_modis.to_numpy(dtype=float))
    return summary


def evaluate(run: Run, examples: pd.DataFrame, climate: pd.DataFrame, models: dict[str, Path], n_samples: int = 400) -> dict[str, Any]:
    out = run.path / "evaluation_v2"; out.mkdir(parents=True, exist_ok=True)
    train = examples.loc[(examples.split == "train") & (examples.spatial_group == "train_regions")].dropna(subset=models_columns_weather())
    baseline = _baseline_fit(train)
    version_index = build_version_index(pd.read_parquet(run.path / "processed_v2/records.parquet"))
    thresholds = train.detections.quantile([.5, .9]).to_numpy()
    result: dict[str, Any] = {"target": "VIIRS/NOAA-20: contagem da semana condicionada à semana alvo positiva e à anterior positiva em MODIS/Terra e VIIRS/NOAA-20",
        "weather": "histórico defasado e trajetória futura sazonal hipotética estimada apenas no treino",
        "training_n": len(train), "test_horizon_weeks": 1, "validation": {}, "test": {},
        "training_dispersion": {r: {"n": len(g), "mean": float(g.detections.mean()),
            "variance_to_mean": float(g.detections.var(ddof=1) / g.detections.mean()) if len(g) > 1 and g.detections.mean() else None}
            for r, g in train.groupby("region_id")}}
    for split in ("validation", "test"):
        source = examples.loc[examples.split == split].copy()
        if source.empty:
            raise ValueError(f"Conjunto {split} vazio")
        source = with_climatology(source, climate)
        source, known_versions = annotate_versions(source, train, version_index)
        result["sensor_versions_train"] = known_versions
        for variant, model in models.items():
            frame = source.copy()
            for name, values in _baseline_predict(baseline, frame).items(): frame[name] = values
            draws, logp = posterior_counts(model, frame, n_samples=n_samples, seed=run.config["seed"])
            frame["pyro_mean"] = draws.mean(axis=0)
            frame["pyro_median"] = np.quantile(draws, .5, axis=0)
            for level, lower, upper in ((80,.1,.9),(95,.025,.975)):
                frame[f"lower_{level}"] = np.quantile(draws, lower, axis=0)
                frame[f"upper_{level}"] = np.quantile(draws, upper, axis=0)
            frame["crps"] = _crps(draws, frame.detections.to_numpy())
            frame["negative_log_score"] = -(logsumexp(logp, axis=0) - np.log(len(logp)))
            frame["season"] = ((frame.week_start.dt.month % 12) // 3 + 1).astype(int)
            frame["intensity"] = np.where(frame.detections <= thresholds[0], "até mediana treino",
                                   np.where(frame.detections <= thresholds[1], "mediana–p90 treino", "acima p90 treino"))
            entry = {"overall": _metrics(frame), "spatial_groups": {}, "by_region": {}, "by_season": {},
                     "by_intensity": {}, "by_sensor_version_support": {}}
            for col, field in (("spatial_group","spatial_groups"), ("region_id","by_region"),
                               ("season","by_season"), ("intensity","by_intensity"),
                               ("sensor_version_support", "by_sensor_version_support")):
                entry[field] = {str(k): _metrics(g) for k, g in frame.groupby(col) if len(g)}
            result[split][variant] = entry
            frame.to_parquet(out / f"predictions_{split}_{variant}.parquet", index=False)
        if split == "validation":
            for name in models:
                if "train_regions" not in result[split][name]["spatial_groups"]:
                    raise ValueError("Validação temporal sem regiões de treino.")
            chosen = min(models, key=lambda v: result[split][v]["spatial_groups"]["train_regions"]["crps"])
            selection = {"variant": chosen, "criterion": "validation_crps_train_regions", "locked_before_test": True}
            write_json(out / "selection.json", selection)
            result["selection"] = selection
    write_json(out / "metrics.json", result)
    selected = result["selection"]["variant"]
    plotted = pd.read_parquet(out / f"predictions_test_{selected}.parquet")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True, sharey=True)
    for ax, group in zip(axes, ("train_regions", "holdout")):
        subset = plotted.loc[plotted.spatial_group == group]
        ax.scatter(subset.detections, subset.pyro_median, alpha=.65, s=18)
        maxval = max(float(subset.detections.max()) if len(subset) else 1., float(subset.pyro_median.max()) if len(subset) else 1., 1.)
        ax.plot([1, maxval], [1, maxval], "--", color="gray", linewidth=1)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(alpha=.2)
        ax.set_title("Células treinadas" if group == "train_regions" else "Células fora do treino")
        ax.set_xlabel("Detecções observadas / semana")
    axes[0].set_ylabel("Mediana prevista condicional")
    fig.suptitle("Teste retrospectivo: apenas semanas positivas elegíveis")
    fig.tight_layout(); fig.savefig(out / "prediction_test.png", dpi=160); plt.close(fig)
    run.stage("evaluation_v2", "complete", selected=selected, metrics=str(out / "metrics.json"))
    return result


def models_columns_weather() -> list[str]:
    from .dual_model import model_columns
    return model_columns(True)
