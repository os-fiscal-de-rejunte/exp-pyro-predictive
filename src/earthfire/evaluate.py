"""Avaliação cronológica e baselines ajustados exclusivamente no treino."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import logsumexp

from .features import BASE_FEATURES, replace_future_by_climatology
from .io import write_json
from .model import posterior_counts
from .runtime import Run


def _point_metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    err = prediction - y
    return {"mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2)))}


def _crps(draws: np.ndarray, y: np.ndarray) -> np.ndarray:
    ordered = np.sort(draws, axis=0)
    n = len(draws)
    first = np.abs(draws - y[None, :]).mean(axis=0)
    coefficients = (2 * np.arange(1, n + 1) - n - 1)[:, None]
    return first - (coefficients * ordered).sum(axis=0) / (n * n)


def fit_baselines(train: pd.DataFrame) -> dict[str, Any]:
    data = train.copy()
    data["month"] = data.week_start.dt.month
    seasonal = data.groupby(["region_id", "month"]).detections.mean().to_dict()
    means = data.groupby("region_id").detections.mean().to_dict()
    regions = sorted(data.region_id.unique())
    columns = BASE_FEATURES + ["region=" + r for r in regions[1:]]
    x = np.column_stack([np.ones(len(data)), data[columns].to_numpy(dtype=float)])
    penalty = np.eye(x.shape[1]); penalty[0, 0] = 0
    coef = np.linalg.solve(x.T @ x + penalty, x.T @ np.log1p(data.detections.to_numpy(dtype=float)))
    return {"seasonal": seasonal, "region_means": means, "overall": float(data.detections.mean()),
            "coef": coef, "columns": columns}


def baseline_predictions(model: dict[str, Any], frame: pd.DataFrame) -> dict[str, np.ndarray]:
    months = frame.week_start.dt.month
    seasonal = np.array([model["seasonal"].get((r, int(m)), model["region_means"].get(r, model["overall"]))
                         for r, m in zip(frame.region_id, months)])
    x = np.column_stack([np.ones(len(frame)), frame[model["columns"]].to_numpy(dtype=float)])
    statistical = np.maximum(1.0, np.expm1(np.clip(x @ model["coef"], -10, 12)))
    return {"persistence": frame.lag1.to_numpy(dtype=float), "seasonal": seasonal, "log_linear_ridge": statistical}


def _summary(frame: pd.DataFrame) -> dict[str, Any]:
    y = frame.detections.to_numpy(dtype=float)
    result: dict[str, Any] = {"n": len(frame)}
    for column in ["persistence", "seasonal", "log_linear_ridge", "pyro_median"]:
        result[column] = _point_metrics(y, frame[column].to_numpy(dtype=float))
    reference = result["seasonal"]["mae"]
    result["pyro_median"]["mae_relative_to_seasonal"] = result["pyro_median"]["mae"] / reference if reference else None
    for level in (80, 95):
        lo, hi = frame[f"lower_{level}"], frame[f"upper_{level}"]
        result[f"interval_{level}"] = {"coverage": float(((y >= lo) & (y <= hi)).mean()),
                                        "mean_width": float((hi - lo).mean())}
    result["crps"] = float(frame.crps.mean())
    result["negative_log_score"] = float(frame.negative_log_score.mean())
    return result


def evaluate(run: Run, examples: pd.DataFrame, climate: pd.DataFrame, paths: dict[str, Path], n_samples: int = 400) -> dict[str, Any]:
    output = run.path / "evaluation"; output.mkdir(parents=True, exist_ok=True)
    baseline = fit_baselines(examples.loc[examples.split == "train"])
    selection: dict[str, Any] = {}
    dispersion = {}
    for name, group in examples.loc[examples.split == "train"].groupby("region_id"):
        mean = float(group.detections.mean())
        variance = float(group.detections.var(ddof=1))
        dispersion[name] = {"n": len(group), "mean": mean, "variance": variance,
                            "variance_to_mean": variance / mean if mean else None}
    all_results: dict[str, Any] = {"definition": "contagem semanal positiva de VIIRS/NOAA-20, condicionada à semana anterior positiva",
        "weather_evaluation": "trajetória climatológica estimada apenas no treino; reanálise histórica revisada não é arquivo de previsão operacional",
        "dispersion_train": dispersion, "dispersion_caveat": "razão variância/média agregada pode refletir sazonalidade e seleção positiva; não é teste formal de Poisson",
        "sampling": n_samples, "validation": {}, "test": {}}
    # A seleção é persistida antes de se abrir o conjunto de teste.
    for split in ("validation", "test"):
        subset = examples.loc[examples.split == split].copy()
        if subset.empty:
            raise ValueError(f"Nenhum exemplo elegível em {split}.")
        future = replace_future_by_climatology(subset, climate)
        for variant, path in paths.items():
            frame = future.copy()
            for name, values in baseline_predictions(baseline, frame).items():
                frame[name] = values
            draws, logp = posterior_counts(path, frame, n_samples=n_samples, seed=run.config["seed"])
            frame["pyro_mean"] = draws.mean(axis=0)
            frame["pyro_median"] = np.quantile(draws, .5, axis=0)
            for level, lower, upper in ((80, .1, .9), (95, .025, .975)):
                frame[f"lower_{level}"] = np.quantile(draws, lower, axis=0)
                frame[f"upper_{level}"] = np.quantile(draws, upper, axis=0)
            frame["crps"] = _crps(draws, frame.detections.to_numpy())
            frame["negative_log_score"] = -(logsumexp(logp, axis=0) - np.log(len(logp)))
            frame["season"] = ((frame.week_start.dt.month % 12) // 3 + 1).astype(int)
            # Faixas de intensidade definidas apenas a partir do treino.
            thresholds = examples.loc[examples.split == "train", "detections"].quantile([.5, .9]).to_numpy()
            frame["intensity"] = np.where(frame.detections <= thresholds[0], "até mediana treino",
                                  np.where(frame.detections <= thresholds[1], "mediana–p90 treino", "acima p90 treino"))
            result = {"overall": _summary(frame), "by_region": {}, "by_season": {}, "by_intensity": {}, "horizon_weeks": 1}
            for key, result_key in (("region_id", "by_region"), ("season", "by_season"), ("intensity", "by_intensity")):
                result[result_key] = {str(label): _summary(group) for label, group in frame.groupby(key) if len(group)}
            frame.to_parquet(output / f"predictions_{split}_{variant}.parquet", index=False)
            all_results[split][variant] = result
        if split == "validation":
            best = min(paths, key=lambda v: all_results[split][v]["overall"]["crps"])
            selection = {"variant": best, "criterion": "validation_crps", "locked_before_test": True}
            write_json(output / "selection.json", selection)
    all_results["selection"] = selection
    write_json(output / "metrics.json", all_results)
    selected = selection["variant"]
    plot = pd.read_parquet(output / f"predictions_test_{selected}.parquet")
    region = plot.region_id.value_counts().index[0]
    plot = plot.loc[plot.region_id == region].sort_values("week_start")
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.scatter(plot.week_start, plot.detections, s=12, label="Registros observados (semanas positivas)")
    ax.vlines(plot.week_start, plot.lower_80, plot.upper_80, alpha=.35, lw=1, label="Intervalo preditivo condicional 80%")
    ax.scatter(plot.week_start, plot.pyro_median, s=9, label="Mediana condicional")
    prefix = "DEMONSTRATIVO — " if run.run_id.startswith("quick_") else ""
    ax.set(title=f"{prefix}Teste retrospectivo — {region}", ylabel="Detecções/semana", xlabel="Início da semana (UTC)")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(output / "prediction_test.png", dpi=150); plt.close(fig)
    run.stage("evaluation", "complete", selected=selected, metrics=str(output / "metrics.json"))
    return all_results
