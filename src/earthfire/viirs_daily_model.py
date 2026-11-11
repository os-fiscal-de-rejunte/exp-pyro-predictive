"""Treinamento e avaliação da detecção diária VIIRS condicionada à máscara válida."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from scipy.special import expit

from .io import read_json, replace_with_retry, sha256_file, write_json
from .mask_evaluate import _baseline_fit, _baselines, _calibration, _score
from .mask_features import FEATURES_BASE, FEATURES_WEATHER
from .mask_model import eligible, posterior_probabilities, train
from .runtime import Run

TARGET = ("ao menos um pixel VIIRS/NOAA-20 VJ114A1.002 com foco no composto diário "
          "de 1 km, condicionado a todos os pixels da área terem classes válidas")


def fit_logistic(frame: pd.DataFrame, features: list[str], penalty: float = 1.0) -> dict[str, Any]:
    """Referência logística regularizada, com transformações obtidas somente no treino."""
    raw = frame[features].to_numpy(dtype=float)
    fill = np.array([np.nanmedian(x) if np.isfinite(x).any() else 0. for x in raw.T])
    clean = np.where(np.isnan(raw), fill, raw)
    mean, scale = clean.mean(axis=0), clean.std(axis=0)
    scale[scale < 1e-6] = 1.
    x = np.column_stack([np.ones(len(clean)), (clean - mean) / scale])
    y = frame.event.to_numpy(dtype=float)

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        z = x @ beta
        loss = float(np.logaddexp(0., z).sum() - y @ z + .5 * penalty * (beta[1:] @ beta[1:]))
        grad = x.T @ (expit(z) - y)
        grad[1:] += penalty * beta[1:]
        return loss, grad

    result = minimize(objective, np.zeros(x.shape[1]), jac=True, method="L-BFGS-B", options={"maxiter": 500})
    if not result.success:
        raise RuntimeError(f"Referência logística não convergiu: {result.message}")
    return {"features": features, "fill": fill.tolist(), "mean": mean.tolist(), "scale": scale.tolist(),
            "coef": result.x.tolist(), "penalty": penalty}


def logistic_predict(model: dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    raw = frame[model["features"]].to_numpy(dtype=float)
    clean = np.where(np.isnan(raw), np.asarray(model["fill"]), raw)
    x = np.column_stack([np.ones(len(clean)), (clean - np.asarray(model["mean"])) / np.asarray(model["scale"])])
    return expit(x @ np.asarray(model["coef"]))


def persistence_fit(frame: pd.DataFrame, global_rate: float) -> dict[int, float]:
    history = frame.dropna(subset=["past_viirs_mask_event"])
    return {int(key): float((group.event.sum() + 2 * global_rate) / (len(group) + 2))
            for key, group in history.groupby("past_viirs_mask_event")}


def training_readiness(examples: pd.DataFrame) -> dict[str, Any]:
    frame = eligible(examples)
    fit = frame.loc[(frame.split == "train") & (frame.spatial_group == "train")]
    validation = frame.loc[frame.split == "validation"]
    train_positive = int(fit.event.sum())
    validation_positive = int(validation.event.sum())
    return {"eligible": bool(len(fit) >= 100 and min(train_positive, len(fit) - train_positive) >= 10 and
                             len(validation) >= 30 and min(validation_positive, len(validation) - validation_positive) >= 5),
            "train_n": len(fit), "train_positive": train_positive,
            "validation_n": len(validation), "validation_positive": validation_positive}


def evaluate_baselines(run: Run, examples: pd.DataFrame) -> dict[str, Any]:
    frame = eligible(examples)
    fit = frame.loc[(frame.split == "train") & (frame.spatial_group == "train")].copy()
    valid = frame.loc[frame.split == "validation"].copy()
    if not training_readiness(examples)["eligible"]:
        raise ValueError("Dados insuficientes para comparar referências na validação.")
    reference = _baseline_fit(fit)
    persistence = persistence_fit(fit, reference["global"])
    baselines = _baselines(reference, valid)
    predictions = {"global_frequency": baselines["global_frequency"], "seasonal": baselines["seasonal"],
        "modis_transition": baselines["persistence"],
        "persistence": np.array([persistence.get(int(value), reference["global"]) if not pd.isna(value)
                                 else reference["global"] for value in valid.past_viirs_mask_event]),
        "logistic_no_weather": logistic_predict(fit_logistic(fit, list(FEATURES_BASE)), valid),
        "logistic_weather": logistic_predict(fit_logistic(fit, list(FEATURES_BASE + FEATURES_WEATHER)), valid)}
    result = {"scope": "validação 2023 antes do treinamento probabilístico; parâmetros ajustados no treino",
              "training_n": len(fit), "validation_n": len(valid),
              "metrics": {key: _score(valid.event.to_numpy(dtype=int), value) for key, value in predictions.items()}}
    output = run.path / "evaluation/viirs_baselines_before_training.json"
    write_json(output, result)
    run.stage("viirs_daily_baselines", "complete", output=str(output), validation_n=len(valid))
    return result


def train_model(run: Run, examples: pd.DataFrame, steps: int) -> Path:
    summary = read_json(run.path / "processed/viirs_daily_summary.json")
    if (sha256_file(run.path / "processed/mask_examples.parquet") != summary["examples_sha256"] or
            sha256_file(Path(__file__).with_name("viirs_daily_data.py")) != summary["input_sha256"]["code"]):
        raise ValueError("Exemplos não conferem com o manifesto de preparação.")
    ready = training_readiness(examples)
    if not ready["eligible"]:
        raise ValueError(f"Dados insuficientes para treinamento e validação: {ready}")
    path = train(run, examples, weather=True, steps=steps)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    # O motor Bayesiano é genérico; o metadado precisa refletir o novo alvo.
    if saved["model_version"] == "mod14a2_8day_weather_v3":
        saved.update({"model_version": "vj114a1_daily_weather_v4b", "target": TARGET,
                      "daily_model_code_sha256": sha256_file(Path(__file__)),
                      "preparation_code_sha256": summary["input_sha256"]["code"],
                      "preparation_manifest_sha256": sha256_file(run.path / "processed/viirs_daily_summary.json")})
        temp = path.with_name(path.name + ".metadata.tmp")
        torch.save(saved, temp)
        replace_with_retry(temp, path)
    if (saved["model_version"] != "vj114a1_daily_weather_v4b" or
            saved.get("daily_model_code_sha256") != sha256_file(Path(__file__))):
        raise ValueError("Checkpoint não pertence ao alvo VIIRS diário.")
    run.stage("viirs_daily_model", "complete", checkpoint=str(path), model_version=saved["model_version"],
              training_n=saved["n_train"], positives=saved["n_positive"])
    return path


def evaluate(run: Run, examples: pd.DataFrame, model_path: Path, samples: int = 400) -> dict[str, Any]:
    summary = read_json(run.path / "processed/viirs_daily_summary.json")
    saved = torch.load(model_path, map_location="cpu", weights_only=False)
    if (saved.get("model_version") != "vj114a1_daily_weather_v4b" or
            saved.get("daily_model_code_sha256") != sha256_file(Path(__file__)) or
            saved.get("preparation_manifest_sha256") != sha256_file(run.path / "processed/viirs_daily_summary.json") or
            sha256_file(Path(__file__).with_name("viirs_daily_data.py")) != summary["input_sha256"]["code"] or
            saved.get("data_sha256") != summary["examples_sha256"]):
        raise ValueError("Modelo e preparação diária não são compatíveis.")
    frame = eligible(examples)
    fit = frame.loc[(frame.split == "train") & (frame.spatial_group == "train")].copy()
    if len(fit) < 100 or fit.event.sum() < 10:
        raise ValueError("Treino insuficiente para avaliar o modelo diário.")
    baseline = _baseline_fit(fit)
    persistence = persistence_fit(fit, baseline["global"])
    no_weather = fit_logistic(fit, list(FEATURES_BASE))
    with_weather = fit_logistic(fit, list(FEATURES_BASE + FEATURES_WEATHER))
    out = run.path / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "statistical_baselines.json", {"no_weather": no_weather, "weather": with_weather,
                                                   "persistence": persistence})
    report: dict[str, Any] = {"event": TARGET, "unit": "probabilidade condicional por área e dia",
        "limitations": ["A máscara futura integralmente válida não é garantida.",
                        "Produto diário de 1 km não é a detecção FIRMS de 375 m.",
                        "Amostra sistemática, oito áreas e arquivos históricos sujeitos a revisão."],
        "training_n": len(fit), "training_positive": int(fit.event.sum()), "splits": {}}
    for split in ("validation", "test"):
        data = frame.loc[frame.split == split].copy()
        if data.empty:
            raise ValueError(f"Sem exemplos elegíveis em {split}.")
        data["event"] = data.event.astype(int)
        draws = posterior_probabilities(model_path, data, n_samples=samples, seed=run.config["seed"])
        data["model_probability"] = draws.mean(axis=0)
        data["parameter_lower_80"] = np.quantile(draws, .1, axis=0)
        data["parameter_upper_80"] = np.quantile(draws, .9, axis=0)
        reference = _baselines(baseline, data)
        data["global_frequency"] = reference["global_frequency"]
        data["seasonal"] = reference["seasonal"]
        # Essa referência é uma transição a partir do MODIS, não persistência VIIRS diária.
        data["modis_transition"] = reference["persistence"]
        data["persistence"] = [persistence.get(int(value), baseline["global"]) if not pd.isna(value)
                               else baseline["global"] for value in data.past_viirs_mask_event]
        data["logistic_no_weather"] = logistic_predict(no_weather, data)
        data["logistic_weather"] = logistic_predict(with_weather, data)
        names = ["model_probability", "global_frequency", "seasonal", "persistence", "modis_transition",
                 "logistic_no_weather", "logistic_weather"]
        entry: dict[str, Any] = {"overall": {name: _score(data.event.to_numpy(), data[name].to_numpy()) for name in names},
            "calibration": _calibration(data.event.to_numpy(), data.model_probability.to_numpy()), "spatial_groups": {},
            "regions": {}, "months": {}}
        for column, key in (("spatial_group", "spatial_groups"), ("region_id", "regions")):
            for name, group in data.groupby(column):
                entry[key][str(name)] = {metric: _score(group.event.to_numpy(), group[metric].to_numpy())
                                         for metric in ("model_probability", "seasonal", "logistic_no_weather", "logistic_weather")}
        for month, group in data.groupby(data.period_start.dt.month):
            entry["months"][str(month)] = _score(group.event.to_numpy(), group.model_probability.to_numpy())
        report["splits"][split] = entry
        data.to_parquet(out / f"viirs_predictions_{split}.parquet", index=False)
        if split == "validation":
            support = {}
            for group in ("train", "holdout"):
                metrics = entry["spatial_groups"].get(group, {})
                model, seasonal = metrics.get("model_probability", {}), metrics.get("seasonal", {})
                support[group] = {"n": model.get("n", 0), "positives": model.get("positive", 0),
                    "eligible": bool(model.get("n", 0) >= 30 and model.get("positive", 0) >= 5 and
                                     model.get("brier", float("inf")) <= seasonal.get("brier", float("-inf")))}
            report["validation_support"] = support
            write_json(out / "viirs_validation_before_test.json", {"support": support, "metrics": entry})
    write_json(out / "viirs_metrics.json", report)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True, sharey=True)
    test = pd.read_parquet(out / "viirs_predictions_test.parquet")
    for ax, group in zip(axes, ("train", "holdout")):
        subset = test.loc[test.spatial_group == group]
        ax.plot([0, 1], [0, 1], "--", color="gray")
        if len(subset):
            bins = _calibration(subset.event.to_numpy(), subset.model_probability.to_numpy())
            ax.scatter([x["mean_probability"] for x in bins], [x["observed_rate"] for x in bins])
            for b in bins:
                ax.annotate(str(b["n"]), (b["mean_probability"], b["observed_rate"]), xytext=(3, 3), textcoords="offset points")
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Probabilidade condicional", title=group)
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Frequência observada")
    fig.suptitle("Calibração retrospectiva VIIRS 1 km; números são tamanhos das faixas")
    fig.tight_layout()
    fig.savefig(out / "viirs_calibration_test.png", dpi=160)
    plt.close(fig)
    run.stage("viirs_daily_evaluate", "complete", training_n=len(fit), test_n=len(test))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/viirs_daily_v4b.json"))
    parser.add_argument("--run-id", default="viirs_daily_v4b_20261111")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("readiness")
    sub.add_parser("baselines")
    t = sub.add_parser("train")
    t.add_argument("--steps", type=int, default=1000)
    e = sub.add_parser("evaluate")
    e.add_argument("--samples", type=int, default=400)
    args = parser.parse_args()
    run = Run(args.config, args.run_id)
    examples = pd.read_parquet(run.path / "processed/mask_examples.parquet")
    path = run.path / "models/mask_weather.pt"
    if args.command == "readiness":
        print(json.dumps(training_readiness(examples), ensure_ascii=False, indent=2))
    elif args.command == "baselines":
        print(json.dumps(evaluate_baselines(run, examples), ensure_ascii=False, indent=2))
    elif args.command == "train":
        print(json.dumps({"checkpoint": str(train_model(run, examples, args.steps))}, ensure_ascii=False))
    else:
        result = evaluate(run, examples, path, args.samples)
        print(json.dumps({"training_n": result["training_n"], "validation_support": result["validation_support"],
                          "test": result["splits"]["test"]["overall"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
