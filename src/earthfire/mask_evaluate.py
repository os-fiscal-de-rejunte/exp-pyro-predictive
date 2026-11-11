"""Avaliação temporal e espacial da ocorrência MODIS condicionada à máscara válida."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import rankdata

from .io import write_json
from .mask_model import eligible, posterior_probabilities
from .runtime import Run


def _score(y: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(probability, dtype=float), 1e-8, 1 - 1e-8)
    n1, n0 = int(y.sum()), int(len(y) - y.sum())
    auc = float((rankdata(p)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else None
    order = np.argsort(-p, kind="stable")
    ranked_y = y[order]
    sorted_p = p[order]
    boundaries = np.r_[np.flatnonzero(np.diff(sorted_p) != 0), len(y) - 1]
    true_positive = np.cumsum(ranked_y)[boundaries]
    precision = true_positive / (boundaries + 1)
    recall = true_positive / n1 if n1 else np.zeros(len(boundaries))
    ap = float((np.diff(np.r_[0., recall]) * precision).sum()) if n1 else None
    return {"n": len(y), "positive": n1, "observed_rate": float(y.mean()),
            "mean_probability": float(p.mean()), "brier": float(np.mean((p - y) ** 2)),
            "log_loss": float(np.mean(-y * np.log(p) - (1 - y) * np.log1p(-p))),
            "roc_auc": auc, "average_precision": ap}


def _calibration(y: np.ndarray, p: np.ndarray) -> list[dict[str, Any]]:
    frame = pd.DataFrame({"event": y, "probability": p})
    frame["bin"] = pd.cut(frame.probability, [-.001, .05, .1, .2, .4, .6, 1.], include_lowest=True)
    return [{"bin": str(k), "n": len(g), "mean_probability": float(g.probability.mean()),
             "observed_rate": float(g.event.mean())} for k,g in frame.groupby("bin", observed=True)]


def _baseline_fit(train: pd.DataFrame) -> dict[str, Any]:
    frame = train.copy()
    frame["month"] = frame.period_start.dt.month
    global_probability = float((frame.event.sum() + .5) / (len(frame) + 1))
    def probs(grouped: pd.core.groupby.DataFrameGroupBy) -> dict[Any, float]:
        return {key: float((group.event.sum() + 2 * global_probability) / (len(group) + 2)) for key, group in grouped}
    history = frame.dropna(subset=["past_modis_event"])
    return {"global": global_probability,
            "month": probs(frame.groupby("month")),
            "month_region": probs(frame.groupby(["region_id", "month"])),
            "transition": probs(history.groupby("past_modis_event"))}


def _baselines(model: dict[str, Any], frame: pd.DataFrame) -> dict[str, np.ndarray]:
    month = frame.period_start.dt.month.to_numpy()
    climatology = np.array([model["month_region"].get((r, int(m)), model["month"].get(int(m), model["global"]))
                            for r,m in zip(frame.region_id, month)])
    persistence = np.array([model["transition"].get(float(v), model["global"]) if not pd.isna(v) else model["global"]
                            for v in frame.past_modis_event])
    return {"global_frequency": np.full(len(frame), model["global"]),
            "seasonal": climatology, "persistence": persistence}


def evaluate_baselines(run: Run, examples: pd.DataFrame) -> dict[str, Any]:
    frame = eligible(examples)
    train = frame.loc[(frame.split == "train") & (frame.spatial_group == "train")]
    validation = frame.loc[(frame.split == "validation") & (frame.spatial_group == "train")]
    if train.empty or validation.empty:
        raise ValueError("Dados insuficientes para as referências de treino e validação.")
    baselines = _baselines(_baseline_fit(train), validation)
    report = {"scope": "validação temporal 2023 nas áreas de treino, antes do ajuste probabilístico",
              "training_n": len(train), "validation_n": len(validation),
              "metrics": {name: _score(validation.event.to_numpy(dtype=int), values) for name,values in baselines.items()}}
    output = run.path / "evaluation/baselines_before_training.json"
    write_json(output, report)
    run.stage("mask_baselines", "complete", output=str(output), training_n=len(train), validation_n=len(validation))
    return report


def evaluate(run: Run, examples: pd.DataFrame, models: dict[str, Path], n_samples: int = 400) -> dict[str, Any]:
    frame = eligible(examples)
    train = frame.loc[(frame.split == "train") & (frame.spatial_group == "train")]
    baseline = _baseline_fit(train)
    out = run.path / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"event": "pelo menos um pixel MOD14A2 marcado como foco no composto, condicionado a todos os pixels terem máscara válida",
        "unit": "probabilidade condicional por área e período MOD14A2 de até 8 dias",
        "limitation": "A disponibilidade futura não é conhecida; não é chance incondicional de fogo nem cobertura VIIRS.",
        "training_n": len(train), "training_positive": int(train.event.sum()), "validation": {}, "test": {}}
    for split in ("validation", "test"):
        data = frame.loc[frame.split == split].copy()
        if data.empty:
            raise ValueError(f"Amostra vazia na avaliação: {split}")
        data["event"] = data.event.astype(int)
        y = data.event.to_numpy()
        baseline_predictions = _baselines(baseline, data)
        for variant, path in models.items():
            draw = posterior_probabilities(path, data, n_samples=n_samples, seed=run.config["seed"])
            p = draw.mean(axis=0)
            output = data.copy()
            output["probability_given_full_mask"] = p
            output["parameter_lower_80"] = np.quantile(draw, .1, axis=0)
            output["parameter_upper_80"] = np.quantile(draw, .9, axis=0)
            for name, pred in baseline_predictions.items():
                output[name] = pred
            entry: dict[str, Any] = {"overall": {"model": _score(y, p),
                **{name: _score(y, values) for name,values in baseline_predictions.items()}},
                "calibration": _calibration(y, p), "spatial_groups": {}, "regions": {}, "months": {}}
            for col, field in (("spatial_group", "spatial_groups"), ("region_id", "regions")):
                for key, group in output.groupby(col):
                    yy, pp = group.event.to_numpy(), group.probability_given_full_mask.to_numpy()
                    entry[field][str(key)] = {"model": _score(yy, pp),
                        "seasonal": _score(yy, group.seasonal.to_numpy())}
            for key, group in output.groupby(output.period_start.dt.month):
                entry["months"][str(key)] = _score(group.event.to_numpy(), group.probability_given_full_mask.to_numpy())
            result[split][variant] = entry
            output.to_parquet(out / f"predictions_{split}_{variant}.parquet", index=False)
        if split == "validation":
            # O critério é registrado antes de qualquer processamento do teste.
            if any("train" not in result[split][v]["spatial_groups"] for v in models):
                raise ValueError("Sem validação temporal para regiões de treino.")
            selected = min(models, key=lambda v: result[split][v]["spatial_groups"]["train"]["model"]["brier"])
            result["selection"] = {"variant": selected, "criterion": "validation_brier_train_regions", "locked_before_test": True}
            write_json(out / "selection.json", result["selection"])
    write_json(out / "metrics.json", result)
    selected = result["selection"]["variant"]
    tested = pd.read_parquet(out / f"predictions_test_{selected}.parquet")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True, sharey=True)
    for ax, group in zip(axes, ("train", "holdout")):
        subset = tested.loc[tested.spatial_group == group]
        bins = _calibration(subset.event.to_numpy(), subset.probability_given_full_mask.to_numpy()) if len(subset) else []
        ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
        if bins:
            ax.scatter([x["mean_probability"] for x in bins], [x["observed_rate"] for x in bins], s=40)
            for item in bins:
                ax.annotate(str(item["n"]), (item["mean_probability"], item["observed_rate"]), xytext=(4, 4), textcoords="offset points")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(alpha=.2)
        ax.set_title("Áreas de treino" if group == "train" else "Áreas reservadas")
        ax.set_xlabel("Probabilidade média condicional")
    axes[0].set_ylabel("Fração observada nos períodos válidos")
    fig.suptitle("Calibração retrospectiva MODIS: números indicam amostras por faixa")
    fig.tight_layout()
    fig.savefig(out / "calibration_test.png", dpi=160)
    plt.close(fig)
    run.stage("mask_evaluation", "complete", selected=result["selection"]["variant"],
              training_n=len(train), training_positive=int(train.event.sum()))
    return result
