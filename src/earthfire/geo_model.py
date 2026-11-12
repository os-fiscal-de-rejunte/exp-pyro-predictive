"""Experimento Pyro com e sem geografia, em coorte idêntica e splits congelados."""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
import torch
from pyro.infer import Predictive, SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.optim import ClippedAdam

from .expansion_evaluate import summarize
from .io import sha256_file, utc_now, write_json

BASE_FEATURES = [
    "latitude",
    "longitude",
    "season_sin",
    "season_cos",
    "latitude_season_sin",
    "latitude_season_cos",
]
GEO_FEATURES = [
    "geo_ndvi",
    "geo_evi",
    "geo_tree_percent",
    "geo_nontree_percent",
    "geo_bare_percent",
    "geo_elevation_m",
    "geo_elevation_range_m",
] + [f"geo_landcover_{k:02d}" for k in range(1, 18)]


def model(x, y=None, prior=0.0):
    intercept = pyro.sample(
        "intercept", dist.Normal(torch.tensor(prior), torch.tensor(1.5))
    )
    weights = pyro.sample(
        "weights",
        dist.Normal(torch.zeros(x.shape[1]), torch.ones(x.shape[1]) * 0.5).to_event(1),
    )
    if y is not None:
        with pyro.plate("examples", x.shape[0]):
            pyro.sample(
                "event",
                dist.Bernoulli(logits=(intercept + x @ weights).clamp(-15, 15)),
                obs=y,
            )


def with_temporal_interactions(frame: pd.DataFrame) -> pd.DataFrame:
    """Permite sazonalidade diferente entre hemisférios sem usar dados futuros."""
    return frame.assign(
        latitude_season_sin=frame.latitude * frame.season_sin,
        latitude_season_cos=frame.latitude * frame.season_cos,
    )


def transform(frame: pd.DataFrame, saved: dict) -> np.ndarray:
    frame = with_temporal_interactions(frame)
    raw = frame[saved["features"]].to_numpy(float, copy=True)
    if np.isinf(raw).any():
        raise ValueError("Features geográficas infinitas.")
    for column in saved.get("unobserved_features", []):
        raw[:, saved["features"].index(column)] = np.nan
    clean = np.where(np.isnan(raw), np.array(saved["fill"]), raw)
    # Indicadores de ausência são parte do treinamento, não observações inventadas.
    data = np.column_stack([clean, np.isnan(raw).astype(float)])
    return ((data - np.array(saved["mean"])) / np.array(saved["scale"])).astype(
        np.float32
    )


def train(
    frame: pd.DataFrame,
    features: list[str],
    path: Path,
    config: dict,
    data_hash: str,
    guard=None,
) -> Path:
    frame = with_temporal_interactions(frame)
    torch.set_num_threads(2)
    fit = frame.loc[
        (frame.split == "train")
        & (frame.spatial_group == "train")
        & frame.event.notna()
    ]
    if (
        len(fit) < 100
        or min(int(fit.event.sum()), len(fit) - int(fit.event.sum())) < 10
    ):
        raise ValueError(
            "Treino geográfico insuficiente; mínimo 100 casos e 10 de cada classe."
        )
    fit = fit.sample(
        n=min(len(fit), config["max_training_examples"]), random_state=config["seed"]
    ).sort_index()
    raw = fit[features].to_numpy(float)
    fill = np.array([np.nanmedian(c) if np.isfinite(c).any() else 0.0 for c in raw.T])
    data = np.column_stack(
        [np.where(np.isnan(raw), fill, raw), np.isnan(raw).astype(float)]
    )
    mean, scale = data.mean(axis=0), data.std(axis=0)
    scale[scale < 1e-6] = 1
    saved = {
        "features": features,
        "unobserved_features": [
            name
            for name, values in zip(features, raw.T)
            if not np.isfinite(values).any()
        ],
        "fill": fill.tolist(),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "seed": config["seed"],
        "data_sha256": data_hash,
        "code_sha256": sha256_file(Path(__file__)),
        "feature_min": {
            name: float(fit[name].min()) if fit[name].notna().any() else None
            for name in features
        },
        "feature_max": {
            name: float(fit[name].max()) if fit[name].notna().any() else None
            for name in features
        },
        "training_n": len(fit),
        "training_regions": sorted(fit.region_id.unique()),
        "prior": math.log((fit.event.sum() + 0.5) / (len(fit) - fit.event.sum() + 0.5)),
        "complete": False,
    }
    x = torch.tensor(transform(fit, saved))
    y = torch.tensor(fit.event.to_numpy(np.float32))
    pyro.clear_param_store()
    pyro.set_rng_seed(config["seed"])
    guide = AutoNormal(model)
    optimizer = ClippedAdam({"lr": 0.01, "clip_norm": 10.0})
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    start, losses = 0, []
    if path.exists():
        previous = torch.load(path, map_location="cpu", weights_only=False)
        if (
            any(
                previous[k] != saved[k]
                for k in ("features", "data_sha256", "code_sha256")
            )
            or previous["steps_requested"] != config["steps"]
        ):
            raise ValueError("Checkpoint geográfico incompatível; use outro diretório.")
        if previous["complete"]:
            return path
        pyro.get_param_store().set_state(previous["params"])
        optimizer.set_state(previous["optimizer"])
        torch.random.set_rng_state(previous["torch_rng"])
        start, losses = previous["step"], previous["losses"]
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    for step in range(start + 1, config["steps"] + 1):
        if step % 25 == 0 and guard is not None:
            guard()
        loss = float(svi.step(x, y, saved["prior"]))
        if not math.isfinite(loss):
            raise FloatingPointError("ELBO geográfica inválida.")
        if step % 100 == 0:
            losses.append({"step": step, "loss": loss})
        if step % 250 == 0 or step == config["steps"]:
            saved.update(
                params=pyro.get_param_store().get_state(),
                optimizer=optimizer.get_state(),
                torch_rng=torch.random.get_rng_state(),
                step=step,
                steps_requested=config["steps"],
                complete=step == config["steps"],
                losses=losses,
                trained_utc=utc_now(),
            )
            tmp = path.with_suffix(".tmp")
            torch.save(saved, tmp)
            tmp.replace(path)
        if time.monotonic() - started > config["max_training_seconds"]:
            raise TimeoutError(
                "Tempo de treinamento geográfico atingido; checkpoint periódico preservado."
            )
    return path


def posterior(path: Path, frame: pd.DataFrame, samples: int = 200) -> np.ndarray:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if not saved["complete"] or saved["code_sha256"] != sha256_file(Path(__file__)):
        raise ValueError("Modelo geográfico incompleto ou incompatível.")
    torch.set_num_threads(2)
    pyro.clear_param_store()
    pyro.set_rng_seed(saved["seed"])
    pyro.get_param_store().set_state(saved["params"])
    guide = AutoNormal(model)
    x = torch.tensor(transform(frame, saved))
    with torch.no_grad():
        values = Predictive(
            model,
            guide=guide,
            num_samples=samples,
            return_sites=("intercept", "weights"),
        )(x, None, saved["prior"])
        return torch.sigmoid(
            values["intercept"].reshape(samples, -1)[:, :1]
            + values["weights"].reshape(samples, -1) @ x.T
        ).numpy()


def experiment(frame: pd.DataFrame, output: Path, config: dict, run=None) -> dict:
    frame = with_temporal_interactions(frame)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "metrics.json").exists():
        raise ValueError("Experimento concluído está congelado; use outro diretório.")
    # Mesmos casos para a ablação; não exige terreno fora da faixa de cobertura.
    selected = frame.loc[
        frame.event.notna()
        & ((frame.geo_landcover_sample_n >= 5) | (frame.geo_ndvi_sample_n >= 5))
    ].copy()
    valid = selected.loc[
        (selected.split == "validation") & (selected.spatial_group == "train")
    ]
    if len(valid) < 30 or valid.event.nunique() < 2:
        raise ValueError("Validação geográfica insuficiente.")
    path = output / "examples.parquet"
    selected.to_parquet(path, index=False)
    variants = {
        "without_geography": BASE_FEATURES,
        "with_geography": BASE_FEATURES + GEO_FEATURES,
    }
    validation, predictions, hashes = {}, {}, {}
    for name, columns in variants.items():
        if run is not None:
            run.guard(require_training_seconds=60)
            run.claim_training("geo_" + name)
        checkpoint = train(
            selected,
            columns,
            output / (name + ".pt"),
            config,
            sha256_file(path),
            guard=run.guard if run is not None else None,
        )
        hashes[name + ".pt"] = sha256_file(checkpoint)
        probability = np.concatenate(
            [
                posterior(checkpoint, valid.iloc[i : i + 512]).mean(axis=0)
                for i in range(0, len(valid), 512)
            ]
        )
        validation[name] = summarize(valid.event, probability)
    chosen = min(validation, key=lambda k: validation[k]["brier"])
    write_json(
        output / "selection.json",
        {
            "selected": chosen,
            "validation": validation,
            "frozen_utc": utc_now(),
            "prior_test_results_already_known": True,
        },
    )
    # Não ajusta parâmetros nem seleção depois de abrir esse teste histórico.
    test = selected.loc[selected.split == "test"]
    metrics = {
        "selected": chosen,
        "validation": validation,
        "test": {},
        "spatial_test": {},
        "test_by_continent": {},
        "cohort": {
            str(split): {
                "n": len(part),
                "positive": int(part.event.sum()),
                "cells": int(part.region_id.nunique()),
            }
            for split, part in selected.groupby("split")
        },
        "interpretation": "Experimento retrospectivo em recortes; teste v6 já conhecido. Não demonstra acurácia mundial ou validação operacional.",
        "data_n": len(selected),
        "trained_utc": utc_now(),
    }
    for name in variants:
        values = (
            np.concatenate(
                [
                    posterior(output / (name + ".pt"), test.iloc[i : i + 512]).mean(
                        axis=0
                    )
                    for i in range(0, len(test), 512)
                ]
            )
            if len(test)
            else np.array([])
        )
        metrics["test"][name] = summarize(test.event, values)
        heldout = test.spatial_group.to_numpy() == "holdout"
        metrics["spatial_test"][name] = summarize(
            test.loc[heldout].event, values[heldout]
        )
        if "continent" in test:
            metrics["test_by_continent"][name] = {
                str(continent): summarize(
                    test.loc[test.continent == continent].event,
                    values[(test.continent == continent).to_numpy()],
                )
                for continent in sorted(test.continent.unique())
            }
        predictions[name] = values
    test = test[["region_id", "period_start", "spatial_group", "event"]].copy()
    for name, values in predictions.items():
        test[name] = values
    test.to_parquet(output / "predictions.parquet", index=False)
    write_json(
        output / "registry.json",
        {
            "artifacts": hashes,
            "examples_sha256": sha256_file(path),
            "config": config,
            "code": {
                name: sha256_file(Path(__file__).with_name(name + ".py"))
                for name in ("geo_model", "geography")
            },
            "selected": chosen,
        },
    )
    write_json(output / "metrics.json", metrics)
    return metrics
