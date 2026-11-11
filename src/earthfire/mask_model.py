"""Classificador bayesiano para evento MODIS em máscaras integralmente válidas."""
from __future__ import annotations

import inspect
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
import torch
from pyro.infer import Predictive, SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.optim import ClippedAdam

from . import mask_features, nasa_complement
from .io import replace_with_retry, sha256_file, utc_now
from .runtime import ResourceLimit, Run


def columns(weather: bool) -> list[str]:
    return list(mask_features.FEATURES_BASE) + (list(mask_features.FEATURES_WEATHER) if weather else [])


def eligible(examples: pd.DataFrame) -> pd.DataFrame:
    """Ambas as variantes usam os mesmos exemplos com meteorologia retrospectiva."""
    finite_weather = np.isfinite(examples[list(mask_features.FEATURES_WEATHER)].to_numpy(dtype=float)).all(axis=1)
    target = examples.event.isin([0, 1])
    return examples.loc[finite_weather & target].copy()


def _model(x: torch.Tensor, y: torch.Tensor | None = None, prior_intercept: float = 0.) -> None:
    intercept = pyro.sample("intercept", dist.Normal(torch.tensor(prior_intercept), torch.tensor(1.5)))
    beta = pyro.sample("beta", dist.Normal(torch.zeros(x.shape[1]), torch.full((x.shape[1],), 0.75)).to_event(1))
    if y is not None:
        with pyro.plate("periods", x.shape[0]):
            pyro.sample("event", dist.Bernoulli(logits=(intercept + x @ beta).clamp(-15, 15)), obs=y)


def code_hashes() -> dict[str, str]:
    return {"model": sha256_file(Path(__file__)),
            "features": sha256_file(Path(inspect.getfile(mask_features))),
            "mask_classes": sha256_file(Path(inspect.getfile(nasa_complement)))}


def transform(frame: pd.DataFrame, saved: dict[str, Any]) -> np.ndarray:
    raw = frame[saved["features"]].to_numpy(dtype=np.float32)
    if np.isinf(raw).any():
        raise ValueError("Features infinitas.")
    fill = np.asarray(saved["fill"], dtype=np.float32)
    x = np.where(np.isnan(raw), fill, raw)
    return (x - np.asarray(saved["mean"], dtype=np.float32)) / np.asarray(saved["scale"], dtype=np.float32)


def train(run: Run, examples: pd.DataFrame, weather: bool, steps: int = 1000) -> Path:
    variant = "weather" if weather else "no_weather"
    target = run.path / "models" / f"mask_{variant}.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    fit = eligible(examples)
    fit = fit.loc[(fit.split == "train") & (fit.spatial_group == "train")].copy()
    positives = int((fit.event == 1).sum())
    negatives = int((fit.event == 0).sum())
    if len(fit) < 100 or min(positives, negatives) < 10:
        raise ValueError(f"Dados insuficientes no treino: {positives} positivos e {negatives} negativos válidos.")
    names = columns(weather)
    raw = fit[names].to_numpy(dtype=np.float32)
    fill = np.array([np.nanmedian(col) if np.isfinite(col).any() else 0. for col in raw.T], dtype=np.float32)
    clean = np.where(np.isnan(raw), fill, raw)
    if not np.isfinite(clean).all():
        raise ValueError("Features inválidas no ajuste.")
    mean, scale = clean.mean(axis=0), clean.std(axis=0)
    scale[scale < 1e-6] = 1.
    x = torch.tensor((clean - mean) / scale, dtype=torch.float32)
    y = torch.tensor(fit.event.to_numpy(dtype=np.float32), dtype=torch.float32)
    prior = float(math.log((positives + 0.5) / (negatives + 0.5)))
    data_hash = sha256_file(run.path / "processed/mask_examples.parquet")
    code_hash = code_hashes()
    torch.set_num_threads(min(2, run.config["operational"]["max_cpu_threads"]))
    pyro.clear_param_store()
    pyro.set_rng_seed(int(run.config["seed"]))
    guide = AutoNormal(_model)
    optimizer = ClippedAdam({"lr": .012, "clip_norm": 10.})
    svi = SVI(_model, guide, optimizer, loss=Trace_ELBO())
    start, losses = 0, []
    if target.exists():
        saved = torch.load(target, map_location="cpu", weights_only=False)
        if (saved["config_sha256"] != run.config_hash or saved["data_sha256"] != data_hash or
                saved["code_sha256"] != code_hash or saved["features"] != names):
            raise ResourceLimit("Checkpoint de máscara incompatível com dados, código ou configuração.")
        if saved.get("complete"):
            return target
        pyro.get_param_store().set_state(saved["param_store"])
        optimizer.set_state(saved["optim_state"])
        torch.random.set_rng_state(saved["torch_rng"])
        start, losses = saved["step"], saved["loss_history"]
    else:
        run.claim_training("mask_" + variant)
        run.manifest["training_runs"] += 1
        run.save()
    run.stage("train_mask_" + variant, "running", n=len(fit), positives=positives, steps=steps)
    started, last_checkpoint, last_step = time.monotonic(), time.monotonic(), start

    def save(step: int, complete: bool) -> None:
        target_tmp = target.with_name(target.name + ".tmp")
        torch.save({"format_version": "1", "model_version": "mod14a2_8day_" + variant + "_v3",
                    "trained_utc": utc_now(), "config_sha256": run.config_hash, "data_sha256": data_hash,
                    "code_sha256": code_hash, "features": names, "fill": fill.tolist(),
                    "mean": mean.tolist(), "scale": scale.tolist(),
                    "feature_min": np.nanmin(clean, axis=0).tolist(), "feature_max": np.nanmax(clean, axis=0).tolist(),
                    "training_regions": sorted(fit.region_id.unique().tolist()), "prior_intercept": prior,
                    "n_train": len(fit), "n_positive": positives, "step": step, "complete": complete,
                    "param_store": pyro.get_param_store().get_state(), "optim_state": optimizer.get_state(),
                    "torch_rng": torch.random.get_rng_state(), "loss_history": losses}, target_tmp)
        replace_with_retry(target_tmp, target)

    try:
        for step in range(start + 1, steps + 1):
            if step % 25 == 0:
                run.guard()
                if time.monotonic() - started > run.config["operational"]["training_seconds"]:
                    raise ResourceLimit("Limite de tempo de treinamento atingido.")
            loss = float(svi.step(x, y, prior))
            if not math.isfinite(loss):
                raise FloatingPointError(f"ELBO inválida no passo {step}")
            last_step = step
            if step == 1 or step % 50 == 0:
                losses.append({"step": step, "loss": loss})
            if step % 250 == 0 or time.monotonic() - last_checkpoint >= run.config["operational"]["checkpoint_seconds"]:
                save(step, False)
                last_checkpoint = time.monotonic()
        save(last_step, True)
    except (ResourceLimit, KeyboardInterrupt, FloatingPointError):
        save(last_step, False)
        run.stage("train_mask_" + variant, "interrupted", step=last_step, checkpoint=str(target))
        raise
    run.stage("train_mask_" + variant, "complete", n=len(fit), positives=positives, step=last_step,
              seconds=time.monotonic() - started, checkpoint=str(target))
    return target


def posterior_probabilities(path: Path, frame: pd.DataFrame, n_samples: int = 400, seed: int = 0) -> np.ndarray:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if not saved.get("complete") or saved["code_sha256"] != code_hashes():
        raise ValueError("Checkpoint incompleto ou incompatível com o código.")
    x = torch.tensor(transform(frame, saved), dtype=torch.float32)
    pyro.clear_param_store()
    pyro.get_param_store().set_state(saved["param_store"])
    pyro.set_rng_seed(seed)
    guide = AutoNormal(_model)
    with torch.no_grad():
        post = Predictive(_model, guide=guide, num_samples=n_samples, return_sites=("intercept", "beta"))(
            x, None, saved["prior_intercept"])
        intercept = post["intercept"].reshape(n_samples, -1)[:, 0]
        beta = post["beta"].reshape(n_samples, -1)
        return torch.sigmoid(intercept[:, None] + beta @ x.T).numpy()
