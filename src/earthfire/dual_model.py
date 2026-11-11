"""Regressão NB bayesiana para contagens observadas condicionadas a positividade."""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
import torch
import torch.nn.functional as F
from pyro.infer import Predictive, SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.optim import ClippedAdam

from . import dual_features, features, power_zarr
from .dual_features import BASE, FUTURE, PAST, ROLLING, SENSOR
from .io import replace_with_retry, sha256_file, utc_now
from .runtime import ResourceLimit, Run


def model_columns(include_weather: bool) -> list[str]:
    return BASE + SENSOR + (PAST + ROLLING + FUTURE if include_weather else [])


def _model(x: torch.Tensor, y: torch.Tensor | None = None, prior_intercept: float = 2.) -> None:
    intercept = pyro.sample("intercept", dist.Normal(torch.tensor(prior_intercept), torch.tensor(1.5)))
    beta = pyro.sample("beta", dist.Normal(torch.zeros(x.shape[1]), torch.full((x.shape[1],), .5)).to_event(1))
    log_alpha = pyro.sample("log_alpha", dist.Normal(torch.tensor(1.), torch.tensor(1.)))
    eta = (intercept + x @ beta).clamp(-12, 12)
    alpha = log_alpha.clamp(-6, 8).exp()
    logits = eta - alpha.log()
    with pyro.plate("weeks", x.shape[0]):
        if y is not None:
            pyro.sample("positive_count", dist.NegativeBinomial(total_count=alpha, logits=logits), obs=y)
            log_p_zero = -alpha * F.softplus(logits)
            pyro.factor("condition_positive", -torch.log(-torch.expm1(log_p_zero)))


def _code_hashes() -> dict[str, str]:
    return {"model": sha256_file(Path(__file__)), "dual_features": sha256_file(Path(dual_features.__file__)),
            "temporal_features": sha256_file(Path(features.__file__)), "power_zarr": sha256_file(Path(power_zarr.__file__))}


def _save(obj: dict[str, Any], path: Path) -> None:
    temp = path.with_name(path.name + ".tmp")
    torch.save(obj, temp)
    replace_with_retry(temp, path)


def train(run: Run, examples: pd.DataFrame, include_weather: bool, steps: int = 900) -> Path:
    variant = "weather" if include_weather else "no_weather"
    path = run.path / "models_v2" / f"{variant}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    # Ambas as variantes são ajustadas no mesmo conjunto completo.
    fit = examples.loc[(examples.split == "train") & (examples.spatial_group == "train_regions")].copy()
    complete = fit[model_columns(True)].notna().all(axis=1)
    fit = fit.loc[complete]
    if len(fit) < 100 or (fit.detections <= 0).any():
        raise ValueError("Menos de 100 semanas positivas elegíveis no treino.")
    columns = model_columns(include_weather)
    raw = fit[columns].to_numpy(dtype=np.float32)
    if not np.isfinite(raw).all():
        raise ValueError("Features de treino não finitas.")
    mean, scale = raw.mean(axis=0), raw.std(axis=0)
    scale[scale < 1e-6] = 1.
    x = torch.tensor((raw - mean) / scale, dtype=torch.float32)
    y = torch.tensor(fit.detections.to_numpy(dtype=np.float32))
    prior = float(np.log(np.median(fit.detections)))
    data_hash = sha256_file(run.path / "processed_v2/examples.parquet")
    code = _code_hashes()
    pyro.clear_param_store(); pyro.set_rng_seed(int(run.config["seed"]))
    torch.set_num_threads(min(run.config["operational"]["max_cpu_threads"], 2))
    guide = AutoNormal(_model)
    optimizer = ClippedAdam({"lr": .015, "clip_norm": 10.})
    svi = SVI(_model, guide, optimizer, loss=Trace_ELBO())
    start, losses = 0, []
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if (saved["config_sha256"] != run.config_hash or saved["data_sha256"] != data_hash or
            saved["code_sha256"] != code or saved["features"] != columns):
            raise ResourceLimit("Checkpoint v2 incompatível com configuração, dados ou código.")
        if saved.get("complete"):
            return path
        pyro.get_param_store().set_state(saved["param_store"])
        optimizer.set_state(saved["optim_state"])
        torch.random.set_rng_state(saved["torch_rng"])
        start, losses = saved["step"], saved["loss_history"]
    else:
        run.claim_training("v2_" + variant)
        run.manifest["training_runs"] += 1
        run.save()
    run.stage("train_v2_" + variant, "running", n=len(y), steps=steps, features=columns)
    started, last_checkpoint, last_step = time.monotonic(), time.monotonic(), start

    def save(step: int, complete_flag: bool) -> None:
        _save({"format_version": "2", "model_version": "dual_nb_positive_" + variant + "_v2",
               "run_id": run.run_id, "trained_utc": utc_now(), "config_sha256": run.config_hash,
               "data_sha256": data_hash, "code_sha256": code, "features": columns,
               "feature_mean": mean.tolist(), "feature_scale": scale.tolist(),
               "feature_min": raw.min(axis=0).tolist(), "feature_max": raw.max(axis=0).tolist(),
               "training_regions": sorted(fit.region_id.unique().tolist()),
               "spatial_holdout_regions": run.config["spatial_holdout_regions"],
               "training_end": run.config["split"]["train_end"], "n_train": len(fit),
               "prior_intercept": prior, "step": step, "complete": complete_flag,
               "param_store": pyro.get_param_store().get_state(), "optim_state": optimizer.get_state(),
               "torch_rng": torch.random.get_rng_state(), "loss_history": losses}, path)

    try:
        for step in range(start + 1, steps + 1):
            if step % 25 == 0:
                run.guard()
                if time.monotonic() - started > run.config["operational"]["training_seconds"]:
                    raise ResourceLimit("Tempo máximo do treinamento atingido.")
            loss = float(svi.step(x, y, prior))
            if not math.isfinite(loss):
                raise FloatingPointError(f"ELBO não finita no passo {step}")
            last_step = step
            if step == 1 or step % 50 == 0:
                losses.append({"step": step, "loss": loss})
            if step % 250 == 0 or time.monotonic() - last_checkpoint >= run.config["operational"]["checkpoint_seconds"]:
                save(step, False); last_checkpoint = time.monotonic()
        save(last_step, True)
    except (ResourceLimit, KeyboardInterrupt, FloatingPointError):
        save(last_step, False)
        run.stage("train_v2_" + variant, "interrupted", step=last_step, checkpoint=str(path))
        raise
    run.stage("train_v2_" + variant, "complete", checkpoint=str(path), step=last_step,
              n_train=len(fit), seconds=time.monotonic() - started)
    return path


def posterior_counts(path: Path, frame: pd.DataFrame, n_samples: int = 400, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if not saved.get("complete") or saved["code_sha256"] != _code_hashes():
        raise ValueError("Checkpoint incompleto ou código incompatível.")
    raw = frame[saved["features"]].to_numpy(dtype=np.float32)
    if not np.isfinite(raw).all():
        raise ValueError("Features ausentes ou inválidas.")
    x = torch.tensor((raw - np.asarray(saved["feature_mean"])) / np.asarray(saved["feature_scale"]), dtype=torch.float32)
    y = torch.tensor(frame.detections.to_numpy(dtype=np.float32)) if "detections" in frame else torch.ones(len(frame))
    pyro.clear_param_store(); pyro.get_param_store().set_state(saved["param_store"]); pyro.set_rng_seed(seed)
    guide = AutoNormal(_model)
    with torch.no_grad():
        post = Predictive(_model, guide=guide, num_samples=n_samples,
                          return_sites=("intercept", "beta", "log_alpha"))(x, y, saved["prior_intercept"])
        beta = post["beta"].reshape(n_samples, -1)
        intercept = post["intercept"].reshape(n_samples, -1)[:, 0]
        log_alpha = post["log_alpha"].reshape(n_samples, -1)[:, 0].clamp(-6, 8)
        alpha = log_alpha.exp()[:, None]
        logits = (intercept[:, None] + beta @ x.T).clamp(-12, 12) - log_alpha[:, None]
        distribution = torch.distributions.NegativeBinomial(total_count=alpha, logits=logits)
        draws = distribution.sample()
        for _ in range(1000):
            zero = draws == 0
            if not zero.any(): break
            draws = torch.where(zero, distribution.sample(), draws)
        if (draws == 0).any():
            raise RuntimeError("Amostragem de contagem positiva não convergiu.")
        log_zero = -alpha * F.softplus(logits)
        log_prob = distribution.log_prob(y[None, :]) - torch.log(-torch.expm1(log_zero))
    return draws.numpy(), log_prob.numpy()
