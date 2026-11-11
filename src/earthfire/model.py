"""Regressão bayesiana NB truncada em zero, via PyTorch e Pyro."""
from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
import torch
import torch.nn.functional as F
from pyro.infer import SVI, Predictive, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.optim import ClippedAdam

from .features import feature_names
from . import features as feature_module
from .io import replace_with_retry, sha256_file, utc_now
from .runtime import ResourceLimit, Run


def _model(x: torch.Tensor, y: torch.Tensor | None = None, prior_intercept: float = 2.0) -> None:
    intercept = pyro.sample("intercept", dist.Normal(torch.tensor(prior_intercept), torch.tensor(1.5)))
    beta = pyro.sample("beta", dist.Normal(torch.zeros(x.shape[1]), torch.full((x.shape[1],), .5)).to_event(1))
    log_alpha = pyro.sample("log_alpha", dist.Normal(torch.tensor(1.0), torch.tensor(1.0)))
    eta = (intercept + x @ beta).clamp(-12, 12)
    alpha = log_alpha.clamp(-6, 8).exp()
    logits = eta - alpha.log()
    with pyro.plate("weeks", x.shape[0]):
        if y is not None:
            pyro.sample("positive_count", dist.NegativeBinomial(total_count=alpha, logits=logits), obs=y)
            # Condiciona na inclusão da semana, que requer ao menos uma detecção.
            log_p_zero = -alpha * F.softplus(logits)
            pyro.factor("condition_positive", -torch.log(-torch.expm1(log_p_zero)))


def _atomic_torch_save(obj: dict[str, Any], path: Path) -> None:
    temp = path.with_name(path.name + ".tmp")
    torch.save(obj, temp)
    replace_with_retry(temp, path)


def train(run: Run, examples: pd.DataFrame, include_weather: bool, steps: int = 900, resume: bool = True) -> Path:
    variant = "weather" if include_weather else "no_weather"
    target = run.path / "models" / f"{variant}.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    train_data = examples.loc[examples.split == "train"].copy()
    if len(train_data) < 30 or (train_data.detections <= 0).any():
        raise ValueError("Dados insuficientes: ao menos 30 semanas positivas de treino são necessárias.")
    regions = sorted(train_data.region_id.unique().tolist())
    columns = feature_names(regions, include_weather)
    raw = train_data[columns].to_numpy(dtype=np.float32)
    if not np.isfinite(raw).all():
        raise ValueError("Features de treino ausentes ou não finitas.")
    mean, scale = raw.mean(axis=0), raw.std(axis=0)
    scale[scale < 1e-6] = 1.0
    prior = float(np.log(np.median(train_data.detections)))
    x = torch.tensor((raw - mean) / scale, dtype=torch.float32)
    y = torch.tensor(train_data.detections.to_numpy(dtype=np.float32))
    dataset_hash = sha256_file(run.path / "processed" / "examples.parquet")
    code_hash = sha256_file(Path(__file__))
    features_hash = sha256_file(Path(feature_module.__file__))
    pyro.clear_param_store()
    pyro.set_rng_seed(int(run.config["seed"]))
    # Matrizes pequenas não justificam saturar os núcleos durante a auditoria.
    torch.set_num_threads(min(run.config["operational"]["max_cpu_threads"], 2))
    guide = AutoNormal(_model)
    optim = ClippedAdam({"lr": 0.015, "clip_norm": 10.0})
    svi = SVI(_model, guide, optim, loss=Trace_ELBO())
    start_step, loss_history = 0, []
    if target.exists() and resume:
        checkpoint = torch.load(target, map_location="cpu", weights_only=False)
        if (checkpoint["config_sha256"] != run.config_hash or checkpoint["data_sha256"] != dataset_hash or
            checkpoint["features"] != columns or checkpoint.get("model_code_sha256") != code_hash or
            checkpoint.get("features_code_sha256") != features_hash):
            raise ResourceLimit("Checkpoint incompatível com dados, código ou configuração.")
        if checkpoint.get("complete"):
            return target
        pyro.get_param_store().set_state(checkpoint["param_store"])
        optim.set_state(checkpoint["optim_state"])
        torch.random.set_rng_state(checkpoint["torch_rng"])
        start_step, loss_history = checkpoint["step"], checkpoint["loss_history"]
    else:
        run.claim_training(variant)
        if run.manifest["training_runs"] >= run.config["operational"]["max_training_runs"]:
            raise ResourceLimit("Limite de treinamentos atingido.")
        run.manifest["training_runs"] += 1
        run.save()
    run.stage("train_" + variant, "running", samples=len(y), steps=steps, features=columns)
    started, last_checkpoint = time.monotonic(), time.monotonic()

    def save(step: int, complete: bool) -> None:
        _atomic_torch_save({"format_version": "1", "model_version": "nb_positive_" + variant + "_v1" + ("_DEMO" if run.run_id.startswith("quick_") else ""),
            "run_id": run.run_id, "trained_utc": utc_now(), "config_sha256": run.config_hash,
            "data_sha256": dataset_hash, "model_code_sha256": code_hash, "features_code_sha256": features_hash,
            "features": columns, "regions": regions,
            "feature_mean": mean.tolist(), "feature_scale": scale.tolist(),
            "feature_min": raw.min(axis=0).tolist(), "feature_max": raw.max(axis=0).tolist(),
            "training_end": run.config["split"]["train_end"], "preprocessing_version": "weekly-grid-v1",
            "prior_intercept": prior, "step": step, "complete": complete,
            "param_store": pyro.get_param_store().get_state(), "optim_state": optim.get_state(),
            "torch_rng": torch.random.get_rng_state(), "loss_history": loss_history}, target)

    last_step = start_step
    try:
        for step in range(start_step + 1, steps + 1):
            if step % 25 == 0:
                run.guard()
                if time.monotonic() - started > run.config["operational"]["training_seconds"]:
                    raise ResourceLimit("Tempo máximo individual de treinamento atingido.")
            loss = float(svi.step(x, y, prior))
            if not math.isfinite(loss):
                raise FloatingPointError(f"ELBO não finita no passo {step}")
            last_step = step
            if step == 1 or step % 50 == 0:
                loss_history.append({"step": step, "loss": loss})
            if step % 250 == 0 or time.monotonic() - last_checkpoint >= run.config["operational"]["checkpoint_seconds"]:
                save(step, False)
                last_checkpoint = time.monotonic()
        save(last_step, True)
    except (ResourceLimit, KeyboardInterrupt, FloatingPointError):
        save(last_step, False)
        run.stage("train_" + variant, "interrupted", checkpoint=str(target), step=last_step)
        raise
    run.stage("train_" + variant, "complete", checkpoint=str(target), step=last_step, seconds=time.monotonic() - started)
    return target


def posterior_counts(path: Path, frame: pd.DataFrame, n_samples: int = 500, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Amostras preditivas de contagens e log-massa do alvo condicional a Y>0."""
    model = torch.load(path, map_location="cpu", weights_only=False)
    if not model.get("complete"):
        raise ValueError("Checkpoint ainda não concluiu o treino.")
    if model.get("model_code_sha256") != sha256_file(Path(__file__)):
        raise ValueError("Código do modelo não corresponde ao checkpoint; use a versão original ou retreine.")
    if model.get("features_code_sha256") != sha256_file(Path(feature_module.__file__)):
        raise ValueError("Código de features não corresponde ao checkpoint; use a versão original ou retreine.")
    raw = frame[model["features"]].to_numpy(dtype=np.float32)
    if not np.isfinite(raw).all():
        raise ValueError("Features ausentes ou não finitas.")
    x = torch.tensor((raw - np.asarray(model["feature_mean"])) / np.asarray(model["feature_scale"]), dtype=torch.float32)
    pyro.clear_param_store()
    pyro.get_param_store().set_state(model["param_store"])
    pyro.set_rng_seed(seed)
    guide = AutoNormal(_model)
    y = torch.tensor(frame["detections"].to_numpy(dtype=np.float32)) if "detections" in frame else torch.ones(len(frame))
    with torch.no_grad():
        post = Predictive(_model, guide=guide, num_samples=n_samples,
                          return_sites=("intercept", "beta", "log_alpha"))(x, y, model["prior_intercept"])
        beta = post["beta"].reshape(n_samples, -1)
        intercept = post["intercept"].reshape(n_samples, -1)[:, 0]
        log_alpha = post["log_alpha"].reshape(n_samples, -1)[:, 0].clamp(-6, 8)
        alpha = log_alpha.exp()[:, None]
        eta = (intercept[:, None] + beta @ x.T).clamp(-12, 12)
        logits = eta - log_alpha[:, None]
        distribution = torch.distributions.NegativeBinomial(total_count=alpha, logits=logits)
        draws = distribution.sample()
        for _ in range(1000):
            zeros = draws == 0
            if not zeros.any(): break
            draws = torch.where(zeros, distribution.sample(), draws)
        if (draws == 0).any():
            raise RuntimeError("Amostragem condicional não convergiu para valores positivos.")
        log_p_zero = -alpha * F.softplus(logits)
        log_prob = distribution.log_prob(y[None, :]) - torch.log(-torch.expm1(log_p_zero))
    return draws.cpu().numpy(), log_prob.cpu().numpy()
