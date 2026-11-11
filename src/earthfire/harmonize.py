"""Comparação pareada por plataforma; nunca preenche ausência ou estabelece verdade."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
import torch
from pyro.infer import SVI, Predictive, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.optim import Adam


def matched_training_weeks(weekly: pd.DataFrame, train_end: str) -> pd.DataFrame:
    data = weekly.copy()
    data["week_start"] = pd.to_datetime(data["week_start"])
    # Apenas semanas totalmente concluídas até o corte e com ambos os sensores positivos.
    data = data.loc[data.week_start + pd.Timedelta(days=6) <= pd.Timestamp(train_end)]
    reference = data.loc[(data["product"] == "modis") & (data.satellite == "Terra"), ["region_id", "week_start", "detections"]]
    other = data.loc[(data["product"] == "viirs-jpss1") & (data.satellite == "N20"), ["region_id", "week_start", "detections"]]
    pairs = reference.merge(other, on=["region_id", "week_start"], suffixes=("_modis_terra", "_viirs_n20"), validate="one_to_one")
    return pairs.loc[(pairs.detections_modis_terra > 0) & (pairs.detections_viirs_n20 > 0)].copy()


def harmonize(weekly: pd.DataFrame, train_end: str, min_pairs: int = 30, steps: int = 500, seed: int = 20261110) -> dict[str, Any]:
    pairs = matched_training_weeks(weekly, train_end)
    base = {"reference": "MODIS/Terra", "comparison": "VIIRS/NOAA-20",
            "condition": "ambos os produtos registraram detecção na mesma célula e semana",
            "train_end": train_end, "matched_weeks": len(pairs),
            "interpretation": "diferença de log(1 + contagem) condicional; não é atividade real nem probabilidade de detecção"}
    if len(pairs) < min_pairs:
        return {**base, "status": "insufficient_data", "baseline": None, "posterior": None,
                "reason": f"São necessários pelo menos {min_pairs} pares; a disponibilidade e a cobertura continuam desconhecidas."}
    pairs["difference"] = np.log1p(pairs.detections_viirs_n20) - np.log1p(pairs.detections_modis_terra)
    pairs["month"] = pairs.week_start.dt.month
    group = pairs.groupby(["region_id", "month"])["difference"].agg(["median", "size"]).reset_index()
    regions = sorted(pairs.region_id.unique())
    idx = torch.tensor(pairs.region_id.map({x: i for i, x in enumerate(regions)}).to_numpy(), dtype=torch.long)
    phase = 2 * np.pi * pairs.week_start.dt.dayofyear.to_numpy() / 365.2425
    seasonal = torch.tensor(np.column_stack([np.sin(phase), np.cos(phase)]), dtype=torch.float32)
    y = torch.tensor(pairs.difference.to_numpy(), dtype=torch.float32)

    def model() -> None:
        intercept = pyro.sample("intercept", dist.Normal(0., 1.))
        scale = pyro.sample("region_scale", dist.HalfNormal(.5))
        region = pyro.sample("region", dist.Normal(torch.zeros(len(regions)), scale).to_event(1))
        season = pyro.sample("season", dist.Normal(torch.zeros(2), .5).to_event(1))
        noise = pyro.sample("noise", dist.HalfNormal(1.))
        with pyro.plate("pairs", len(y)):
            pyro.sample("observed_difference", dist.Normal(intercept + region[idx] + seasonal @ season, noise), obs=y)

    pyro.clear_param_store(); pyro.set_rng_seed(seed)
    guide = AutoNormal(model)
    svi = SVI(model, guide, Adam({"lr": .02}), loss=Trace_ELBO())
    for _ in range(steps):
        loss = svi.step()
        if not np.isfinite(loss):
            raise FloatingPointError("Ajuste probabilístico de harmonização divergiu")
    samples = Predictive(model, guide=guide, num_samples=500, return_sites=("intercept", "region", "season", "noise"))()
    posterior = {name: {"mean": np.asarray(t.detach()).mean(axis=0).reshape(-1).tolist(),
                        "q05": np.quantile(np.asarray(t.detach()), .05, axis=0).reshape(-1).tolist(),
                        "q95": np.quantile(np.asarray(t.detach()), .95, axis=0).reshape(-1).tolist()}
                 for name, t in samples.items()}
    return {**base, "status": "exploratory", "regions": regions,
            "baseline": group.to_dict(orient="records"), "posterior": posterior,
            "limitations": ["seleção em pares positivos", "cobertura desconhecida", "sensores não são verdade de referência"]}
