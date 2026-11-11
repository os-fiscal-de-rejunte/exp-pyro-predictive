import numpy as np
import pandas as pd
import torch

from earthfire.dual_features import BASE, FUTURE, PAST, ROLLING, SENSOR
from earthfire.dual_model import posterior_counts, train


def test_dual_model_checkpoint_positive_predictions(project):
    project.config["spatial_holdout_regions"] = []
    columns = BASE + SENSOR + PAST + ROLLING + FUTURE
    rng = np.random.default_rng(12)
    n = 130
    frame = pd.DataFrame({name: rng.normal(3, 1, n) for name in columns})
    frame["region_id"] = "a"; frame["split"] = "train"; frame["spatial_group"] = "train_regions"
    frame["detections"] = rng.integers(1, 20, n)
    (project.path / "processed_v2").mkdir()
    frame.to_parquet(project.path / "processed_v2/examples.parquet", index=False)
    target = train(project, frame, include_weather=True, steps=5)
    draws, logp = posterior_counts(target, frame.head(3), n_samples=20, seed=1)
    assert draws.shape == (20, 3) and (draws > 0).all() and np.isfinite(logp).all()
    assert torch.load(target, weights_only=False)["n_train"] == n
    assert train(project, frame, include_weather=True, steps=5) == target
