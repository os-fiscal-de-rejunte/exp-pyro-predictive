import numpy as np
import pandas as pd

from earthfire.mask_model import columns, posterior_probabilities, train


def test_mask_model_roundtrip_and_train_only_imputation(project):
    rng = np.random.default_rng(42)
    n = 160
    frame = pd.DataFrame({key: rng.normal(size=n) for key in columns(True)})
    frame["past_modis_missing"] = 0
    frame["past_viirs_missing"] = 0
    frame.loc[:10, "past_viirs_log_records"] = np.nan
    frame["event"] = np.arange(n) % 2
    frame["split"] = "train"
    frame["spatial_group"] = "train"
    frame["region_id"] = "synthetic"
    folder = project.path / "processed"
    folder.mkdir(parents=True)
    frame.to_parquet(folder / "mask_examples.parquet", index=False)
    checkpoint = train(project, frame, weather=True, steps=3)
    probabilities = posterior_probabilities(checkpoint, frame.iloc[:4], n_samples=8, seed=10)
    assert probabilities.shape == (8, 4)
    assert np.isfinite(probabilities).all() and ((probabilities > 0) & (probabilities < 1)).all()
    assert train(project, frame, weather=True, steps=3) == checkpoint
