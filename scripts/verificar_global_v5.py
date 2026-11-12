"""Verifica os artefatos reais da expansão, sem treinar ou consultar a rede."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from earthfire.global_pipeline import predict
from earthfire.io import read_json, sha256_file, utc_now, write_json


def verify(root: Path, check_originals: bool = True) -> dict:
    manifest = read_json(root / "processed/manifest.json")
    dataset = root / "processed/mask_examples.parquet"
    assert sha256_file(dataset) == manifest["examples_sha256"]
    frame = pd.read_parquet(dataset)
    assert not frame.duplicated(["region_id", "period_start"]).any()
    assert (frame.weather_last_date < frame.period_start).all()
    available = frame.history_last_produced_at.notna()
    assert (
        frame.loc[available, "history_last_produced_at"]
        < frame.loc[available, "period_start"]
    ).all()
    assert frame.loc[frame.event.notna(), "fully_observed"].all()
    assert (
        not frame.loc[frame.event.isna(), "observation_state"]
        .isin(["observed_no_detection", "positive_detection"])
        .any()
    )
    assert (
        frame.loc[frame.split == "train", "period_end_exclusive"]
        <= pd.Timestamp("2023-01-01")
    ).all()
    assert (
        frame.loc[frame.split == "validation", "period_end_exclusive"]
        <= pd.Timestamp("2024-01-01")
    ).all()
    assert (
        frame.loc[frame.split == "test", "period_start"] >= pd.Timestamp("2024-01-01")
    ).all()
    registry = read_json(root / "model_registry.json")
    for name, digest in registry["artifacts"].items():
        assert sha256_file(root / name) == digest, name
    predictions = pd.read_parquet(root / "evaluation/predictions.parquet")
    for name in [
        "global_frequency",
        "seasonal",
        "persistence",
        "pyro_weather",
        "pyro_no_weather",
        "logistic_weather",
        "logistic_no_weather",
        "routed_probability",
    ]:
        assert (
            np.isfinite(predictions[name]).all()
            and predictions[name].between(0, 1).all()
        ), name
    eligible = predictions.loc[predictions.release_eligible]
    example = None
    if not eligible.empty:
        selected = eligible.iloc[0]
        location = frame.loc[frame.region_id == selected.region_id].iloc[0]
        example = predict(
            float(location.latitude),
            float(location.longitude),
            str(selected.period_start.date()),
            root,
        )
        assert example["status"] == "ok", example
        assert 0 <= example["probability"] <= 1
        assert abs(example["probability"] - selected.routed_probability) < 0.05
        write_json(root / "evaluation/example_prediction.json", example)
    unsupported = predict(0.0, -140.0, "2024-01-01", root)
    assert (
        unsupported["status"] == "insufficient_data"
        and unsupported["probability"] is None
    )
    checked = 0
    if check_originals:
        inventory = read_json(root / "catalog/local_inventory.json")
        for item in inventory["files"]:
            assert sha256_file(Path("DATA") / item["name"]) == item["sha256"], item[
                "name"
            ]
            checked += 1
    assert (root / "evaluation/comparison.png").read_bytes().startswith(b"\x89PNG")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = read_json(root / "evaluation/metrics.json")
    bins = metrics["calibration"]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Calibração ideal")
    ax.scatter(
        [b["mean_probability"] for b in bins],
        [b["observed_rate"] for b in bins],
        s=[max(20, b["n"]) for b in bins],
        alpha=0.7,
    )
    ax.set(
        xlabel="Probabilidade média",
        ylabel="Frequência observada",
        xlim=(0, 1),
        ylim=(0, 1),
        title="Teste: campeão global; tamanho indica amostras",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "evaluation/calibration.png")
    plt.close(fig)
    result = {
        "status": "passed",
        "checked_utc": utc_now(),
        "originals_rehashed": checked,
        "area_periods": len(frame),
        "test_predictions": len(predictions),
        "eligible_test_predictions": len(eligible),
        "example_status": example["status"] if example else "insufficient_data",
    }
    write_json(root / "verification.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir", type=Path, default=Path("runs/global_v5b_20261111")
    )
    parser.add_argument("--skip-originals", action="store_true")
    args = parser.parse_args()
    print(verify(args.run_dir, not args.skip_originals))
