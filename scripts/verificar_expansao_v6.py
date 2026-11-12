"""Verificação offline dos dados reais e das consultas da expansão v6."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from earthfire.expansion_grid import point_cell
from earthfire.expansion_predict import forecast, load, predict
from earthfire.io import read_json, sha256_file, utc_now, write_json


def verify(
    root: Path, check_originals: bool = True, output_dir: Path | None = None
) -> dict:
    output = output_dir or (
        ROOT / "runs/verification_maxcoverage_v6"
        if root.resolve().is_relative_to((ROOT / "artifacts").resolve())
        else root
    )
    registry, catalog, _ = load(root)
    if (root / "package_manifest.json").exists():
        for name, digest in read_json(root / "package_manifest.json")["files"].items():
            assert sha256_file(root / name) == digest, name
    metrics = read_json(root / "evaluation/metrics.json")
    sample_meta = read_json(root / "training_sample.json")
    sample_path = root / "processed/mask_examples.parquet"
    assert sha256_file(sample_path) == sample_meta["sample_sha256"]
    sample = pd.read_parquet(sample_path)
    assert len(sample) <= 12000
    assert (sample.split == "train").all()
    assert (sample.spatial_group == "train").all()
    assert not sample.region_id.isin(sample_meta["buffer_cells"]).any()
    assert sample.weather_complete.all()
    seen: set[str] = set()
    totals = {"rows": 0, "unknown": 0, "weather_complete": 0}
    for name, digest in registry["features"].items():
        path = root / name
        assert sha256_file(path) == digest, name
        frame = pd.read_parquet(path)
        assert not frame.duplicated(["region_id", "period_start"]).any()
        ids = set(frame.region_id)
        assert not ids.intersection(seen), "Célula reaparece em outra partição"
        seen.update(ids)
        assert (frame.weather_last_date < frame.period_start).all()
        known = frame.history_last_produced_at.notna()
        assert (
            frame.loc[known, "history_last_produced_at"]
            < frame.loc[known, "period_start"]
        ).all()
        assert frame.loc[frame.event.notna(), "fully_observed"].all()
        assert (
            not frame.loc[frame.event.isna(), "observation_state"]
            .isin(["positive_detection", "observed_no_detection"])
            .any()
        )
        assert (
            frame.loc[frame.split == "train", "period_end_exclusive"]
            <= pd.Timestamp("2023-01-01")
        ).all()
        assert (
            frame.loc[frame.split == "validation", "period_start"]
            >= pd.Timestamp("2023-01-01")
        ).all()
        assert (
            frame.loc[frame.split == "validation", "period_end_exclusive"]
            <= pd.Timestamp("2024-01-01")
        ).all()
        assert (
            frame.loc[frame.split == "test", "period_start"]
            >= pd.Timestamp("2024-01-01")
        ).all()
        assert (
            frame.loc[frame.split == "test", "period_end_exclusive"]
            <= pd.Timestamp("2026-01-01")
        ).all()
        totals["rows"] += len(frame)
        totals["unknown"] += int(frame.event.isna().sum())
        totals["weather_complete"] += int(frame.weather_complete.sum())
    for rid, info in catalog.items():
        assert point_cell(info["latitude"], info["longitude"])["region_id"] == rid
        assert info["nrows"] == info["ncols"] == 41
    coverage = read_json(root / "coverage.geojson")
    covered_ids = [f["properties"]["cell_id"] for f in coverage["features"]]
    assert len(covered_ids) == len(set(covered_ids))
    released = sum(v["release"].get("eligible", False) for v in catalog.values())
    assert released == metrics["released_cells"]
    assert np.isclose(
        metrics["released_area_km2"],
        sum(v["area_km2"] for v in catalog.values() if v["release"].get("eligible")),
    )
    n_test, n_eligible = 0, 0
    replay = None
    for path in sorted((root / "evaluation/test_parts").glob("*.parquet")):
        frame = pd.read_parquet(path)
        assert frame.event.isin([0, 1]).all()
        assert (frame.period_start >= pd.Timestamp("2024-01-01")).all()
        for column in (
            "global_frequency",
            "seasonal",
            "regional_seasonal",
            "persistence",
            "routed_probability",
        ):
            available = frame[column].dropna()
            assert np.isfinite(available).all() and available.between(0, 1).all()
        eligible = frame.loc[frame.release_eligible]
        assert eligible.routed_probability.notna().all()
        n_test += len(frame)
        n_eligible += len(eligible)
        if replay is None and len(eligible):
            row = eligible.sort_values(["region_id", "period_start"]).iloc[0]
            info = catalog[row.region_id]
            replay = predict(
                info["latitude"], info["longitude"], str(row.period_start.date()), root
            )
            assert replay["status"] == "ok"
            # A posterior variacional usa outra amostra Monte Carlo na API pública.
            assert abs(replay["probability"] - row.routed_probability) < 0.06
            write_json(output / "evaluation/example_prediction.json", replay)
    assert n_test == metrics["test_n"]
    current = []
    example_future = None
    issued = pd.Timestamp.now(tz="UTC")
    day = issued.date()
    current_start = day.replace(month=1, day=1) + timedelta(
        days=((day.timetuple().tm_yday - 1) // 8) * 8
    )
    next_start = min(
        current_start + timedelta(days=8),
        day.replace(year=day.year + 1, month=1, day=1),
    )
    for rid, info in sorted(catalog.items()):
        if not info["release"].get(
            "static_projection_model"
        ) or next_start.month not in info["release"].get(
            "static_projection_months", []
        ):
            continue
        current.append(rid)
        if example_future is None:
            value = forecast(
                info["latitude"], info["longitude"], root, issued.isoformat()
            )
            json.dumps(value, allow_nan=False)
            assert value["status"] == "ok"
            assert 1 <= value["horizon_days"] <= 8
            assert pd.Timestamp(value["valid_from"]) > pd.Timestamp(value["issued_at"])
            lo, hi = value["probability_interval_95"]
            assert 0 <= lo <= hi <= 1 and 0 <= value["probability"] <= 1
            example_future = value
    if example_future:
        write_json(output / "evaluation/example_forecast.json", example_future)
    unsupported = predict(0.0, -140.0, "2024-01-01", root)
    assert (
        unsupported["status"] == "insufficient_data"
        and unsupported["probability"] is None
    )
    originals = 0
    if check_originals:
        inventory = read_json(ROOT / "artifacts/global_v5/catalog/local_inventory.json")
        for item in inventory["files"]:
            assert sha256_file(ROOT / "DATA" / item["name"]) == item["sha256"], item[
                "name"
            ]
            originals += 1
    result = {
        "status": "passed",
        "checked_utc": utc_now(),
        "originals_rehashed": originals,
        "feature_cells": len(seen),
        **totals,
        "test_rows": n_test,
        "released_test_rows": n_eligible,
        "released_cells": released,
        "currently_projectable_cells": len(current),
        "currently_projectable_cell_ids": current,
        "currently_projectable_area_km2": sum(catalog[r]["area_km2"] for r in current),
        "historical_example_status": replay["status"]
        if replay
        else "insufficient_data",
        "future_example_status": example_future["status"]
        if example_future
        else "insufficient_data",
    }
    write_json(output / "verification.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir", type=Path, default=ROOT / "runs/maxcoverage_v6b_20261111"
    )
    parser.add_argument("--skip-originals", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    result = verify(args.run_dir, not args.skip_originals, args.output_dir)
    print({k: v for k, v in result.items() if not k.endswith("_ids")})
