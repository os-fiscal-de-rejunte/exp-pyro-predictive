"""Reprocessa o exemplo real NASA offline; não mede acurácia preditiva."""

from pathlib import Path

import pytest

from earthfire.geography import parse_csv, sampling_grid, snapshot
from earthfire.io import read_json, sha256_file

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "artifacts/geography_demo"


def test_real_nasa_csv_contract_and_frozen_checksums():
    manifest = read_json(DATA / "manifest.json")
    for filename, digest in manifest["files"].items():
        assert sha256_file(DATA / filename) == digest
    config = read_json(ROOT / "config/geography.json")
    metadata = {
        product: read_json(DATA / "metadata" / ("layers_" + product + ".json"))["data"]
        for product in config["products"]
    }
    data = parse_csv(
        sorted((DATA / "source_csv").glob("*.csv")),
        config,
        metadata,
        retrieved_at="2026-11-12T12:00:00Z",
    )
    assert len(data) == 1719 and data.point_id.nunique() == 9
    assert data["product"].nunique() == 4
    context = snapshot(data, sampling_grid(-21.229167, -54.341825), "2026-11-12")
    assert context["status"] == "ok"
    assert context["features"]["geo_ndvi"] == pytest.approx(0.6603666666666667)
    assert context["features"]["geo_elevation_m"] == pytest.approx(450.44444444444446)
    assert context["features"]["geo_tree_percent"] == pytest.approx(21.555555555555557)
    assert not context["availability_verified"]
