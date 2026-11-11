"""Teste sintético: datas de arquivo ausente não se tornam dias sem foco."""
from types import SimpleNamespace

import pandas as pd

from earthfire import viirs_recent_audit
from earthfire.io import sha256_file, write_json


def test_recent_audit_distinguishes_missing_file_from_unknown_mask(tmp_path, monkeypatch):
    run_path = tmp_path / "runs/recent"
    source = run_path / "viirs_coverage"
    source.mkdir(parents=True)
    ref = tmp_path / "runs/nasa_complement_v3b_20261111/coverage/mask_summary.parquet"
    ref.parent.mkdir(parents=True)
    pd.DataFrame([{"region_name": "test", "region_id": "grid-test"}]).to_parquet(ref, index=False)
    files = {}
    for day in ("2026-10-01", "2026-10-03"):
        name = day + ".h5"
        path = source / name
        path.write_bytes(b"synthetic")
        files[name] = {"start_utc": day + "T00:00:00Z", "sha256": sha256_file(path)}
    write_json(source / "manifest.json", {"files": files})
    run = SimpleNamespace(path=run_path, root=tmp_path, guard=lambda: None, stage=lambda *a, **k: None)

    def fake_mask(path, root, geometry, day):
        valid = day == "2026-10-01"
        return {"fully_observed": valid, "event": 1 if valid else None}

    monkeypatch.setattr(viirs_recent_audit, "read_mask", fake_mask)
    result = viirs_recent_audit.audit(run, "test", "2026-10-01", "2026-10-03")
    assert result["missing_granule_days"] == ["2026-10-02"]
    assert result["unknown_coverage_days"] == 1 and result["days_with_fire_when_fully_classified"] == 1
