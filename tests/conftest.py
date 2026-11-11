from __future__ import annotations

import json
from pathlib import Path

import pytest

from earthfire.runtime import Run


@pytest.fixture
def project(tmp_path: Path) -> Run:
    config = {
        "data_dir": "DATA", "runs_dir": "runs", "seed": 42,
        "operational": {"total_seconds": 3600, "reserve_seconds": 0, "training_seconds": 120,
            "max_training_runs": 3, "max_cpu_threads": 2, "memory_fraction_initial_available": .8,
            "new_files_fraction_initial_free": .2, "new_files_max_bytes": 1000000000,
            "nasa_power_requests": 5, "network_retries": 1, "network_timeout_seconds": 2,
            "checkpoint_seconds": 1},
        "regions": [{"name": "test", "latitude": 0., "longitude": 0.}],
        "split": {"train_end": "2018-12-31", "validation_end": "2019-06-30", "test_start": "2019-07-01", "test_end": "2020-12-31"},
    }
    (tmp_path / "DATA").mkdir()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return Run(config_path, "test", tmp_path)
