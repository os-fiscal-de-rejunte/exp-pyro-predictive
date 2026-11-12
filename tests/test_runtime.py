import time
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor

import pytest

from earthfire.io import read_json, write_json
from earthfire.runtime import ResourceLimit


def test_deadline_and_resume_configuration_guard(project):
    budget = read_json(project.budget_path)
    budget["started_epoch"] = time.time() - 7200
    write_json(project.budget_path, budget)
    with pytest.raises(ResourceLimit, match="Reserva"):
        project.guard()


def test_external_disk_cleanup_does_not_reset_run_storage_limit(project, monkeypatch):
    budget = read_json(project.budget_path)
    budget["new_files_limit_bytes"] = 4096
    write_json(project.budget_path, budget)
    (project.path / "synthetic_payload.bin").write_bytes(b"x" * 8192)
    monkeypatch.setattr(
        "earthfire.runtime.shutil.disk_usage",
        lambda _: SimpleNamespace(free=budget["initial_free_disk"] + 10**9),
    )
    with pytest.raises(ResourceLimit, match="arquivos"):
        project.guard()


def test_parallel_collectors_share_atomic_resource_snapshot(project):
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda _: project.guard(), range(4)))
    snapshot = read_json(project.path / "resources.json")
    assert snapshot["tracked_run_bytes"] > 0
    assert (
        snapshot["peak_sampled_rss_bytes"] >= snapshot["registered_process_rss_bytes"]
    )
