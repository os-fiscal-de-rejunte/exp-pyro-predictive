from pathlib import Path

import pytest

from earthfire.power_zarr import CachedZarrStore


class Response:
    status_code = 200
    headers = {"Content-Length": "3", "ETag": "unit-test"}
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def raise_for_status(self): pass
    def iter_content(self, chunk_size): yield b"abc"


class Session:
    def __init__(self): self.calls = 0
    def get(self, *args, **kwargs): self.calls += 1; return Response()


def test_zarr_store_cache_budget_and_integrity(project):
    session = Session()
    store = CachedZarrStore(project, project.path / "zarr-test", session=session, min_interval=0)
    assert store["T2M/0.0.0"] == b"abc" and store["T2M/0.0.0"] == b"abc"
    assert session.calls == 1 and store.manifest["requests"] == 1
    with pytest.raises(KeyError): _ = store["../outside"]
    (project.path / "zarr-test/objects/T2M/0.0.0").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="incompatível"): _ = store["T2M/0.0.0"]
