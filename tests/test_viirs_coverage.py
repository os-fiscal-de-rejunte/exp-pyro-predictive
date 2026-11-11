"""Testes sintéticos; não demonstram acesso ou desempenho científico real."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from earthfire import viirs_coverage
from earthfire.io import write_json
from earthfire.viirs_coverage import HDF5_MAGIC, download_granules, download_with_gaps, query_granules


def fake_run(path: Path):
    return SimpleNamespace(path=path, config={"operational": {"network_retries": 1, "network_timeout_seconds": 2}},
                           guard=lambda: None)


def test_public_discovery_saves_metadata_and_rejects_untrusted_host(tmp_path):
    title = "VJ114A1.A2025257.h13v10.002.2025258083145"
    link = f"https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/VJ114A1.002/{title}/{title}.h5"
    body = {"feed": {"entry": [{"title": title, "time_start": "2025-09-14T00:00:00Z",
                                "time_end": "2025-09-14T23:59:00Z", "links": [{"href": link}]}]}}

    class Response:
        status_code = 200
        headers = {"CMR-Hits": "2"}
        content = json.dumps(body).encode()

        def raise_for_status(self):
            pass

        def json(self):
            return body

    client = SimpleNamespace(get=lambda *args, **kwargs: Response())
    result = query_granules(fake_run(tmp_path), (-47.2, -11.8, -46.5, -11.2), "2025-09-14", "2025-09-15", 1, client)
    assert result["truncated"] and result["returned"] == 1
    assert Path(result["cmr_response"]).exists()
    body["feed"]["entry"][0]["links"][0]["href"] = "https://example.com/unsafe.h5"
    with pytest.raises(ValueError, match="não reconhecido"):
        query_granules(fake_run(tmp_path), (-47.2, -11.8, -46.5, -11.2), "2025-09-14", "2025-09-15", 1, client)


def test_download_requires_token_and_never_writes_it_to_manifest(tmp_path):
    title = "VJ114A1.A2025257.h13v10.002.2025258083145"
    url = f"https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/VJ114A1.002/{title}/{title}.h5"
    discovery = {"product": "VJ114A1.002", "granules": [{"filename": title + ".h5", "url": url}]}
    run = fake_run(tmp_path)
    assert download_granules(run, discovery, None)["status"] == "authentication_required"
    data = HDF5_MAGIC + b"synthetic-test-content"

    class Response:
        status_code = 200
        headers = {"Content-Length": str(len(data))}

        def __enter__(self): return self
        def __exit__(self, *args): return False
        def raise_for_status(self): pass
        def iter_content(self, chunk_size): return iter([data])

    def get(_url, **kwargs):
        assert kwargs["headers"]["Authorization"] == "Bearer token-para-teste"
        return Response()

    result = download_granules(run, discovery, "token-para-teste", SimpleNamespace(get=get))
    assert result["status"] == "ok" and result["downloaded"] == 1
    assert "token-para-teste" not in Path(result["manifest"]).read_text(encoding="utf-8")


def test_unavailable_granule_is_recorded_and_not_fabricated(tmp_path, monkeypatch):
    folder = tmp_path / "viirs_coverage"
    folder.mkdir()
    manifest = folder / "manifest.json"
    write_json(manifest, {"files": {}, "requests": 0})
    calls = []

    def fake_download(run, discovery, token, session=None):
        names = [x["filename"] for x in discovery["granules"]]
        calls.append(names)
        if "bad.h5" in names:
            write_json(manifest, {"files": {"good.h5": {"sha256": "fake"}}, "requests": 3})
            return {"status": "data_unavailable", "failed_file": "bad.h5", "reason": "HTTP 404"}
        write_json(manifest, {"files": {"good.h5": {"sha256": "fake"}, "next.h5": {"sha256": "fake"}}, "requests": 4})
        return {"status": "ok"}

    monkeypatch.setattr(viirs_coverage, "download_granules", fake_download)
    discovery = {"product": "VJ114A1.002", "granules": [{"filename": x} for x in ("good.h5", "bad.h5", "next.h5")]}
    result = download_with_gaps(fake_run(tmp_path), discovery, "test-token")
    assert result["status"] == "partial" and result["unavailable_files"] == ["bad.h5"]
    assert len(calls) == 2 and "bad.h5" not in calls[1]
    assert "bad.h5" in json.loads((folder / "unavailable_granules.json").read_text(encoding="utf8"))["files"]


def test_collection_stops_after_three_consecutive_unavailable_files(tmp_path, monkeypatch):
    folder = tmp_path / "viirs_coverage"
    folder.mkdir()

    def failing_download(run, discovery, token, session=None):
        return {"status": "data_unavailable", "failed_file": discovery["granules"][0]["filename"], "reason": "HTTP 503"}

    monkeypatch.setattr(viirs_coverage, "download_granules", failing_download)
    discovery = {"product": "VJ114A1.002", "granules": [{"filename": f"bad{i}.h5"} for i in range(5)]}
    with pytest.raises(ConnectionError, match="Três grânulos consecutivos"):
        download_with_gaps(fake_run(tmp_path), discovery, "test-token")
    failures = json.loads((folder / "unavailable_granules.json").read_text(encoding="utf8"))["files"]
    assert len(failures) == 3
