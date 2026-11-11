import json

import pytest
import requests

from earthfire.firms_api import fetch_area, validate_query


def test_firms_api_limits_days_and_redacts_key(project, monkeypatch):
    with pytest.raises(ValueError):
        validate_query("MODIS_SP", (-50., -20., -49., -19.), "2024-01-01", 6)
    monkeypatch.setenv("FIRMS_MAP_KEY", "secretToken123")

    class FailedSession:
        def get(self, url, **kwargs):
            raise requests.ConnectionError("connection failed to " + url)

    result = fetch_area(project, "MODIS_SP", (-50., -20., -49., -19.), "2024-01-01", 1, FailedSession())
    assert result["status"] == "unavailable"
    assert "secretToken123" not in json.dumps(result)
    assert "secretToken123" not in (project.path / "firms_api/manifest.json").read_text(encoding="utf-8")


def test_firms_api_saves_valid_csv_without_key(project, monkeypatch):
    monkeypatch.setenv("FIRMS_MAP_KEY", "secretToken123")
    content = b"latitude,longitude,acq_date,acq_time,satellite\n-20,-49,2024-01-01,1200,Terra\n"

    class Response:
        status_code = 200
        headers = {"Content-Length": str(len(content))}
        def iter_content(self, chunk_size):
            yield content
        def close(self):
            pass

    class Session:
        def get(self, url, **kwargs):
            return Response()

    result = fetch_area(project, "MODIS_SP", (-50., -21., -48., -19.), "2024-01-01", 1, Session())
    assert result["status"] == "ok"
    assert "secretToken123" not in json.dumps(result)
    assert "secretToken123" not in (project.path / "firms_api/manifest.json").read_text(encoding="utf-8")
