"""Respostas sintéticas para testar cache e falhas, sem acesso externo."""

import json
from datetime import datetime

import pytest

from earthfire.global_collect import collect
from earthfire.io import read_json
from earthfire.runtime import ResourceLimit


def configure(project):
    project.config["mask"] = {
        "start_year": 2022,
        "end_year": 2022,
        "radius_km": 20,
        "max_requests": 10,
    }
    project.config["regions"][0]["group"] = "train"
    project.config["operational"]["network_concurrency"] = 2


class Response:
    status_code = 200

    def __init__(self, params):
        day = datetime.strptime(params["startDate"], "A%Y%j").date().isoformat()
        self.body = {
            "band": "FireMask",
            "units": "class-flag",
            "nrows": 2,
            "ncols": 2,
            "xllcorner": 0,
            "yllcorner": 0,
            "cellsize": 1000,
            "subset": [
                {
                    "calendar_date": day,
                    "modis_date": params["startDate"],
                    "data": [5, 5, 5, 7],
                    "tile": "h00v00",
                    "proc_date": "2022010000000",
                }
            ],
        }
        self.content = json.dumps(self.body).encode()

    def raise_for_status(self):
        pass

    def json(self):
        return self.body


def test_collect_cache_is_verified_and_resumed(project, monkeypatch):
    configure(project)

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get(self, url, params, **kwargs):
            return Response(params)

    monkeypatch.setattr("earthfire.global_collect.requests.Session", Session)
    monkeypatch.setattr("earthfire.global_collect.time.sleep", lambda _: None)
    result = collect(project)
    assert result["rows"] == 5
    before = read_json(project.path / "coverage/manifest.json")["requests"]
    assert collect(project) == result
    assert read_json(project.path / "coverage/manifest.json")["requests"] == before
    cache = next((project.path / "coverage/cache").glob("*.json"))
    cache.write_text("{}")
    with pytest.raises(ValueError, match="Cache MODIS"):
        collect(project)


def test_collect_stops_after_rate_limit(project, monkeypatch):
    configure(project)

    class Limited:
        status_code = 429

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get(self, *args, **kwargs):
            return Limited()

    monkeypatch.setattr("earthfire.global_collect.requests.Session", Session)
    monkeypatch.setattr("earthfire.global_collect.time.sleep", lambda _: None)
    with pytest.raises(ResourceLimit, match="429"):
        collect(project)
    assert read_json(project.path / "coverage/manifest.json")["requests"] <= 2
