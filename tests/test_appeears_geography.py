"""Testes sintéticos de autenticação, escala, qualidade, tempo e exportação NASA."""

import json
import hashlib
from pathlib import Path

import pandas as pd
import pytest
import requests
from types import SimpleNamespace

from earthfire.appeears import AppEEARSClient, AppEEARSError, AuthenticationRequired
from earthfire.geography import (
    collect,
    parse_csv,
    sampling_grid,
    snapshot,
    tasks_for_points,
)
from earthfire.geography_view import export_view
from earthfire.io import read_json
from earthfire.runtime import ResourceLimit

ROOT = Path(__file__).resolve().parents[1]


class Response:
    def __init__(self, data=None, status=200):
        self.data, self.status_code = data, status
        self.headers = {}

    def json(self):
        return self.data

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def iter_content(self, size):
        yield self.data


class Session:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def test_authentication_is_redacted_and_not_cached(tmp_path):
    session = Session([Response({"token": "synthetic-secret"})])
    client = AppEEARSClient(tmp_path, token="fixture", session=session)
    client.authenticate("fixture-user", "fixture-password")
    assert client.token == "synthetic-secret"
    content = "".join(p.read_text() for p in tmp_path.rglob("*.json"))
    assert "synthetic-secret" not in content and "fixture-password" not in content


def test_idempotent_submission_does_not_duplicate_tasks(tmp_path):
    session = Session([Response({"task_id": "synthetic-task"})])
    client = AppEEARSClient(tmp_path, token="fixture", session=session)
    request = {"task_type": "point", "params": {}}
    assert client.submit(request) == client.submit(request) == "synthetic-task"
    assert len(session.calls) == 1
    with pytest.raises(ValueError):
        client.metadata("../../escape")
    with pytest.raises(ValueError):
        client.download_csv("../../escape")


def test_ambiguous_post_is_not_retried(tmp_path):
    session = Session([requests.Timeout("fixture")])
    client = AppEEARSClient(tmp_path, token="fixture", session=session)
    with pytest.raises(AppEEARSError):
        client.submit({"task_type": "point"})
    with pytest.raises(AppEEARSError, match="confirmação"):
        client.submit({"task_type": "point"})
    assert len(session.calls) == 1


def test_expired_token_and_rate_limit_are_explicit(tmp_path):
    for status, kind in [(401, AuthenticationRequired), (429, AppEEARSError)]:
        client = AppEEARSClient(
            tmp_path / str(status),
            token="fixture",
            session=Session([Response(status=status)]),
        )
        with pytest.raises(kind):
            client.task("synthetic")


def test_signed_download_does_not_forward_token_and_checks_checksum(tmp_path):
    payload = b"ID,Date\na,2020-01-01\n"
    digest = hashlib.sha256(payload).hexdigest()
    bundle = Response(
        {
            "files": [
                {
                    "file_name": "nested/real.csv",
                    "file_id": "fixture-file",
                    "file_size": len(payload),
                    "sha256": digest,
                }
            ]
        }
    )
    redirect = Response(status=302)
    redirect.headers["Location"] = (
        "https://fixture.s3.amazonaws.com/data?signature=synthetic"
    )
    session = Session([bundle, redirect, Response(payload)])
    client = AppEEARSClient(tmp_path, token="synthetic-token", session=session)
    files = client.download_csv("fixture-task")
    assert files[0].read_bytes() == payload
    assert session.calls[-1][2]["headers"] == {}
    assert "synthetic-token" not in "".join(
        p.read_text() for p in tmp_path.rglob("*.json")
    )


def test_download_outside_nasa_storage_is_rejected(tmp_path):
    bundle = Response(
        {"files": [{"file_name": "data.csv", "file_id": "fixture", "file_size": 0}]}
    )
    redirect = Response(status=302)
    redirect.headers["Location"] = "https://untrusted.invalid/data"
    client = AppEEARSClient(
        tmp_path, token="fixture", session=Session([bundle, redirect])
    )
    with pytest.raises(AppEEARSError, match="destinos"):
        client.download_csv("fixture-task")


def test_sampling_and_static_period_are_distinct():
    config = read_json(ROOT / "config/geography.json")
    points = sampling_grid(-21.229167, -54.341825)
    assert len(points) == 9 and len({p["id"] for p in points}) == 9
    tasks = tasks_for_points(points, "2020-01-01", "2025-12-31", config)
    assert len(tasks) == 2
    assert tasks[1]["params"]["dates"][0]["startDate"] == "02-11-2000"
    polar = tasks_for_points(sampling_grid(75, 20), "2020-01-01", "2025-12-31", config)
    assert len(polar) == 1


def fixture_data(tmp_path):
    points = sampling_grid(-21.229167, -54.341825)
    point = points[0]
    path = tmp_path / "synthetic.csv"
    rows = []
    for day, ndvi, qa in [
        ("2020-01-01", 0.8, 2116),
        ("2020-02-01", 0.9, 3),
        ("2020-03-01", -0.3, 0),
    ]:
        rows.append(
            {
                "ID": point["id"],
                "Latitude": point["latitude"],
                "Longitude": point["longitude"],
                "Date": day,
                "MOD13Q1_061__250m_16_days_NDVI": ndvi,
                "MOD13Q1_061__250m_16_days_VI_Quality": qa,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)
    config = {
        "products": {
            "MOD13Q1.061": {
                "layers": ["_250m_16_days_NDVI", "_250m_16_days_VI_Quality"],
                "resolution_m": 250,
                "temporal": "16day",
                "availability_lag_days_assumed": 30,
            }
        }
    }
    metadata = {
        "MOD13Q1.061": {
            "_250m_16_days_NDVI": {
                "IsQA": False,
                "ScaleFactor": 0.0001,
                "FillValue": -3000,
                "ValidMin": -0.2,
                "ValidMax": 1.0,
                "Units": "NDVI",
            },
            "_250m_16_days_VI_Quality": {"IsQA": True},
        }
    }
    data = parse_csv([path], config, metadata, retrieved_at="2026-11-12T00:00:00Z")
    return data, points, config, metadata


def test_scaled_csv_qa_bits_and_no_future_information(tmp_path):
    data, points, _, _ = fixture_data(tmp_path)
    assert data.value.iloc[0] == 0.8  # não aplicar 0,0001 novamente
    assert data.quality_usable.tolist() == [True, False, False]
    assert snapshot(data, points, "2020-02-10")["status"] == "insufficient_data"
    context = snapshot(data, points, "2020-02-20")
    assert context["features"]["geo_ndvi"] == 0.8
    assert not snapshot(data, points, "2020-02-20", operational=True)["samples"][0][
        "layers"
    ]
    changed = data.copy()
    changed.loc[changed.date >= "2020-02-01", "value"] = 999
    assert snapshot(changed, points, "2020-02-20")["features"] == context["features"]
    page = export_view(context, tmp_path / "view")
    assert page.exists() and "NDVI" in page.read_text(encoding="utf-8")
    json.dumps(context, allow_nan=False)


def test_conflicting_duplicate_exports_are_rejected(tmp_path):
    data, points, config, metadata = fixture_data(tmp_path)
    assert len(data) == 3
    second = tmp_path / "conflict.csv"
    original = pd.read_csv(tmp_path / "synthetic.csv")
    original.loc[0, "MOD13Q1_061__250m_16_days_NDVI"] = 0.1
    original.to_csv(second, index=False)
    with pytest.raises(ValueError, match="conflitantes"):
        parse_csv(
            [tmp_path / "synthetic.csv", second],
            config,
            metadata,
            retrieved_at="2026-11-12T00:00:00Z",
        )
    empty = snapshot(pd.DataFrame(), points, "2026-11-12")
    assert empty["status"] == "insufficient_data" and empty["features"] == {}


def test_annual_composite_ends_one_year_after_actual_start(tmp_path):
    point = sampling_grid(-21.0, -54.0)[0]
    path = tmp_path / "annual.csv"
    pd.DataFrame(
        [
            {
                "ID": point["id"],
                "Latitude": point["latitude"],
                "Longitude": point["longitude"],
                "Date": "2020-03-05",
                "MOD44B_061_Percent_Tree_Cover": 40,
                "MOD44B_061_Quality": 0,
                "MOD44B_061_Cloud": 0,
            }
        ]
    ).to_csv(path, index=False)
    config = read_json(ROOT / "config/geography.json")
    spec = config["products"]["MOD44B.061"]
    spec["layers"] = ["Percent_Tree_Cover", "Quality", "Cloud"]
    metadata = {
        "MOD44B.061": {
            "Percent_Tree_Cover": {
                "IsQA": False,
                "FillValue": 253,
                "ValidMin": 0,
                "ValidMax": 100,
            },
            "Quality": {"IsQA": True},
            "Cloud": {"IsQA": True},
        }
    }
    data = parse_csv(
        [path],
        {"products": {"MOD44B.061": spec}},
        metadata,
        retrieved_at="2026-11-12T00:00:00Z",
    )
    assert data.period_end_exclusive.iloc[0] == pd.Timestamp("2021-03-05")
    assert snapshot(data, [point], "2022-02-01")["status"] == "insufficient_data"


def test_resource_stop_is_structured_without_synthetic_observations(tmp_path):
    def stop(_):
        raise ResourceLimit("fixture budget exhausted")

    client = SimpleNamespace(
        cache=tmp_path, metadata=lambda _: {"value": {}}, submit=stop
    )
    cfg = {
        "products": {"fixture": {"layers": ["value"], "temporal": "16day"}},
        "max_points_per_task": 100,
        "wait_seconds": 0,
    }
    result = collect(
        client, sampling_grid(-21.0, -54.0, side=1), "2020-01-01", "2020-12-31", cfg
    )
    assert result["status"] == "resource_limit" and result["files"] == []


def test_view_network_failure_is_not_a_zero_geographic_measurement(
    tmp_path, monkeypatch
):
    from earthfire import geography_view as view

    def offline(_):
        raise AppEEARSError("fixture offline")

    monkeypatch.setattr(
        view, "AppEEARSClient", lambda *a, **kw: SimpleNamespace(metadata=offline)
    )
    value = view.describe_geography(
        -21.0, -54.0, "2026-11-12", cache_dir=tmp_path, fetch=False
    )
    assert value["status"] == "source_unavailable" and value["features"] == {}


def test_view_labels_partially_finished_collection(tmp_path, monkeypatch):
    from earthfire import geography_view as view

    data, _, _, _ = fixture_data(tmp_path)
    monkeypatch.setattr(
        view, "AppEEARSClient", lambda *a, **kw: SimpleNamespace(metadata=lambda _: {})
    )
    monkeypatch.setattr(view, "collect", lambda *a, **kw: {"status": "pending"})
    monkeypatch.setattr(view, "parse_csv", lambda *a, **kw: data)
    value = view.describe_geography(
        -21.229167, -54.341825, "2020-02-20", cache_dir=tmp_path
    )
    assert value["status"] == "partial" and not value["collection_complete"]


def test_cache_only_view_never_fetches_missing_metadata(tmp_path, monkeypatch):
    from earthfire import geography_view as view

    def unexpected_request(_):
        raise AssertionError("cache-only must not contact the API")

    monkeypatch.setattr(
        view,
        "AppEEARSClient",
        lambda *a, **kw: SimpleNamespace(metadata=unexpected_request),
    )
    result = view.describe_geography(
        -21.0, -54.0, "2026-11-12", cache_dir=tmp_path, fetch=False
    )
    assert result["status"] == "source_unavailable" and result["features"] == {}
