import numpy as np
import pandas as pd
import json
import sys
from types import SimpleNamespace

from earthfire import geos_forecast
from earthfire.geos_forecast import daily_weather, parse_ascii, parse_grads_start


def test_geos_ascii_and_time():
    body = """t2m, [2][1][1]
[0][0], 300.0

[1][0], 301.0

time, [2]
739931.0208333, 739931.0625
lat, [1]
-11.5
lon, [1]
-46.875
"""
    values, times, lat, lon = parse_ascii(body, "t2m")
    assert values.tolist() == [300., 301.] and len(times) == 2
    assert (lat, lon) == (-11.5, -46.875)
    assert str(parse_grads_start('String grads_min "00:30z10nov2026";')) == "2026-11-10 00:30:00+00:00"


def test_geos_daily_units_and_incomplete_days():
    dates = pd.date_range("2026-11-10T00:30:00Z", periods=26, freq="h")
    hourly = pd.DataFrame({"time": dates, "t2m": 300., "qv2m": .01, "ps": 100000.,
                           "u10m": 3., "v10m": 4., "prectot": 1e-5})
    daily = daily_weather(hourly)
    assert len(daily) == 1 and daily.iloc[0].date == "2026-11-10"
    assert np.isclose(daily.iloc[0].T2M, 26.85)
    assert np.isclose(daily.iloc[0].WS10M, 5.)
    assert np.isclose(daily.iloc[0].precip_geos_mm, .864)
    assert 0 < daily.iloc[0].RH2M_derived < 100


def test_geos_network_failure_is_explicit(tmp_path, monkeypatch):
    run = SimpleNamespace(path=tmp_path, stage=lambda *args, **kwargs: None)
    monkeypatch.setattr(geos_forecast, "Run", lambda *args: run)

    class OfflineClient:
        def __init__(self, _run):
            pass

        def fetch(self, *args):
            raise ConnectionError("HTTP 502")

    monkeypatch.setattr(geos_forecast, "GeosClient", OfflineClient)
    monkeypatch.setattr(sys, "argv", ["geos_forecast", "--cycle", "20261110_00"])
    geos_forecast.main()
    result = json.loads((tmp_path / "geos_forecast/forecast_20261110_00.json").read_text(encoding="utf-8"))
    assert result["status"] == "weather_unavailable" and result["daily"] is None
