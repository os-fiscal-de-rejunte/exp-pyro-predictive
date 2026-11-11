import pandas as pd

from earthfire.recent_firms import add_weather_context, lookup, normalize


def test_recent_modis_satellite_codes_and_unknown_location(project):
    raw = pd.DataFrame([{"latitude": "-11.5", "longitude": "-46.875", "acq_date": "2026-11-08",
                         "acq_time": "930", "satellite": "T", "instrument": "MODIS",
                         "confidence": "80", "daynight": "D", "version": "6.1NRT"}])
    normalized, report = normalize(raw, "modis")
    assert report["retained"] == 1
    assert normalized.iloc[0].satellite == "Terra" and normalized.iloc[0].satellite_raw == "T"
    output = project.path / "recent"
    output.mkdir()
    normalized.to_parquet(output / "observations.parquet", index=False)
    found = lookup(project, -11.5, -46.875)
    assert found["status"] == "location_recognized"
    assert found["products"]["modis/Terra"]["records"] == 1
    missing = lookup(project, -12., -47.)
    assert missing["products"]["modis/Terra"]["state"] == "missing_unknown"
    assert missing["products"]["modis/Terra"]["records"] is None


def test_recent_viirs_confidence_categories_are_product_specific():
    raw = pd.DataFrame([{"latitude": "-11.5", "longitude": "-46.875", "acq_date": "2026-11-08",
                         "acq_time": "930", "satellite": "N20", "instrument": "VIIRS",
                         "confidence": value, "daynight": "D", "version": "2.0NRT"}
                        for value in ("low", "nominal", "high", "l", "n", "h", "invalid")])
    normalized, report = normalize(raw, "viirs-jpss1")
    assert report["confidence_invalid"] == 1
    assert normalized.confidence.tolist() == ["low", "nominal", "high", "l", "n", "h", "invalid"]


def test_weather_context_uses_eight_historical_days(project, monkeypatch):
    days = pd.date_range("2024-01-01", periods=8).strftime("%Y%m%d")
    values = {name: dict.fromkeys(days, value) for name,value in
              (("T2M", 25), ("RH2M", 70), ("PRECTOTCORR", 2), ("WS10M", 3))}
    units = {"T2M": "C", "RH2M": "%", "PRECTOTCORR": "mm/day", "WS10M": "m/s"}

    class Client:
        def __init__(self, *args, **kwargs):
            pass
        def fetch(self, latitude, longitude, start, end, parameters):
            return {"request": {"start": "20240101", "end": "20240108", "parameters": ",".join(parameters)},
                    "retrieved_utc": "2024-01-15T00:00:00Z", "response": {
                        "properties": {"parameter": values}, "header": {"fill_value": -999},
                        "parameters": {k: {"units": v} for k,v in units.items()}}}

    monkeypatch.setattr("earthfire.recent_firms.NasaPowerClient", Client)
    result = add_weather_context(project, {}, -11.5, -46.875, "2024-01-08")
    assert result["weather_history"]["precipitation_sum_mm"] == 16
    assert result["weather_history"]["temperature_mean_c"] == 25
