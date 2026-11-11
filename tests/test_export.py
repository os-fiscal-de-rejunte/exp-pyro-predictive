import json

import pandas as pd

from earthfire.export import export


def test_calendar_and_geojson_contract(project):
    processed = project.path / "processed"; processed.mkdir()
    calendar = pd.DataFrame({"region_id": ["a"], "product": ["modis"], "week_start": pd.to_datetime(["2001-01-01"]),
                             "detections": [None], "observation_state": ["missing_unknown"]})
    calendar.to_parquet(processed / "calendar.parquet", index=False)
    export(project)
    output = project.path / "exports"
    geojson = json.loads((output / "regions.geojson").read_text(encoding="utf-8"))
    exported = json.loads((output / "calendar.json").read_text(encoding="utf-8"))
    assert geojson["type"] == "FeatureCollection" and len(geojson["features"]) == 1
    assert exported["records"][0]["detections"] is None
    assert exported["records"][0]["observation_state"] == "missing_unknown"
