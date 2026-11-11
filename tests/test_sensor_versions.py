import pandas as pd

from earthfire.sensor_versions import annotate_versions, build_version_index


def test_new_sensor_version_flagged_without_looking_at_test_labels():
    records = []
    for week, modis in [("2021-12-06", "6.03"), ("2024-01-08", "61.03")]:
        for product, satellite, version in [("modis", "Terra", modis), ("viirs-jpss1", "N20", "2")]:
            records.append({"region_id": "a", "acquired_utc": pd.Timestamp(week, tz="UTC"),
                            "product": product, "satellite": satellite, "version": version})
    index = build_version_index(pd.DataFrame(records))
    train = pd.DataFrame([{"region_id": "a", "week_start": pd.Timestamp("2021-12-13")}])
    test = pd.DataFrame([{"region_id": "a", "week_start": pd.Timestamp("2024-01-15")}])
    result, known = annotate_versions(test, train, index)
    assert known == {"modis_terra_versions": ["6.03"], "viirs_n20_versions": ["2"]}
    assert result.iloc[0].sensor_version_support == "unseen_version"
