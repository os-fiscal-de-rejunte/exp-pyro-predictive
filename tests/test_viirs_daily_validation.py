"""Teste sintético da seleção do complemento de validação."""
import json
from datetime import date
from types import SimpleNamespace

from earthfire import viirs_daily_validation
from earthfire.io import sha256_file, write_json


def test_extra_validation_dates_are_fixed_and_missing_is_recorded(tmp_path, monkeypatch):
    base = tmp_path / "viirs_coverage"
    base.mkdir()
    entries = []
    for month in viirs_daily_validation.MONTHS:
        for day in (1, 8, 15, 22):
            d = date(2023, month, day)
            title = f"VJ114A1.A{d.year}{d.timetuple().tm_yday:03d}.h13v10.002.2024001000000"
            url = f"https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/VJ114A1.002/{title}/{title}.h5"
            entries.append({"title": title, "time_start": d.isoformat() + "T00:00:00Z", "links": [{"href": url}]})
    raw_tmp = base / "raw.json"
    write_json(raw_tmp, {"feed": {"entry": entries}})
    digest = sha256_file(raw_tmp)
    raw_tmp.rename(base / f"cmr_{digest}.json")
    write_json(base / "daily_progress.json", {"config_sha256": "cfg", "partitions": {
        "site/2023": {"tile": "h13v10", "cmr_sha256": digest}}})
    run = SimpleNamespace(path=tmp_path, config_hash="cfg", config={"regions": [{"name": "site"}],
                          "viirs_daily": {"days_of_month": [1, 8, 15, 22]}},
                          guard=lambda: None, stage=lambda *a, **k: None)

    def fake_download(run, discovery, token, session=None):
        names = [x["filename"] for x in discovery["granules"]]
        return {"status": "partial", "unavailable_files": [names[0]]}

    monkeypatch.setattr(viirs_daily_validation, "download_with_gaps", fake_download)
    result = viirs_daily_validation.collect(run, "synthetic-token")
    data = json.loads((base / "extra_validation_progress.json").read_text(encoding="utf8"))
    partition = data["partitions"]["site/2023"]
    assert result["files"] == 15 and len(partition["missing_days"]) == 1
    assert all(int(d[5:7]) in viirs_daily_validation.MONTHS for d in partition["expected_days"])
