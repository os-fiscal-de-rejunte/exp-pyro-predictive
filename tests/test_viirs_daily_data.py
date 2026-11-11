"""Dados sintéticos para as regras espaciais e temporais da máscara diária."""
from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset

from earthfire.viirs_daily_data import add_previous_daily_mask, history_features, read_mask


def make_granule(path: Path, classes: np.ndarray, x_start: int = -1000) -> None:
    with Dataset(str(path), "w", format="NETCDF4") as ds:
        ds.ShortName = "VJ114A1"
        ds.VersionID = "002"
        ds.PlatformShortName = "JPSS-1"
        ds.RangeBeginningDate = "2021-01-15"
        ds.ProductionTime = "2021-01-16 10:00:00"
        grid = ds.createGroup("HDFEOS").createGroup("GRIDS").createGroup("VIIRS_Grid_Daily_Fire")
        grid.createDimension("x", 4)
        grid.createDimension("y", 4)
        grid.createVariable("XDim", "f8", ("x",))[:] = [x_start + 100 * i for i in range(4)]
        grid.createVariable("YDim", "f8", ("y",))[:] = [2000, 1900, 1800, 1700]
        fields = grid.createGroup("Data Fields")
        projection = fields.createVariable("Projection", "i4")
        projection.grid_mapping_name = "sinusoidal"
        projection.earth_radius = 6371007.181
        fields.createVariable("FireMask", "u1", ("y", "x"))[:] = classes


def test_viirs_mask_preserves_cloud_and_unknown(tmp_path):
    path = tmp_path / "test.h5"
    classes = np.full((4, 4), 5, dtype=np.uint8)
    classes[1:3, 1:3] = [[5, 8], [4, 0]]
    make_granule(path, classes)
    geom = {"xllcorner": -900, "yllcorner": 1700, "nrows": 2, "ncols": 2, "cellsize_m": 100}
    result = read_mask(path, tmp_path, geom, "2021-01-15")
    assert result["fire_pixels"] == 1 and result["cloud_pixels"] == 1
    assert result["unknown_pixels"] == 1 and result["event"] is None
    classes[2, 1:3] = [5, 3]
    make_granule(path, classes)
    result = read_mask(path, tmp_path, geom, "2021-01-15")
    assert result["event"] == 1 and result["valid_pixels"] == 4
    classes[2, 2] = 10
    make_granule(path, classes)
    result = read_mask(path, tmp_path, geom, "2021-01-15")
    assert result["event"] is None and result["invalid_pixels"] == 1


def test_features_use_past_only_and_keep_missing_unknown():
    target = pd.Timestamp("2021-01-20")
    dates = pd.DatetimeIndex(["2021-01-10", "2021-01-11", "2021-01-18", "2021-01-19", "2021-01-20"])
    modis = pd.DataFrame({"period_end_exclusive": pd.to_datetime(["2021-01-15", "2021-01-19"]),
                          "produced_at": pd.to_datetime(["2021-01-16", "2021-01-21"]), "event": [0, 1]})
    result = history_features(target, dates, modis)
    assert result["past_viirs_records"] == 2
    assert result["viirs_history_end"] == pd.Timestamp("2021-01-18")
    assert result["past_modis_event"] == 0 and result["past_modis_produced_at"] < target
    empty = history_features(target, pd.DatetimeIndex([]), modis.iloc[0:0])
    assert empty["past_viirs_records"] is None and np.isnan(empty["past_viirs_log_records"])
    assert empty["past_modis_missing"] == 1


def test_adjacent_tiles_are_mosaicked_and_missing_edge_is_unknown(tmp_path):
    primary, neighbor = tmp_path / "primary.h5", tmp_path / "neighbor.h5"
    main = np.full((4, 4), 5, dtype=np.uint8)
    other = np.full((4, 4), 5, dtype=np.uint8)
    other[1, 3] = 9
    make_granule(primary, main)
    make_granule(neighbor, other, x_start=-1400)
    geom = {"xllcorner": -1100, "yllcorner": 1700, "nrows": 2, "ncols": 2, "cellsize_m": 100}
    incomplete = read_mask(primary, tmp_path, geom, "2021-01-15")
    assert incomplete["event"] is None and incomplete["masked_pixels"] == 2
    complete = read_mask(primary, tmp_path, geom, "2021-01-15", [neighbor])
    assert complete["source_tiles_complete"] and complete["event"] == 1 and complete["fire_pixels"] == 1


def test_daily_persistence_requires_past_production():
    data = pd.DataFrame({"region_id": ["a"] * 3, "period_start": pd.to_datetime(["2021-01-01", "2021-01-08", "2021-01-15"]),
                         "production_time": ["2021-01-02 00:00:00", "2021-01-16 00:00:00", "2021-01-16 01:00:00"],
                         "event": pd.array([1, 0, 1], dtype="Int64")})
    result = add_previous_daily_mask(data).reset_index(drop=True)
    assert pd.isna(result.loc[0, "past_viirs_mask_event"])
    assert result.loc[1, "past_viirs_mask_event"] == 1
    assert pd.isna(result.loc[2, "past_viirs_mask_event"])
