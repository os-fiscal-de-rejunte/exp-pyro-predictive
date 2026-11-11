import pytest

from earthfire.viirs_daily_halo import adjacent_tile


def test_identifies_adjacent_tile_and_rejects_corners():
    geometry = {"xllcorner": -1100, "yllcorner": 1700, "nrows": 2, "ncols": 2, "cellsize_m": 100}
    tile, latitude, longitude = adjacent_tile(geometry, -1000, 2000, 4, 4, "h12v08")
    assert tile == "h11v08" and -90 < latitude < 90 and -180 < longitude < 180
    inside = {**geometry, "xllcorner": -900}
    assert adjacent_tile(inside, -1000, 2000, 4, 4, "h12v08") is None
    corner = {**geometry, "yllcorner": 1950}
    with pytest.raises(ValueError, match="múltiplas bordas"):
        adjacent_tile(corner, -1000, 2000, 4, 4, "h12v08")
