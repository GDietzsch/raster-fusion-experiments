import numpy as np
import pytest
from rasterio.transform import from_origin


@pytest.fixture
def profile():
    return dict(driver="GTiff", height=24, width=30, count=1, dtype="float32", crs="EPSG:31983", transform=from_origin(500000, 7500000, 30, 30), nodata=-9999)


@pytest.fixture
def raster_file(tmp_path, profile):
    from src.geospatial.raster_io import write_raster
    a = np.arange(720, dtype="float32").reshape(24, 30)
    a[0, 0] = np.nan
    path = tmp_path / "input.tif"
    write_raster(path, a, profile)
    return path, a
