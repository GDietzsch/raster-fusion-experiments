"""Single-band GeoTIFF IO with explicit grid and invalid-pixel handling."""
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window


def windows(height: int, width: int, size: int = 256):
    for row in range(0, height, size):
        for col in range(0, width, size):
            yield Window(col, row, min(size, width-col), min(size, height-row))


def read_raster(path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as ds:
        check_dataset(ds)
        data = ds.read(1, masked=True).astype("float32").filled(np.nan)
        data[~np.isfinite(data)] = np.nan
        return data, ds.profile.copy()


def check_dataset(ds) -> None:
    if ds.count != 1:
        raise ValueError(f"Expected one band: {ds.name} has {ds.count}")
    if ds.crs is None:
        raise ValueError(f"Missing CRS: {ds.name}")
    if abs(ds.transform.determinant) < 1e-20:
        raise ValueError(f"Singular affine transform: {ds.name}")


def write_raster(path: Path, data: np.ndarray, profile: dict, *, categorical: bool = False) -> None:
    """Write a new product, preserving grid; callers must protect input paths."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if data.shape != (profile["height"], profile["width"]):
        raise ValueError("Array dimensions do not match output grid")
    out = {k: profile[k] for k in ("height", "width", "crs", "transform")}
    out.update(driver="GTiff", count=1, dtype="uint8" if categorical else "float32",
               nodata=0 if categorical else -9999.0, compress="deflate")
    with rasterio.open(path, "w", **out) as ds:
        for win in windows(out["height"], out["width"]):
            block = data[win.toslices()]
            ds.write(np.where(np.isfinite(block), block, out["nodata"]).astype(out["dtype"]), 1, window=win)


def same_grid(ds, reference: dict) -> bool:
    return (ds.crs == reference["crs"] and ds.width == reference["width"]
            and ds.height == reference["height"] and ds.transform.almost_equals(reference["transform"], precision=1e-9))
