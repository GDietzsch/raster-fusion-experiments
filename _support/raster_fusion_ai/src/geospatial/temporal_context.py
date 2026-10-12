"""Input-only neighborhood features, identically computed for fitting/inference."""
from contextlib import ExitStack
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from scipy.ndimage import uniform_filter

from src.geospatial.raster_io import check_dataset


def validate_sizes(sizes):
    if (not sizes or len(set(sizes)) != len(sizes)
            or any(type(s) is not int or s < 3 or s % 2 == 0 for s in sizes)):
        raise ValueError('Neighborhood sizes must be unique odd integers >= 3')


def feature_names(layers, sizes):
    validate_sizes(sizes)
    names = [layer['name'] for layer in layers]
    result = names.copy()
    for size in sizes:
        for statistic in ('mean', 'std'):
            result.extend(f'{name}__{statistic}_{size}x{size}' for name in names)
    if len(set(result)) != len(result):
        raise ValueError('Feature names must be unique')
    return result


def neighborhood_values(values, sizes):
    """Finite neighbors only; zero counts, out-of-coverage cells do not.

    Population standard deviation. All operations use float64 internally.
    No reflection or invented zero observations at the edge of the raster.
    """
    validate_sizes(sizes)
    valid = np.isfinite(values)
    filled = np.where(valid, values, 0).astype(np.float64)
    features = [values.astype(np.float32)]
    for size in sizes:
        kernel = (size, size, 1)
        weight = uniform_filter(valid.astype(float), kernel, mode='constant', cval=0)
        total = uniform_filter(filled, kernel, mode='constant', cval=0)
        square = uniform_filter(filled * filled, kernel, mode='constant', cval=0)
        mean = np.divide(total, weight, out=np.full_like(total, np.nan), where=weight > 1e-12)
        second = np.divide(square, weight, out=np.full_like(square, np.nan), where=weight > 1e-12)
        std = np.sqrt(np.maximum(second - mean * mean, 0))
        features.extend([mean.astype(np.float32), std.astype(np.float32)])
    return np.concatenate(features, axis=-1)


def spatial_windows(paths, methods, profile, tile_size, sizes):
    """Read halo around tiles so features never depend on processing tile size."""
    validate_sizes(sizes)
    if len(paths) != len(methods) or tile_size < 1:
        raise ValueError('Invalid input methods or tile size')
    radius = max(sizes) // 2
    height, width = profile['height'], profile['width']
    with ExitStack() as stack:
        readers = []
        for path, method in zip(paths, methods):
            source = stack.enter_context(rasterio.open(path))
            check_dataset(source)
            if source.nodata == 0:
                raise ValueError(f'{path}: zero must not be declared NoData')
            readers.append(stack.enter_context(WarpedVRT(source,
                crs=profile['crs'], transform=profile['transform'], width=width, height=height,
                dtype='float32', nodata=float('nan'), resampling=Resampling[method], warp_mem_limit=64)))
        for row in range(0, height, tile_size):
            for col in range(0, width, tile_size):
                h, w = min(tile_size, height-row), min(tile_size, width-col)
                r0, c0 = max(0, row-radius), max(0, col-radius)
                r1, c1 = min(height, row+h+radius), min(width, col+w+radius)
                read_window = Window(c0, r0, c1-c0, r1-r0)
                raw = np.stack([ds.read(1, window=read_window, masked=True).filled(np.nan)
                                for ds in readers], axis=-1)
                features = neighborhood_values(raw, sizes)
                yield Window(col, row, w, h), features[row-r0:row-r0+h, col-c0:col-c0+w]
