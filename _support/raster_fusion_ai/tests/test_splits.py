import numpy as np
import pytest
from scipy.ndimage import maximum_filter
from src.splitting.spatial_split import split_grid, context_mask
from src.splitting.split_visualization import save_split


def config(method="spatial_block"):
    return dict(method=method, train=.7, validation=.15, test=.15, block_size=16, buffer_pixels=2, seed=42)


@pytest.mark.parametrize("method", ["spatial_block", "spatial_stripes", "random_split"])
def test_splits_reproducible(method):
    cfg = config(method)
    if method == "random_split":
        cfg["buffer_pixels"] = 0
    a = split_grid((96, 112), cfg)
    np.testing.assert_array_equal(a, split_grid(a.shape, cfg))
    assert all((a == k).any() for k in (1, 2, 3))
    if cfg["buffer_pixels"]:
        for k in (1, 2, 3):
            assert not np.any((maximum_filter((a == k).astype(int), size=5) > 0) & (a != k) & (a != 0))


def test_context_does_not_cross_boundary():
    a = np.ones((10, 10), dtype="uint8")
    a[:, 5:] = 2
    valid = context_mask(a, 2)
    assert not valid[:, 3:7].any()
    assert valid[:, :3].all()


def test_split_outputs(tmp_path, profile):
    split = split_grid((24, 30), {**config("spatial_stripes"), "buffer_pixels": 0})
    save_split(split, profile, tmp_path)
    assert (tmp_path / "spatial_split.png").stat().st_size > 1000
    assert (tmp_path / "spatial_split.tif").exists()
