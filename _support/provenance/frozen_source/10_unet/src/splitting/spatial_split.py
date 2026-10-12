"""Spatial units are assigned before any patches or learned preprocessing."""
import logging
import numpy as np
from scipy.ndimage import minimum_filter


def split_grid(shape: tuple[int, int], cfg: dict) -> np.ndarray:
    """Return geographic zones, independent of values/target availability.

    Buffer excludes b cells on each side of a boundary (Chebyshev distance).
    Fractions describe allocation of blocks; actual pixel fractions are reported.
    """
    h, w = shape
    method = cfg["method"].removesuffix("_split")
    fractions = np.array([cfg["train"], cfg["validation"], cfg["test"]])
    rng = np.random.default_rng(cfg.get("seed", 42))
    if method in ("spatial_stripes", "stripes"):
        cuts = np.rint(np.cumsum(fractions) * w).astype(int)
        zones = np.ones(shape, dtype="uint8")
        zones[:, cuts[0]:cuts[1]] = 2
        zones[:, cuts[1]:] = 3
    else:
        if method in ("random", "random_split"):
            logging.warning("Random split is a diagnostic; spatial autocorrelation can inflate performance")
            block = 1
        elif method in ("spatial_block", "block"):
            block = int(cfg.get("block_size", 96))
            if block < 1:
                raise ValueError("block_size must be positive")
        else:
            raise ValueError(f"Unknown split method: {method}")
        nh, nw = (h+block-1)//block, (w+block-1)//block
        n = nh*nw
        if n < 3:
            raise ValueError("At least three spatial blocks are required; reduce block_size")
        counts = np.maximum(1, np.floor(fractions*n).astype(int))
        while counts.sum() > n:
            candidates = np.where(counts > 1)[0]
            counts[candidates[np.argmax(counts[candidates])]] -= 1
        while counts.sum() < n:
            counts[np.argmax(fractions*n-counts)] += 1
        labels = rng.permutation(np.repeat(np.arange(1, 4, dtype="uint8"), counts)).reshape(nh, nw)
        zones = np.repeat(np.repeat(labels, block, axis=0), block, axis=1)[:h, :w]
    buffer = int(cfg.get("buffer_pixels", 0))
    if buffer:
        original = zones.copy()
        for label in (1, 2, 3):
            interior = minimum_filter((original == label).astype("uint8"), size=2*buffer+1, mode="nearest") > 0
            zones[(original == label) & ~interior] = 0
    if any(not (zones == label).any() for label in (1, 2, 3)):
        raise ValueError("A split is empty after buffering. Increase region/blocks or reduce buffer; random splits usually require buffer=0.")
    return zones


def context_mask(zones: np.ndarray, radius: int) -> np.ndarray:
    """Centers whose full input neighborhood stays inside a single zone."""
    result = np.zeros_like(zones, dtype=bool)
    for label in (1, 2, 3):
        result |= minimum_filter((zones == label).astype("uint8"), size=2*radius+1, mode="nearest") > 0
    return result
