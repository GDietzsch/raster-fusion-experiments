"""Persist split IDs, counts and a fixed categorical legend."""
from pathlib import Path
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from src.geospatial.raster_io import write_raster


def save_split(split: np.ndarray, profile: dict, directory: Path, stem: str = "spatial_split") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    write_raster(directory / f"{stem}.tif", split, profile, categorical=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ListedColormap(["#eeeeee", "#3b82f6", "#fbbf24", "#dc2626"])
    im = ax.imshow(split, cmap=colors, norm=BoundaryNorm(np.arange(-0.5, 4.5), 4), interpolation="nearest")
    bar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3])
    bar.ax.set_yticklabels(["Excluded / invalid", "Training", "Validation", "Test"])
    ax.set(xlabel="Column", ylabel="Row", title="Spatial split")
    fig.tight_layout()
    fig.savefig(directory / f"{stem}.png", dpi=140)
    plt.close(fig)
    rows = []
    for label, name in enumerate(["excluded", "training", "validation", "test"]):
        count = int((split == label).sum())
        rows.append(dict(id=label, split=name, pixels=count, percent=100*count/split.size))
        logging.info("%s: %s pixels (%.2f%%)", name, count, rows[-1]["percent"])
    pd.DataFrame(rows).to_csv(directory / f"{stem}_statistics.csv", index=False)
