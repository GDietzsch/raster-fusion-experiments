"""Show fixed assessment locations and spatial exclusion; no outcome-based choices."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
SOURCE = (Path(__file__).resolve().parents[2] / 'raster_fusion_ai/runs/enriched_cnn/20260926T223539611207Z/horizon_0/samples.npz')
with np.load(SOURCE) as s:
    vp = s['validacao_modelo_positions']
    tp = s['ajuste_positions']
fig, axes = plt.subplots(2, 2, figsize=(11, 6.6), sharex=True, sharey=True, layout='constrained')
rng = np.random.default_rng(42)
for row, fold in enumerate(('west', 'east')):
    with np.load(OUT/f'H0_{fold}/indices.npz') as z:
        vi = z['validation_ids']
        vi = rng.choice(vi, min(7000, len(vi)), replace=False)
        for col, gap in enumerate((750, 41000)):
            ax = axes[row, col]
            ti = z[f'ajuste_{gap}_ids']
            ti = rng.choice(ti, min(7000, len(ti)), replace=False)
            ax.scatter(tp[ti, 1]*.25, -tp[ti, 0]*.25, s=.4, alpha=.4, color='#356fa3', label='Training centers')
            ax.scatter(vp[vi, 1]*.25, -vp[vi, 0]*.25, s=.4, alpha=.5, color='#c45432', label='Assessment centers')
            ax.set_title(f'{fold.capitalize()} assessment | minimum gap {gap/1000:g} km')
            ax.set_aspect('equal')
            ax.spines[['top','right']].set_visible(False)
            ax.axvline(5053.5*.25, color='#777777', linestyle='--', lw=.8)
            if row == 1:
                ax.set_xlabel('Easting relative to raster origin (km)')
            if col == 0:
                ax.set_ylabel('Northing relative to raster origin (km)')
axes[0,0].legend(loc='upper left', markerscale=6, fontsize=8, frameon=True)
fig.suptitle('H0 spatial exclusion design (display subsample; all years pooled)', fontsize=13)
fig.savefig(OUT/'spatial_design.png', dpi=220)
plt.close(fig)
