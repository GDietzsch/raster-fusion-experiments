"""Render paired target/prediction maps for every external winner evaluation."""
from argparse import ArgumentParser
import gc
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT


def aligned_target(path: Path, profile: dict) -> np.ndarray:
    with rasterio.open(path) as source:
        with WarpedVRT(source, crs=profile['crs'], transform=profile['transform'],
                       width=profile['width'], height=profile['height'],
                       nodata=float('nan'), dtype='float32',
                       resampling=Resampling.nearest) as vrt:
            return vrt.read(1, masked=True).filled(np.nan).astype(np.float32)


def render_one(record: dict, output: Path) -> dict:
    with rasterio.open(record['prediction']) as source:
        profile = source.profile.copy()
        prediction = source.read(1, masked=True).filled(np.nan).astype(np.float32)
    target = aligned_target(Path(record['target_source']), profile)
    support = np.isfinite(target) & np.isfinite(prediction)
    n = int(support.sum())
    if n != record['common_pixels'] or n != record['metrics']['n']:
        raise ValueError(f"Common support mismatch H{record['horizon']} target {record['target_year']}")
    limit = max(float(np.max(np.abs(target[support]))),
                float(np.max(np.abs(prediction[support]))))
    if not np.isfinite(limit) or limit <= 0:
        raise ValueError('Invalid shared color limit')
    norm = SymLogNorm(linthresh=0.5, linscale=1, vmin=-limit, vmax=limit)
    stride = 2
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    image = None
    for axis, title, values in zip(axes, ('Target original', 'Previsão do vencedor'),
                                   (target, prediction)):
        shown = np.where(support, values, np.nan)[::stride, ::stride]
        image = axis.imshow(shown, cmap='RdBu_r', norm=norm, interpolation='nearest')
        axis.set_title(title, fontsize=16, pad=10)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_aspect('equal')
    fig.colorbar(image, ax=list(axes), fraction=0.035, pad=0.025,
                 label='Intensidade do target (unidades originais)')
    fig.suptitle(f"Teste externo H{record['horizon']} · entrada {record['input_year']} → "
                 f"target {record['target_year']}", fontsize=17)
    fig.supxlabel(f"Mesmo suporte válido: {n:,} pixels · RMSE {record['metrics']['rmse']:.4f} · "
                  "azul: previsão negativa · mapas reduzidos 2× para exibição",
                  fontsize=11)
    path = output / f"teste_externo_H{record['horizon']}_{record['input_year']}_{record['target_year']}.png"
    fig.savefig(path, dpi=600, facecolor='white', metadata={
        'Title': f"H{record['horizon']} target {record['target_year']}: original versus previsão",
        'Description': 'Mesma máscara e escala de cores; fonte numérica são os GeoTIFFs originais.'})
    plt.close(fig)
    del fig, axes, target, prediction, support
    gc.collect()
    return {'horizon': record['horizon'], 'input_year': record['input_year'],
            'target_year': record['target_year'], 'path': str(path), 'dpi': 600,
            'figure_pixels': [9000, 3600], 'display_stride': stride,
            'common_pixels': n, 'shared_color_limit': [-limit, limit],
            'prediction': record['prediction'], 'target': record['target_source']}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    output = args.output.resolve()
    if not output.is_dir():
        raise FileNotFoundError(output)
    status = json.loads((run / 'run.json').read_text(encoding='utf-8'))
    if status.get('status') != 'completed':
        raise ValueError('Run not completed')
    manifest = []
    for horizon in (0, 1):
        records = sorted((run / f'horizon_{horizon}').glob('external_target_*_metrics.json'))
        if len(records) != (3 if horizon == 0 else 1):
            raise ValueError(f'Unexpected number of external evaluations for H{horizon}')
        for path in records:
            record = json.loads(path.read_text(encoding='utf-8'))
            item = render_one(record, output)
            manifest.append(item)
            print(item['path'], flush=True)
    (output / 'figuras_teste_externo.json').write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()
