"""Paired, post-hoc spatial exclusion sensitivity on frozen winning configurations.

Only development samples and input patch caches are read. Original runs are read-only.
Run with raster_fusion_ai/.venv/Scripts/python.exe; --prepare performs no fitting.
"""
from pathlib import Path
import argparse
import hashlib
import json
import logging
import platform
import shutil
import sys
import time

ROOT = (Path(__file__).resolve().parents[2] / 'raster_fusion_ai')
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import rasterio
import scipy
from scipy.ndimage import minimum_filter
from scipy.spatial import cKDTree
import torch
import yaml
from src.services import enriched_cnn as cnn
from src.services import enriched_linear as common
from src.splitting.spatial_split import split_grid

CNN_RUN = ROOT / 'runs/enriched_cnn/20260926T223539611207Z'
RESNET_RUN = ROOT / 'runs/enriched_resnet/20260927T140640055671Z'
CASES = ((0, 'CNN', CNN_RUN), (1, 'ResNet', RESNET_RUN))
GAPS = (750, 41000)
SEED = 4201


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def save(p, x):
    common._write_json(p, x)


def digest(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def sample_indices(distances, years, gap, counts, priorities):
    parts = []
    for year in sorted(counts):
        valid = np.flatnonzero((years == year) & (distances >= gap))
        n = counts[year]
        assert len(valid) >= n > 0
        chosen = valid[np.argsort(priorities[valid], kind='stable')[:n]]
        parts.append(chosen)
    return np.sort(np.concatenate(parts))


def self_test():
    # Distances ignore year: prior-year copies of assessment locations must be purged.
    t = cKDTree(np.array([[0., 0.], [100000., 100000.]]))
    d = t.query(np.array([[0., 0.], [0., 750.], [0., 40999.], [0., 41000.]]))[0]
    assert np.array_equal(d >= 41000, [False, False, False, True])
    years = np.array([2020, 2020, 2021, 2021])
    dist = np.array([1000., 42000., 43000., 44000.])
    priorities = np.array([0.1, 0.2, 0.3, 0.4])
    a = sample_indices(dist, years, 750, {2020: 1, 2021: 1}, priorities)
    b = sample_indices(dist, years, 41000, {2020: 1, 2021: 1}, priorities)
    assert np.array_equal(a, [0, 2]) and np.array_equal(b, [1, 2])
    assert np.array_equal(years[a], years[b])
    assert np.all(dist[b] >= 41000)


def prepare():
    self_test()
    if (OUT / 'prepared.json').exists():
        print('Preparation already completed; reusing frozen protocol.', flush=True)
        return
    cfg = yaml.safe_load((CNN_RUN / 'config.yaml').read_text(encoding='utf-8'))
    with rasterio.open(CNN_RUN / 'spatial_zones.tif') as ds:
        shape, transform, crs = ds.shape, ds.transform, str(ds.crs)
        original_zones = ds.read(1)
    assert np.allclose([abs(transform.a), abs(transform.e)], [250, 250], rtol=0, atol=1e-8)
    assert transform.b == 0 and transform.d == 0
    cut = shape[1] / 2
    model_specs = []
    for h, name, run in CASES:
        winner = read(run / f'horizon_{h}/winner.json')
        art = torch.load(winner['model_path'], map_location='cpu', weights_only=True)
        model_specs.append(dict(horizon=h, model=name, winner_path=str(run / f'horizon_{h}/winner.json'),
            source_model=winner['model_path'], source_model_sha256=digest(Path(winner['model_path'])),
            columns=art['columns'], feature_names=art['feature_names'], params=art['params'], epochs=art['epoch'],
            architecture=art.get('architecture', 'compact_cnn'), original_rmse=winner['validation_metrics']['rmse']))
    # Freeze choices before inspecting assessment target distributions or new predictions.
    protocol = dict(kind='post_hoc_spatial_exclusion_sensitivity', created_utc=common._utc(),
        models=model_specs, gaps_m=list(GAPS), folds=['west', 'east'], split_column=cut,
        fold_rule='Original validation sample centers west/east of full raster midpoint; no outcome-based boundary selection.',
        metric='Euclidean center distance in original projected CRS, pooled over all assessment years; nearest assessment center >= gap.',
        training_pool='Original final-fit ajuste + selecao_camadas only; never validation labels.',
        matching='Equal sample count separately in every original stage and target year, limited by 41 km availability. Shared random priorities; cap 100000 per stage-year.',
        sampling_seed=SEED, training_seed=4210, batch_size=512,
        normalization='Refit center-input and target scalers exclusively on selected training data for each condition.',
        epochs='Frozen original winning epoch count, trained from scratch. No early stopping or selection on assessment.',
        target_years={'H0_train_and_validation': [2020, 2021, 2022], 'H1_train': [2020, 2021], 'H1_validation': [2022]},
        external_data_accessed=False, shape=list(shape), transform=list(transform), crs=crs,
        limitations=[
            'Post-hoc: model families, channels and epochs were previously selected on development data, including original validation.',
            'Assessment is restricted to previously sampled original validation points, not all eligible raster cells.',
            'H0 original training sample mixes unequal sampling fractions from two zones; matching retains original stage-year composition.',
            'Two geographic halves and one training seed are sensitivity checks, not independent replications or spatial cross-validation folds.',
            'Exclusion changes training geography and covariate/target distributions despite equal sample counts; not a causal estimate of leakage.',
            'Gap of 41 km does not ensure disjoint CNN contexts (41 km per axis), derived KDE/graph supports, or independent errors.',
            'Cached inputs remain fixed, input-only context may cross held-out regions. This is not a rebuild of derived covariates within each split.',
            'Original 3-pixel buffer erodes both sides using Chebyshev distance. New 750 m is an explicit Euclidean center-gap control, not an exact rerun of original split.',
            'No external refit/evaluation: this experiment cannot establish the cause of the external performance drop.'],
        reference='https://doi.org/10.1111/ecog.02881')
    save(OUT / 'protocol.json', protocol)
    source_dir = OUT / 'source_snapshot'
    source_dir.mkdir(exist_ok=True)
    for path in [Path(__file__), ROOT/'src/services/enriched_cnn.py', ROOT/'src/services/enriched_linear.py', ROOT/'src/splitting/spatial_split.py']:
        shutil.copy2(path, source_dir / path.name)
    save(OUT/'environment.json', dict(python=sys.version, platform=platform.platform(), numpy=np.__version__,
        scipy=scipy.__version__, torch=torch.__version__, cuda=torch.cuda.is_available(),
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None))
    z = split_grid(shape, dict(cfg['split'], buffer_pixels=0))
    geometry = []
    for b in (3, 82, 164):
        counts = []
        for label in (1, 2, 3):
            mask = minimum_filter((z == label).astype('uint8'), size=2*b+1, mode='nearest') > 0
            counts.append(int(mask.sum()))
        geometry.append(dict(buffer_each_side_pixels=b, buffer_each_side_m=b*250,
                             zone1=counts[0], zone2=counts[1], zone3=counts[2],
                             scope='Full grid before common eligibility mask'))
    common._write_csv(OUT/'original_geometry_counts.csv', geometry)
    del z
    sources, coverage = [], []
    for h, name, _ in CASES:
        folder = CNN_RUN / f'horizon_{h}'
        sample_path = folder/'samples.npz'
        assert digest(sample_path) == read(folder/'patch_bank.json')['sample_source_sha256']
        stages = cnn._load_stages(sample_path)
        cnn._check_stages(stages, original_zones, h)
        for path in [sample_path, folder/'patch_bank.json'] + [folder/f'{s}_patches.npy' for s in cnn.STAGES]:
            sources.append(dict(path=str(path), bytes=path.stat().st_size, sha256=digest(path)))
        vx, vy, vp, vyears = stages['validacao_modelo']
        for fold in ('west', 'east'):
            val_ids = np.flatnonzero(vp[:, 1] < cut if fold == 'west' else vp[:, 1] >= cut)
            assert len(val_ids) >= 1000
            pixel_scale = np.array([abs(transform.e), abs(transform.a)])
            tree = cKDTree(vp[val_ids].astype(np.float64) * pixel_scale)
            arrays = {'validation_ids': val_ids}
            training_counts = {gap: 0 for gap in GAPS}
            for s_index, stage in enumerate(('ajuste', 'selecao_camadas')):
                x, y, p, years = stages[stage]
                distances = tree.query(p.astype(np.float64)*pixel_scale, workers=4)[0]
                priorities = np.random.default_rng(SEED+h*10000+s_index*100+(fold=='east')).random(len(y))
                counts = {int(year): min(100000, int(((years==year)&(distances>=41000)).sum())) for year in np.unique(years)}
                assert min(counts.values()) >= 1000, (h, fold, stage, counts)
                for gap in GAPS:
                    ids = sample_indices(distances, years, gap, counts, priorities)
                    arrays[f'{stage}_{gap}_ids'] = ids
                    arrays[f'{stage}_{gap}_distance_m'] = distances[ids]
                    assert np.all(distances[ids] >= gap)
                    training_counts[gap] += len(ids)
                    for year in sorted(counts):
                        chosen = ids[years[ids] == year]
                        coverage.append(dict(horizon=h, model=name, fold=fold, gap_m=gap, stage=stage,
                            target_year=year, eligible=int(((years==year)&(distances>=gap)).sum()),
                            selected=len(chosen), min_distance_m=float(distances[chosen].min()),
                            target_mean=float(y[chosen].mean()), target_positive=int((y[chosen]>0).sum())))
            assert training_counts[750] == training_counts[41000]
            dest = OUT/f'H{h}_{fold}'
            dest.mkdir(exist_ok=True)
            np.savez_compressed(dest/'indices.npz', **arrays)
            save(dest/'assessment.json', dict(n=len(val_ids), years={str(int(yr)): int((vyears[val_ids]==yr).sum()) for yr in np.unique(vyears[val_ids])},
                 target_mean=float(vy[val_ids].mean()), target_positive=int((vy[val_ids]>0).sum()),
                 training_n=training_counts[750], indices_sha256=digest(dest/'indices.npz')))
            print(f'PREPARED H{h} {fold}: train={training_counts[750]} eval={len(val_ids)}', flush=True)
    common._write_csv(OUT/'training_coverage.csv', coverage)
    save(OUT/'sources.json', sources)
    save(OUT/'prepared.json', dict(status='prepared', protocol_sha256=digest(OUT/'protocol.json'),
         self_tests_passed=True, created_utc=common._utc()))
    print('PREPARED: 8 fits, fixed 25/32 epochs, one seed, two geographic halves.', flush=True)


def run():
    prepare()
    assert digest(OUT/'protocol.json') == read(OUT/'prepared.json')['protocol_sha256']
    for source in read(OUT/'sources.json'):
        path = Path(source['path'])
        assert path.stat().st_size == source['bytes'] and digest(path) == source['sha256']
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    settings = dict(random_state=4210, batch_size=512)
    protocol = read(OUT/'protocol.json')
    start = time.perf_counter()
    for spec in protocol['models']:
        h = spec['horizon']
        sample_dir = CNN_RUN/f'horizon_{h}'
        stages = cnn._load_stages(sample_dir/'samples.npz')
        banks = {s: np.load(sample_dir/f'{s}_patches.npy', mmap_mode='r') for s in cnn.STAGES}
        original = torch.load(spec['source_model'], map_location='cpu', weights_only=True)
        assert digest(Path(spec['source_model'])) == spec['source_model_sha256']
        for fold in protocol['folds']:
            folder = OUT/f'H{h}_{fold}'
            assert digest(folder/'indices.npz') == read(folder/'assessment.json')['indices_sha256']
            with np.load(folder/'indices.npz') as z:
                indices = {k: z[k] for k in z.files}
            vi = indices['validation_ids']
            valid = cnn._view(banks['validacao_modelo'], stages['validacao_modelo'][1], vi)
            if not (folder/'original_predictions.npy').exists():
                op = cnn.predict_view(original, valid, 512, device)
                np.save(folder/'original_predictions.npy', op)
                save(folder/'original_metrics.json', common.metrics(valid.y, op, subsets=False))
            for gap in protocol['gaps_m']:
                dest = folder/f'gap_{gap}m'
                dest.mkdir(exist_ok=True)
                if (dest/'metrics.json').exists():
                    print(f'REUSE H{h} {fold} gap={gap}', flush=True)
                    continue
                parts, ys, xs = [], [], []
                for stage in ('ajuste', 'selecao_camadas'):
                    ids = indices[f'{stage}_{gap}_ids']
                    parts.append((banks[stage], ids))
                    ys.append(stages[stage][1][ids])
                    xs.append(stages[stage][0][ids])
                train = cnn.PatchView(parts, ys)
                scale = cnn._scales(np.concatenate(xs), train.y, tuple(spec['columns']))
                print(f'START H{h} {fold} gap={gap}m n_train={len(train)} n_eval={len(valid)} epochs={spec["epochs"]}', flush=True)
                save(OUT/'status.json', dict(status='running', horizon=h, fold=fold, gap_m=gap, updated_utc=common._utc()))
                started = time.perf_counter()
                model, history, _ = cnn.fit_cnn(train, tuple(spec['columns']), scale, spec['params'],
                    spec['epochs'], settings, device, architecture=spec['architecture'])
                torch.save(model, dest/'model.pt')
                common._write_csv(dest/'training_history.csv', history)
                pred = cnn.predict_view(model, valid, 512, device)
                np.save(dest/'predictions.npy', pred)
                metric = common.metrics(valid.y, pred, subsets=False)
                metric.update(horizon=h, model=spec['model'], fold=fold, gap_m=gap,
                    n_train=len(train), epochs=len(history), seconds=time.perf_counter()-started,
                    model_sha256=digest(dest/'model.pt'), prediction_sha256=digest(dest/'predictions.npy'),
                    zero_rmse=float(np.sqrt(np.mean(valid.y**2))),
                    training_mean_rmse=float(np.sqrt(np.mean((valid.y-train.y.mean())**2))))
                save(dest/'metrics.json', metric)
                print(f'DONE H{h} {fold} gap={gap} RMSE={metric["rmse"]:.6f} R2={metric["r2"]:.6f} seconds={metric["seconds"]:.1f}', flush=True)
        del stages, banks
    save(OUT/'status.json', dict(status='fits_completed', elapsed_seconds=time.perf_counter()-start, updated_utc=common._utc()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
        handlers=[logging.FileHandler(OUT/'training.log', encoding='utf-8'), logging.StreamHandler(sys.stdout)])
    prepare() if args.prepare else run()
