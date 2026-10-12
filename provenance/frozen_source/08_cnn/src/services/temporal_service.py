"""Temporal pilot: common geographic grid, train-only scaling and uniform sampling.

Uses the complete eligible training support for min/max, and bounded uniform
samples for model fitting/validation. These are tabular pilots, not CNN runs.
External regions and spatial zone 3 are never evaluated by this workflow.
Same-year models use spatial validation; one-year-ahead models use temporal
validation on the same geographic training zone.
"""
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
import csv
import hashlib
import json
import logging
import time

import joblib
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
import yaml
from threadpoolctl import threadpool_limits

from src.geospatial.raster_io import check_dataset, write_raster
from src.splitting.spatial_split import split_grid
from src.tabular_pipeline.models import make_model
from src.utils.config import load_layers, save_json
from src.evaluation.metrics import regression_metrics


class UniformSample:
    """Keep the k smallest iid random priorities: uniform without replacement."""
    def __init__(self, capacity: int, seed: int):
        if capacity < 1:
            raise ValueError('Sample capacity must be positive')
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)
        self.keys = np.empty(0)
        self.values = None
        self.positions = np.empty((0, 2), dtype=np.int32)

    def add(self, values: np.ndarray, positions: np.ndarray):
        if not len(values):
            return
        keys = self.rng.random(len(values))
        if len(keys) > self.capacity:
            selected = np.argpartition(keys, self.capacity - 1)[:self.capacity]
            keys, values, positions = keys[selected], values[selected], positions[selected]
        self.keys = np.concatenate((self.keys, keys))
        self.values = values.copy() if self.values is None else np.concatenate((self.values, values))
        self.positions = np.concatenate((self.positions, positions))
        if len(self.keys) > self.capacity:
            selected = np.argpartition(self.keys, self.capacity - 1)[:self.capacity]
            self.keys, self.values, self.positions = self.keys[selected], self.values[selected], self.positions[selected]


class MinMax:
    def __init__(self, names):
        self.names = names
        self.minimum = np.full(len(names), np.inf)
        self.maximum = np.full(len(names), -np.inf)
        self.count = 0

    def update(self, values):
        if len(values):
            self.minimum = np.minimum(self.minimum, values.min(axis=0))
            self.maximum = np.maximum(self.maximum, values.max(axis=0))
            self.count += len(values)

    def parameters(self):
        if not self.count:
            raise ValueError('No valid training pixels for normalization')
        return [dict(name=name, method='minmax', center=float(lo),
                     scale=float(hi-lo) if hi-lo > 1e-12 else 1., maximum=float(hi),
                     training_count=self.count)
                for name, lo, hi in zip(self.names, self.minimum, self.maximum)]


def scale_values(values, parameters):
    center = np.array([p['center'] for p in parameters], dtype=np.float64)
    scale = np.array([p['scale'] for p in parameters], dtype=np.float64)
    return ((values - center) / scale).astype(np.float32)


def original_target(values, parameter):
    return np.asarray(values, dtype=np.float64) * parameter['scale'] + parameter['center']


def _resolve(base, value):
    return (base / value).resolve()


def protocol_for(cfg, horizon):
    return cfg['temporal']['protocols'][str(horizon)]


def read_config(path):
    path = path.resolve()
    cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
    base = path.parent
    cfg['_region'] = _resolve(base, cfg['data']['region_dir'])
    cfg['_reference'] = _resolve(base, cfg['data']['reference'])
    cfg['_manifest'] = _resolve(base, cfg['data']['layers_manifest'])
    cfg['_output'] = _resolve(base, cfg['output']['directory'])
    horizons = cfg['temporal']['horizons']
    if not horizons or len(set(horizons)) != len(horizons) or any(h not in (0, 1) for h in horizons):
        raise ValueError('Supported horizons are 0 and 1, without duplicates')
    if not cfg['data'].get('zero_is_valid', False):
        raise ValueError('This workflow requires explicit zero_is_valid: true')
    for horizon in horizons:
        protocol = protocol_for(cfg,horizon)
        training,validation=protocol['training_target_years'],protocol['validation_target_years']
        if not training or not validation or len(set(training)) != len(training) or len(set(validation)) != len(validation):
            raise ValueError('Require nonempty, unique target years')
        if protocol['validation_mode'] == 'temporal':
            if max(training) >= min(validation) or protocol['validation_zone'] != 1:
                raise ValueError('Temporal validation requires later target years on training zone 1')
        elif protocol['validation_mode'] == 'spatial':
            if set(training) != set(validation) or protocol['validation_zone'] != 2:
                raise ValueError('Spatial validation uses the same years but held-out zone 2')
        else:
            raise ValueError('Unknown validation mode')
        if cfg['_reference'] not in {cfg['_region'] / str(year) / 'target.tif' for year in training}:
            raise ValueError('Reference must be a training-year target, not a held-out target')
    fractions = [cfg['split'][k] for k in ('train', 'validation', 'test')]
    if any(v <= 0 for v in fractions) or not np.isclose(sum(fractions), 1.):
        raise ValueError('Require positive split fractions summing to 1')
    if cfg['split']['method'] != 'spatial_block':
        raise ValueError('Temporal pilots require spatial_block splitting')
    if cfg['split']['block_size'] < 1 or cfg['split']['buffer_pixels'] < 0:
        raise ValueError('Invalid spatial block/buffer size')
    for key in ('samples_per_training_year', 'samples_per_validation_year', 'window_size', 'cpu_threads'):
        if cfg['pilot'][key] < 1:
            raise ValueError(f'{key} must be positive')
    if not cfg['pilot']['models'] or set(cfg['pilot']['models']) - {'ridge', 'hist_gradient_boosting', 'linear'}:
        raise ValueError('Pilot supports ridge, linear and hist_gradient_boosting')
    if cfg['pilot']['models'].get('hist_gradient_boosting', {}).get('early_stopping', False) is not False:
        raise ValueError('Disable HGB internal random holdout; use the external temporal/spatial validation')
    if not 72 <= cfg.get('figures',{}).get('dpi',600) <= 1200:
        raise ValueError('Figure DPI must be between 72 and 1200')
    layers = load_layers(cfg['_manifest'])
    if any(v.get('normalization') != 'minmax' for v in layers):
        raise ValueError('Pilot requires explicit minmax for all inputs')
    return cfg, layers


def pair_paths(cfg, layers, target_year, horizon):
    folder = cfg['_region'] / str(target_year-horizon) / 'inputs'
    return [folder / layer['filename'] for layer in layers] + [cfg['_region'] / str(target_year) / 'target.tif']


def _windows(profile, size):
    for row in range(0, profile['height'], size):
        for col in range(0, profile['width'], size):
            yield Window(col, row, min(size, profile['width']-col), min(size, profile['height']-row))


def aligned_windows(paths, methods, profile, size):
    with ExitStack() as stack:
        readers = []
        for path, method in zip(paths, methods):
            source = stack.enter_context(rasterio.open(path))
            check_dataset(source)
            if source.nodata == 0:
                raise ValueError(f'{path} declares zero as NoData, contrary to data semantics')
            readers.append(stack.enter_context(WarpedVRT(source, crs=profile['crs'],
                transform=profile['transform'], width=profile['width'], height=profile['height'],
                dtype='float32', nodata=float('nan'), resampling=Resampling[method], warp_mem_limit=64)))
        for window in _windows(profile, size):
            values = np.stack([reader.read(1, window=window, masked=True).astype('float32').filled(np.nan)
                               for reader in readers], axis=-1)
            yield window, values


def collect(cfg, layers, profile, zones, years, horizon, label, capacity, stats=None):
    samples, positions, sample_years, counts = [], [], [], []
    methods = [v['resampling'] for v in layers] + [cfg['data']['target_resampling']]
    for year in years:
        logging.info('Reading horizon %s: inputs %s -> target %s, spatial zone %s', horizon, year-horizon, year, label)
        sample = UniformSample(capacity, cfg['pilot']['seed'] + year*10 + horizon)
        eligible = zero_targets = 0
        last_progress = time.monotonic()
        for window, values in aligned_windows(pair_paths(cfg,layers,year,horizon), methods, profile, cfg['pilot']['window_size']):
            zone = zones[window.toslices()]
            valid = (zone == label) & np.isfinite(values).all(axis=-1)
            vectors = values[valid]
            if stats is not None:
                stats.update(vectors)
            eligible += len(vectors)
            zero_targets += int(np.count_nonzero(vectors[:,-1] == 0))
            rows, cols = np.where(valid)
            pixel_positions = np.column_stack((rows+int(window.row_off), cols+int(window.col_off))).astype('int32')
            sample.add(vectors, pixel_positions)
            if time.monotonic() - last_progress > 25:
                logging.info('Horizon %s target %s: row %s/%s, %s eligible pixels so far',
                    horizon,year,int(window.row_off),profile['height'],eligible)
                last_progress = time.monotonic()
        if sample.values is None:
            raise ValueError(f'No eligible pixels for target {year}, horizon {horizon}, zone {label}')
        samples.append(sample.values)
        positions.append(sample.positions)
        sample_years.append(np.full(len(sample.values),year,dtype=np.int32))
        counts.append(dict(input_year=year-horizon,target_year=year,zone=label,
            eligible_pixels=eligible,zero_target_pixels=zero_targets,sampled_pixels=len(sample.values)))
        logging.info('Year %s: %s eligible pixels, %s sampled; zeros retained',year,eligible,len(sample.values))
    return np.concatenate(samples), np.concatenate(positions), np.concatenate(sample_years), counts


def _score(target, prediction):
    if not np.isfinite(prediction).all():
        raise ValueError('Model produced non-finite predictions')
    return regression_metrics(target, prediction, np.ones(target.shape,dtype=bool))


def _hash(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def _grid(profile):
    return dict(crs=profile['crs'].to_wkt(),transform=list(profile['transform'])[:6],
                width=profile['width'],height=profile['height'])


def train(config_path):
    cfg, layers = read_config(config_path)
    all_paths = {cfg['_reference']}
    for horizon in cfg['temporal']['horizons']:
        protocol=protocol_for(cfg,horizon)
        for year in protocol['training_target_years'] + protocol['validation_target_years']:
            paths = pair_paths(cfg,layers,year,horizon)
            expected = {layer['filename'] for layer in layers}
            folder = paths[0].parent
            actual = {p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.suffix.lower() in ('.tif','.tiff')}
            if actual != expected:
                raise ValueError(f'Input schema mismatch in {folder}: missing={expected-actual}, extra={actual-expected}')
            all_paths.update(paths)
    for path in all_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    with rasterio.open(cfg['_reference']) as ds:
        check_dataset(ds)
        profile = ds.profile.copy()
    # Every raster is checked before writing a run or beginning any training.
    for path in sorted(all_paths):
        with rasterio.open(path) as ds:
            check_dataset(ds)
            if ds.nodata == 0:
                raise ValueError(f'{path}: zero is declared NoData but must be valid')
            if ds.crs != profile['crs'] or not np.allclose(ds.res, (abs(profile['transform'].a),abs(profile['transform'].e)),rtol=1e-6):
                raise ValueError(f'{path}: changed CRS/resolution requires a separate data review')
    run = cfg['_output'] / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run.mkdir(parents=True,exist_ok=False)
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler = logging.FileHandler(run/'training.log',encoding='utf-8')
    handler.setFormatter(formatter)
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    log.addHandler(handler)
    log.addHandler(stream)
    meta = dict(status='running',kind='temporal_tabular_pilot',started=datetime.now(timezone.utc).isoformat(),
                final_external_test_used=False,run=str(run))
    save_json(run/'run.json',meta)
    try:
        cfg_snapshot = {k:v for k,v in cfg.items() if not k.startswith('_')}
        cfg_snapshot['data'] = {**cfg_snapshot['data'], 'region_dir':str(cfg['_region']),
                               'reference':str(cfg['_reference']),'layers_manifest':str(cfg['_manifest'])}
        cfg_snapshot['output'] = {'directory':str(cfg['_output'])}
        (run/'config.yaml').write_text(yaml.safe_dump(cfg_snapshot,sort_keys=False),encoding='utf-8')
        (run/'layers.yaml').write_text(yaml.safe_dump({'layers':layers},sort_keys=False),encoding='utf-8')
        logging.info('Pilot run: %s; hashing %s original rasters',run,len(all_paths))
        fingerprints = [dict(path=str(p),bytes=p.stat().st_size,sha256=_hash(p)) for p in sorted(all_paths)]
        save_json(run/'fingerprints.json',fingerprints)
        import importlib.metadata
        import platform
        import sys
        save_json(run/'environment.json',dict(python=sys.version,platform=platform.platform(),
            packages={name:importlib.metadata.version(name) for name in ('numpy','scikit-learn','rasterio','joblib','threadpoolctl')}))
        zones = split_grid((profile['height'],profile['width']),cfg['split'])
        write_raster(run/'spatial_zones.tif',zones,profile,categorical=True)
        save_json(run/'grid.json',_grid(profile))
        records = []
        with threadpool_limits(limits=cfg['pilot']['cpu_threads']):
            for horizon in cfg['temporal']['horizons']:
                protocol=protocol_for(cfg,horizon)
                folder = run / f'horizon_{horizon}'
                folder.mkdir()
                stats = MinMax([v['name'] for v in layers]+['target'])
                train_values,train_positions,train_years,train_counts = collect(cfg,layers,profile,zones,
                    protocol['training_target_years'],horizon,1,cfg['pilot']['samples_per_training_year'],stats)
                parameters = stats.parameters()
                validation_values,validation_positions,validation_years,validation_counts = collect(cfg,layers,profile,zones,
                    protocol['validation_target_years'],horizon,protocol['validation_zone'],cfg['pilot']['samples_per_validation_year'])
                scaled = scale_values(train_values,parameters)
                val_scaled = scale_values(validation_values,parameters)
                save_json(folder/'normalization.json',dict(inputs=parameters[:-1],target=parameters[-1],
                    scope='all valid zone-1 training pixels across training target years; before sampling',
                    clipping=False,zero_is_valid=True))
                save_json(folder/'sampling.json',dict(training=train_counts,validation=validation_counts,
                    method='uniform without replacement per year via random priorities; equal per-year cap',
                    validation_scope=protocol['validation_mode']+'; sampled metrics, not a full-map evaluation',
                    reserved_zone=3,protocol=protocol))
                np.savez_compressed(folder/'samples.npz',train=train_values,train_positions=train_positions,train_years=train_years,
                    validation=validation_values,validation_positions=validation_positions,validation_years=validation_years)
                x,y = scaled[:,:-1],scaled[:,-1]
                vx,vy = val_scaled[:,:-1],validation_values[:,-1].astype(np.float64)
                baseline = _score(vy,np.full(len(vy),float(train_values[:,-1].mean(dtype=np.float64))))
                zero_baseline = _score(vy,np.zeros(len(vy)))
                save_json(folder/'baselines.json',dict(training_sample_mean=baseline,constant_zero=zero_baseline))
                for model_name, options in cfg['pilot']['models'].items():
                    logging.info('Training horizon=%s model=%s on %s samples',horizon,model_name,len(x))
                    start = time.monotonic()
                    model = make_model(model_name,cfg['pilot']['seed'],options,'cpu')
                    model.fit(x,y)
                    prediction = original_target(model.predict(vx),parameters[-1])
                    metrics = _score(vy,prediction)
                    positive = vy > 0
                    absent = vy == 0
                    diagnostics = dict(positive_target=_score(vy[positive],prediction[positive]) if positive.any() else None,
                        zero_target=_score(vy[absent],prediction[absent]) if absent.any() else None,
                        prediction_min=float(prediction.min()),prediction_max=float(prediction.max()),
                        negative_predictions=int(np.count_nonzero(prediction<0)),
                        validation_values_outside_training_range=[dict(name=p['name'],count=int(np.count_nonzero((val_scaled[:,i]<0)|(val_scaled[:,i]>1)))) for i,p in enumerate(parameters)])
                    model_dir = folder / model_name
                    model_dir.mkdir()
                    artifact = dict(format='raster_fusion_temporal_v1',model=model,layers=layers,
                        inputs_normalization=parameters[:-1],target_normalization=parameters[-1],
                        horizon_years=horizon,grid=_grid(profile),training_target_years=protocol['training_target_years'],
                        validation_protocol=protocol,
                        original_inputs=[str(p) for p in sorted(all_paths)],cpu_threads=cfg['pilot']['cpu_threads'])
                    joblib.dump(artifact,model_dir/'model.joblib')
                    # Validate serialization and inverse scaling on an independent reload.
                    restored = joblib.load(model_dir/'model.joblib')
                    np.testing.assert_allclose(original_target(restored['model'].predict(vx[:128]),restored['target_normalization']),prediction[:128])
                    save_json(model_dir/'metrics.json',dict(validation_original_units=metrics,diagnostics=diagnostics,
                        normalized_validation=_score(val_scaled[:,-1],model.predict(vx)),
                        baseline_training_mean=baseline,baseline_zero=zero_baseline,
                        seconds=time.monotonic()-start))
                    np.save(model_dir/'validation_predictions.npy',prediction)
                    record = dict(horizon=horizon,model=model_name,train_samples=len(x),validation_samples=len(vy),
                        rmse=metrics['rmse'],mae=metrics['mae'],r2=metrics['r2'],baseline_mean_rmse=baseline['rmse'],
                        baseline_zero_rmse=zero_baseline['rmse'],seconds=time.monotonic()-start,checkpoint=str(model_dir/'model.joblib'))
                    records.append(record)
                    logging.info('Completed horizon=%s %s: validation RMSE %.6f, MAE %.6f, R2 %s',horizon,model_name,metrics['rmse'],metrics['mae'],metrics['r2'])
                    _write_summary(run,records,cfg)
                    if cfg.get('figures',{}).get('enabled',True):
                        from src.evaluation.temporal_figures import export_model
                        logging.info('Exporting high-resolution figures for horizon=%s model=%s',horizon,model_name)
                        export_model(model_dir,cfg.get('figures',{}).get('dpi',600))
                best = min((r for r in records if r['horizon']==horizon),key=lambda r:r['rmse'])
                save_json(folder/'selected_model.json',dict(selection='minimum sampled validation RMSE; preliminary',**best))
        if cfg.get('figures',{}).get('enabled',True):
            from src.evaluation.temporal_figures import export_comparison
            export_comparison(run,cfg.get('figures',{}).get('dpi',600))
        meta.update(status='completed',completed=datetime.now(timezone.utc).isoformat(),trained_models=len(records))
        save_json(run/'run.json',meta)
        logging.info('Pilot completed: %s models. External test remains untouched.',len(records))
        return run
    except BaseException as exc:
        meta.update(status='cancelled' if isinstance(exc,KeyboardInterrupt) else 'failed',error=str(exc))
        save_json(run/'run.json',meta)
        logging.exception('Pilot did not finish')
        raise
    finally:
        log.removeHandler(handler)
        log.removeHandler(stream)
        handler.close()


def _write_summary(run, records, cfg):
    with (run/'comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    lines=['# Piloto temporal com dados reais','',
        'Modelos tabulares por pixel. Métricas em unidades originais do target, sobre amostras uniformes da validação.',
        'Treino usa somente zona espacial 1. No mesmo ano, validação usa zona 2 dos mesmos anos, sem compartilhar locais com o treino. No horizonte futuro, validação usa zona 1 de anos posteriores, para avaliar transferência temporal nos mesmos locais. Zona 3 e região externa permanecem reservadas.',
        'Normalização Min–Max de cada entrada e do target ajustada em todos os pixels válidos de treino, antes da amostragem; zeros incluídos. Sem truncamento.',
        'Grade comum definida por um target de treino; entradas reamostradas conforme manifesto e targets por '+cfg['data']['target_resampling']+'. Originais preservados.', '',
        '| Horizonte | Modelo | Amostras treino | Amostras validação | RMSE | MAE | R² | RMSE previsão zero |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in records:
        r2='indefinido' if r['r2'] is None else f"{r['r2']:.6f}"
        lines.append(f"| {r['horizon']} ano(s) | {r['model']} | {r['train_samples']} | {r['validation_samples']} | {r['rmse']:.6f} | {r['mae']:.6f} | {r2} | {r['baseline_zero_rmse']:.6f} |")
    lines += ['', 'Esta rodada usa uma seed, amostragem e dois modelos iniciais. Não é a comparação completa com contexto espacial/CNN nem um teste externo final.',
        'O bloco e o buffer são parâmetros iniciais; não demonstram independência espacial em fenômenos com dependência de maior alcance.',
        'Os pares de cada horizonte e a cobertura amostrada estão em sampling.json. As coordenadas das amostras estão em samples.npz.',
        'Os modelos produzem previsões em escala original por predict_temporal.py. Previsões negativas ou acima da faixa de treino são preservadas e contabilizadas, sem clipping silencioso.']
    for horizon in cfg['temporal']['horizons']:
        protocol=protocol_for(cfg,horizon)
        lines += ['',f"Horizonte {horizon}: targets de treino {protocol['training_target_years']}, targets de validação {protocol['validation_target_years']}, validação {protocol['validation_mode']}, zona {protocol['validation_zone']}."]
    (run/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def predict(model_path, input_dir, output, input_year, reference=None):
    model_path,input_dir,output = model_path.resolve(),input_dir.resolve(),output.resolve()
    artifact = joblib.load(model_path)
    if artifact.get('format') not in ('raster_fusion_temporal_v1', 'raster_fusion_temporal_spatial_v1'):
        raise ValueError('Expected a temporal checkpoint')
    layers = artifact['layers']
    expected={v['filename'] for v in layers}
    actual={p.relative_to(input_dir).as_posix() for p in input_dir.rglob('*') if p.suffix.lower() in ('.tif','.tiff')}
    if actual != expected:
        raise ValueError(f'Input schema mismatch: missing={expected-actual}, extra={actual-expected}')
    paths = [input_dir/v['filename'] for v in layers]
    reference = (reference or paths[0]).resolve()
    protected = {p.resolve() for p in paths+[reference,model_path]}
    protected.update(Path(p).resolve() for p in artifact['original_inputs'])
    if output.suffix.lower() not in ('.tif','.tiff') or output in protected:
        raise ValueError('Output must be a new GeoTIFF and must not overwrite an input')
    if output.exists() or output.with_suffix('.json').exists():
        raise FileExistsError(output)
    pending = output.with_name(output.stem+'.pending.tif')
    if pending.exists() or pending in protected:
        raise FileExistsError(pending)
    with rasterio.open(reference) as ds:
        check_dataset(ds)
        profile=ds.profile.copy()
    trained=artifact['grid']
    if profile['crs'] != rasterio.crs.CRS.from_wkt(trained['crs']) or not np.allclose(
            [abs(profile['transform'].a),abs(profile['transform'].e)], [abs(trained['transform'][0]),abs(trained['transform'][4])],rtol=1e-6):
        raise ValueError('Inference CRS/resolution must match training; extent and dimensions may differ')
    out_profile={k:profile[k] for k in ('crs','transform','width','height')}
    out_profile.update(driver='GTiff',count=1,dtype='float32',nodata=float('nan'),compress='deflate',tiled=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    valid_count=0
    progress_started=time.monotonic()
    if artifact['format'] == 'raster_fusion_temporal_spatial_v1':
        from src.geospatial.temporal_context import spatial_windows, feature_names
        spec = artifact['spatial_features']
        if spec.get('version') != 1 or spec.get('statistics') != ['mean', 'std'] or spec.get('missing') != 'finite_neighbors':
            raise ValueError('Unsupported spatial feature specification')
        names = feature_names(layers, spec['sizes'])
        if names != [p['name'] for p in artifact['inputs_normalization']]:
            raise ValueError('Spatial feature schema differs from saved normalization')
        windows = spatial_windows(paths, [v['resampling'] for v in layers], profile, 512, spec['sizes'])
    else:
        windows = aligned_windows(paths, [v['resampling'] for v in layers], profile, 512)
    try:
        with rasterio.open(pending,'w',**out_profile) as writer, threadpool_limits(limits=artifact['cpu_threads']):
            for window,values in windows:
                valid=np.isfinite(values).all(axis=-1)
                result=np.full(valid.shape,np.nan,dtype=np.float32)
                if valid.any():
                    vectors=scale_values(values[valid],artifact['inputs_normalization'])
                    pred=original_target(artifact['model'].predict(vectors),artifact['target_normalization'])
                    if not np.isfinite(pred).all():
                        raise ValueError('Non-finite prediction')
                    result[valid]=pred
                    valid_count+=len(pred)
                writer.write(result,1,window=window)
                if time.monotonic()-progress_started >= 20:
                    logging.info('Predicting horizon=%s: row %s/%s, %s valid pixels written',
                        artifact['horizon_years'], int(window.row_off+window.height), profile['height'], valid_count)
                    progress_started=time.monotonic()
        if not valid_count:
            raise ValueError('No overlapping valid input pixels')
        pending.rename(output)
        save_json(output.with_suffix('.json'),dict(model=str(model_path),input_year=input_year,
            predicted_target_year=input_year+artifact['horizon_years'],target_scale='original',
            horizon_years=artifact['horizon_years'],valid_pixels=valid_count,layers=layers,
            spatial_features=artifact.get('spatial_features'),
            normalization=dict(inputs=artifact['inputs_normalization'],target=artifact['target_normalization']),grid=_grid(profile)))
        return output
    except BaseException:
        # Remove only this call's partial output, never any source or completed file.
        if pending.exists():
            pending.unlink()
        raise
