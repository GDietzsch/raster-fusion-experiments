"""Auditable ordinary least-squares experiment on enriched temporal rasters.

The four phenomena are the units of relevance selection. All 24 possible
predictors define a common valid-pixel mask for every arm of an horizon.
"""
from __future__ import annotations

from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
import csv
import hashlib
import importlib.metadata
import json
import logging
import math
import platform
import shutil
import sys

import joblib
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import yaml

from src.geospatial.raster_io import check_dataset, write_raster
from src.services.temporal_service import UniformSample, aligned_windows
from src.splitting.spatial_split import split_grid


PHENOMENA = ('Garimpos', 'Ocorrencias_Minerais', 'Pistas_Regulares', 'Pistas_Irregulares')
PRODUCTS = ('original', 'magnitude', 'direcao_seno', 'direcao_cosseno', 'conectividade', 'suporte')
ARMS = ('originais', 'originais_filtradas', 'gpa', 'grafos', 'todas', 'todas_filtradas')
FEATURE_NAMES = tuple(f'{phenomenon}__{product}' for phenomenon in PHENOMENA for product in PRODUCTS)
REFERENCES = (
    ('LinearRegression, scikit-learn', 'https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html'),
    ('Valavi et al. (2019), blockCV, Methods in Ecology and Evolution', 'https://doi.org/10.1111/2041-210X.13107'),
    ('Choosing blocks for spatial cross-validation (2025), Frontiers in Remote Sensing', 'https://doi.org/10.3389/frsen.2025.1531097'),
    ('Permutation importance with correlated features, scikit-learn', 'https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html'),
    ('Median absolute deviation, SciPy', 'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.median_abs_deviation.html'),
    ('CE90 as positional accuracy, USGS (2024)', 'https://pubs.usgs.gov/publication/ofr20241023/full'),
)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def _sha256(path: Path) -> str:
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def _feature_path(root: Path, region: str, year: int, phenomenon: str, product: str) -> Path:
    if product == 'suporte':
        return root/region/str(year)/'auxiliares'/phenomenon/'suporte.tif'
    return root/region/str(year)/'inputs'/phenomenon/f'{product}.tif'


def _paths(root: Path, region: str, input_year: int) -> list[Path]:
    return [_feature_path(root, region, input_year, phenomenon, product)
            for phenomenon in PHENOMENA for product in PRODUCTS]


def _target_path(cfg: dict, region: str, target_year: int) -> Path:
    if region == 'treinamento':
        return cfg['root']/region/str(target_year)/'targets'/'target.tif'
    return Path(cfg['manifest']['config']['evaluation_targets'][f'{region}/{target_year}'])


def _methods(cfg: dict) -> list[str]:
    return ['nearest' if product == 'suporte' else 'bilinear'
            for _ in PHENOMENA for product in PRODUCTS] + [cfg['raw']['evaluation']['target_resampling']]


def _profile(path: Path) -> dict:
    with rasterio.open(path) as ds:
        check_dataset(ds)
        if ds.nodata == 0:
            raise ValueError(f'Zero was declared NoData: {path}')
        return ds.profile.copy()


def load_config(path: Path) -> dict:
    path = path.resolve()
    raw = yaml.safe_load(path.read_text(encoding='utf-8'))
    root = (path.parent/raw['data_root']).resolve()
    output = (path.parent/raw['output_root']).resolve()
    manifest_path = root/'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('status') != 'complete':
        raise ValueError('Enriched data manifest is not complete')
    if raw['evaluation']['primary_metric'] != 'rmse' or raw['evaluation']['tie_breakers'] != ['mae', 'fewer_features']:
        raise ValueError('Winner rule must be RMSE, then MAE, then fewer features')
    if raw['evaluation']['target_resampling'] != 'nearest':
        raise ValueError('Targets must be aligned with nearest-neighbor resampling')
    if raw['evaluation']['final_h0_target_years'] != [2020, 2021, 2022] or raw['evaluation']['final_h1_target_years'] != [2022]:
        raise ValueError('The external years changed; revise the experiment protocol')
    if raw['sampling']['per_stage_year'] < 1 or raw['sampling']['window_size'] < 1:
        raise ValueError('Sampling parameters must be positive')
    if not 0 < raw['evaluation']['high_target_positive_quantile'] < 1:
        raise ValueError('High-target quantile must be between zero and one')
    if raw['runtime']['cpu_threads'] < 1:
        raise ValueError('cpu_threads must be positive')
    if (root/raw['reference_target']).resolve() != _target_path({'root':root,'manifest':manifest}, 'treinamento', 2020):
        raise ValueError('Reference must be the 2020 development target')
    return dict(raw=raw, path=path, root=root, output=output, manifest=manifest,
                reference=(root/raw['reference_target']).resolve())


def audit_catalog(cfg: dict, verify_hashes: bool = False, on_verified=None) -> dict:
    """Inspect external inputs, but never read an external target in preflight."""
    root, registered = cfg['root'], cfg['manifest']['files']
    reference = _profile(cfg['reference'])
    inspected = []
    for region in ('treinamento', 'teste_final'):
        for year in (2019, 2020, 2021, 2022):
            phenomenon_profiles = {phenomenon:_profile(_feature_path(root,region,year,phenomenon,'original'))
                                   for phenomenon in PHENOMENA}
            for name, path in zip(FEATURE_NAMES, _paths(root, region, year)):
                relative = path.relative_to(root).as_posix()
                if relative not in registered or not path.is_file():
                    raise FileNotFoundError(f'Missing manifest entry or input: {path}')
                item = registered[relative]
                if path.stat().st_size != item['bytes']:
                    raise ValueError(f'Input size does not match manifest: {path}')
                with rasterio.open(path) as ds:
                    check_dataset(ds)
                    if ds.nodata == 0:
                        raise ValueError(f'Zero is valid and cannot be NoData: {path}')
                    phenomenon = name.split('__',1)[0]
                    pair_profile = phenomenon_profiles[phenomenon]
                    if ds.crs != pair_profile['crs'] or ds.width != pair_profile['width'] or ds.height != pair_profile['height'] or not ds.transform.almost_equals(pair_profile['transform']):
                        raise ValueError(f'Input and its derived grid differ within region/year: {path}')
                    if ds.crs != reference['crs'] or not np.allclose(ds.res, reference['transform'].a, rtol=1e-6):
                        raise ValueError(f'Input CRS or resolution differs from development reference: {path}')
                if verify_hashes:
                    if _sha256(path) != item['sha256']:
                        raise ValueError(f'Input hash does not match manifest: {path}')
                    if on_verified is not None:
                        on_verified(len(inspected) + 1, relative)
                inspected.append(dict(path=relative, name=name, bytes=item['bytes'], sha256=item['sha256']))
    for year in (2020, 2021, 2022):
        target = _target_path(cfg, 'treinamento', year)
        rel = target.relative_to(root).as_posix()
        if not target.is_file() or rel not in registered:
            raise FileNotFoundError(target)
        with rasterio.open(target) as ds:
            check_dataset(ds)
            if ds.nodata == 0 or ds.crs != reference['crs'] or not np.allclose(ds.res, reference['transform'].a, rtol=1e-6):
                raise ValueError(f'Unexpected development target grid or NoData: {target}')
        if verify_hashes:
            if _sha256(target) != registered[rel]['sha256']:
                raise ValueError(f'Target hash does not match manifest: {target}')
            if on_verified is not None:
                on_verified(len(inspected) + 1, rel)
        inspected.append(dict(path=rel, name='target', bytes=registered[rel]['bytes'], sha256=registered[rel]['sha256']))
    for year in (2020, 2021, 2022):
        if not _target_path(cfg, 'teste_final', year).is_file():
            raise FileNotFoundError(f'External evaluation target missing for {year}')
    return dict(input_and_development_target_files=len(inspected), recorded_files=inspected,
                hashes_recomputed=verify_hashes, external_targets_opened=False,
                reference=dict(crs=reference['crs'].to_wkt(), width=reference['width'], height=reference['height'],
                               transform=list(reference['transform'])[:6]))


def check_only(config_path: Path):
    cfg = load_config(config_path)
    audit = audit_catalog(cfg, verify_hashes=False)
    print(f"PRECHECK OK: {audit['input_and_development_target_files']} inputs/targets, 24 channels, H0 and H1. External targets not opened.")


def _stage(horizon: int, target_year: int, zone: int) -> str | None:
    if horizon == 0:
        return {1:'ajuste', 2:'selecao_camadas', 3:'validacao_modelo'}.get(zone)
    if zone == 1:
        return {2020:'ajuste', 2021:'selecao_camadas', 2022:'validacao_modelo'}.get(target_year)
    return None


def collect_development(cfg: dict, profile: dict, zones: np.ndarray, horizon: int, destination: Path) -> dict:
    """One raster pass per target year; sample all stages on one common mask."""
    result = {'ajuste':[], 'selecao_camadas':[], 'validacao_modelo':[]}
    counts = []
    cap = cfg['raw']['sampling']['per_stage_year']
    tile = cfg['raw']['sampling']['window_size']
    for target_year in (2020, 2021, 2022):
        input_year = target_year-horizon
        paths = _paths(cfg['root'], 'treinamento', input_year)+[_target_path(cfg,'treinamento',target_year)]
        samplers = {stage:UniformSample(cap, cfg['raw']['sampling']['seed']+horizon*10000+target_year*10+zone)
                    for zone in (1,2,3) if (stage := _stage(horizon,target_year,zone))}
        eligible = {stage:0 for stage in samplers}
        logging.info('H%s: development input %s -> target %s; %s stages', horizon,input_year,target_year,list(samplers))
        for window, values in aligned_windows(paths, _methods(cfg), profile, tile):
            finite = np.isfinite(values).all(axis=-1)
            zone_tile = zones[window.toslices()]
            for zone in (1,2,3):
                stage = _stage(horizon,target_year,zone)
                if stage is None:
                    continue
                valid = finite & (zone_tile == zone)
                if not np.any(valid):
                    continue
                rows, cols = np.where(valid)
                eligible[stage] += len(rows)
                positions = np.column_stack((rows+int(window.row_off), cols+int(window.col_off))).astype(np.int32)
                samplers[stage].add(values[valid], positions)
        for stage,sampler in samplers.items():
            if sampler.values is None:
                raise ValueError(f'No common valid pixels for H{horizon}, {stage}, {target_year}')
            result[stage].append((target_year,sampler.values,sampler.positions))
            counts.append(dict(horizon=horizon,stage=stage,input_year=input_year,target_year=target_year,
                               eligible=eligible[stage],sampled=len(sampler.values),
                               zero_target=int(np.count_nonzero(sampler.values[:,-1] == 0))))
            logging.info('H%s %s %s: eligible=%s sampled=%s',horizon,stage,target_year,eligible[stage],len(sampler.values))
    combined = {}
    for stage,parts in result.items():
        if not parts:
            raise ValueError(f'Empty stage: H{horizon} {stage}')
        values = np.concatenate([part[1] for part in parts]).astype(np.float32)
        positions = np.concatenate([part[2] for part in parts])
        years = np.concatenate([np.full(len(part[1]),part[0],dtype=np.int16) for part in parts])
        combined[stage] = (values[:,:-1],values[:,-1].astype(np.float64),positions,years)
    np.savez_compressed(destination, **{f'{stage}_{part}':array for stage,arrays in combined.items()
                                         for part,array in zip(('x','y','positions','target_year'),arrays)})
    return dict(stages=combined, counts=counts)


def _indices(products: tuple[str, ...], phenomena: tuple[str, ...] = PHENOMENA) -> tuple[int, ...]:
    return tuple(FEATURE_NAMES.index(f'{phenomenon}__{product}')
                 for phenomenon in phenomena for product in products)


def _arm_indices(name: str, phenomena: tuple[str, ...] = PHENOMENA) -> tuple[int, ...]:
    if name.startswith('originais'):
        return _indices(('original',),phenomena)
    if name == 'gpa':
        return _indices(('magnitude','direcao_seno','direcao_cosseno'))
    if name == 'grafos':
        return _indices(('conectividade','suporte'))
    if name.startswith('todas'):
        return _indices(PRODUCTS,phenomena)
    raise ValueError(name)


def _fit(x: np.ndarray, y: np.ndarray, columns: tuple[int, ...]):
    model = make_pipeline(StandardScaler(),LinearRegression())
    model.fit(x[:,columns],y)
    return model


def _predict(model, x: np.ndarray, columns: tuple[int, ...]) -> np.ndarray:
    prediction = np.asarray(model.predict(x[:,columns]),dtype=np.float64)
    if not np.isfinite(prediction).all():
        raise ValueError('Non-finite prediction from linear regression')
    return prediction


def _subsets():
    for size in range(1,len(PHENOMENA)+1):
        yield from combinations(PHENOMENA,size)


def _rmse(y, p) -> float:
    return float(np.sqrt(np.mean(np.square(p-y),dtype=np.float64)))


def _mae(y, p) -> float:
    return float(np.mean(np.abs(p-y),dtype=np.float64))


def select_phenomena(arm: str, x_fit: np.ndarray, y_fit: np.ndarray,
                     x_select: np.ndarray, y_select: np.ndarray,
                     fit_model=_fit) -> tuple[tuple[str, ...], list[dict]]:
    """All 15 nonempty phenomenon subsets, chosen on the feature-selection stage."""
    records = []
    for subset in _subsets():
        columns = _arm_indices(arm,subset)
        model = fit_model(x_fit,y_fit,columns)
        pred = _predict(model,x_select,columns)
        records.append(dict(arm=arm,phenomena=list(subset),features=[FEATURE_NAMES[i] for i in columns],
                            n_features=len(columns),selection_rmse=_rmse(y_select,pred),
                            selection_mae=_mae(y_select,pred)))
    winner = min(records,key=lambda row:(row['selection_rmse'],row['selection_mae'],row['n_features'],row['phenomena']))
    return tuple(winner['phenomena']),records


def _product_groups() -> dict[str,tuple[int,...]]:
    products = (('original',),('magnitude',),('direcao_seno','direcao_cosseno'),
                ('conectividade','suporte'))
    return {f'{phenomenon}__{group[0]}':_indices(group,(phenomenon,))
            for phenomenon in PHENOMENA for group in products}


def select_all_features(x_fit: np.ndarray, y_fit: np.ndarray,
                        x_select: np.ndarray, y_select: np.ndarray,
                        fit_model=_fit) -> tuple[tuple[int,...],list[dict]]:
    """Backward selection among 16 interpretable product groups on the selection stage.

    Direction components and graph support/connectivity remain paired. Every
    candidate is fitted only on the initial-fit stage. The later model-selection
    stage is never read here.
    """
    groups = _product_groups()
    active = list(groups)
    records = []

    def evaluate(names: list[str], step: int, removed: str | None):
        columns = tuple(i for name in names for i in groups[name])
        model = fit_model(x_fit,y_fit,columns)
        p = _predict(model,x_select,columns)
        row = dict(arm='todas_filtradas',step=step,removed_group=removed,active_groups=list(names),
                   phenomena=[phenomenon for phenomenon in PHENOMENA
                              if any(name.startswith(f'{phenomenon}__') for name in names)],
                   features=[FEATURE_NAMES[i] for i in columns],n_features=len(columns),
                   selection_rmse=_rmse(y_select,p),selection_mae=_mae(y_select,p),accepted=False)
        records.append(row)
        return row

    current = evaluate(active,0,None)
    current['accepted'] = True
    for step in range(1,len(groups)):
        choices = [evaluate([name for name in active if name != removed],step,removed)
                   for removed in active]
        best = min(choices,key=lambda row:(row['selection_rmse'],row['selection_mae'],
                                           row['n_features'],row['removed_group']))
        if best['selection_rmse'] > current['selection_rmse'] + 1e-12:
            break
        best['accepted'] = True
        active = best['active_groups']
        current = best
    return tuple(FEATURE_NAMES.index(name) for name in current['features']),records


def _capture(y: np.ndarray, p: np.ndarray, fraction: float) -> float | None:
    mass = float(np.sum(y,dtype=np.float64))
    if mass <= 0 or np.any(y < 0):
        return None
    count = max(1,math.ceil(len(y)*fraction))
    # Pixel order is the documented deterministic tie-breaker.
    rank = np.lexsort((np.arange(len(p),dtype=np.int64),-p))[:count]
    return float(np.sum(y[rank],dtype=np.float64)/mass)


def metrics(y: np.ndarray, p: np.ndarray, high_threshold: float | None = None,
            subsets: bool = True) -> dict:
    y = np.asarray(y,dtype=np.float64)
    p = np.asarray(p,dtype=np.float64)
    if y.ndim != 1 or p.shape != y.shape or not len(y) or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('Metrics require same-size nonempty finite one-dimensional arrays')
    error = p-y
    center = float(np.median(error))
    total = float(np.sum(np.square(y-y.mean()),dtype=np.float64))
    values = dict(n=int(len(y)),rmse=_rmse(y,p),mae=_mae(y,p),
                  r2=None if total <= 0 else float(1-np.sum(np.square(error),dtype=np.float64)/total),
                  bias=float(np.mean(error)),median_error=center,
                  p90_absolute_error=float(np.quantile(np.abs(error),0.90)),
                  nmad=float(1.482602218505602*np.median(np.abs(error-center))),
                  negative_predictions=int(np.count_nonzero(p<0)),
                  target_zero=int(np.count_nonzero(y==0)),
                  capture_5pct=_capture(y,p,0.05),capture_10pct=_capture(y,p,0.10),
                  capture_20pct=_capture(y,p,0.20))
    if subsets:
        values['subsets'] = {}
        for name,mask in (('zero',y==0),('positive',y>0),
                          ('high',y>=high_threshold if high_threshold is not None else np.zeros(len(y),dtype=bool))):
            values['subsets'][name] = metrics(y[mask],p[mask],subsets=False) if np.any(mask) else None
    return values


def _training_arm(arm: str, stage: dict, high_threshold: float, directory: Path,
                  grid_width: int, seed: int) -> dict:
    x0,y0 = stage['ajuste'][:2]
    x1,y1 = stage['selecao_camadas'][:2]
    xv,yv = stage['validacao_modelo'][:2]
    chosen = PHENOMENA
    selection_rows = []
    if arm == 'originais_filtradas':
        chosen,selection_rows = select_phenomena(arm,x0,y0,x1,y1)
        columns = _arm_indices(arm,chosen)
    elif arm == 'todas_filtradas':
        columns,selection_rows = select_all_features(x0,y0,x1,y1)
        chosen = tuple(phenomenon for phenomenon in PHENOMENA
                       if any(FEATURE_NAMES[i].startswith(f'{phenomenon}__') for i in columns))
    else:
        columns = _arm_indices(arm,chosen)
    initial = _fit(x0,y0,columns)
    initial_metrics = metrics(y1,_predict(initial,x1,columns),high_threshold)
    combined_x = np.concatenate((x0,x1))
    combined_y = np.concatenate((y0,y1))
    model = _fit(combined_x,combined_y,columns)
    regression = model.named_steps['linearregression']
    singular = np.asarray(regression.singular_,dtype=np.float64)
    condition = (float(singular[0]/singular[-1]) if len(singular) and singular[-1] > 0 else None)
    if condition is not None and not math.isfinite(condition):
        condition = None
    training_metrics = metrics(combined_y,_predict(model,combined_x,columns),high_threshold)
    validation_prediction = _predict(model,xv,columns)
    validation_metrics = metrics(yv,validation_prediction,high_threshold)
    validation_uncertainty = _bootstrap_block_errors(yv,validation_prediction,
                                                     stage['validacao_modelo'][2][:,0],
                                                     stage['validacao_modelo'][2][:,1],
                                                     grid_width,seed=seed)
    validation_by_year = []
    for target_year in sorted(set(stage['validacao_modelo'][3])):
        mask = stage['validacao_modelo'][3] == target_year
        validation_by_year.append(dict(target_year=int(target_year),
                                       metrics=metrics(yv[mask],validation_prediction[mask],high_threshold)))
    folder = directory/arm
    folder.mkdir()
    joblib.dump(dict(model=model,columns=columns,feature_names=[FEATURE_NAMES[i] for i in columns],
                     phenomena=chosen,arm=arm),folder/'model.joblib')
    record = dict(arm=arm,phenomena=list(chosen),n_features=len(columns),
                  feature_names=[FEATURE_NAMES[i] for i in columns],
                  linear_rank=int(regression.rank_),condition_number=condition,
                  initial_selection_metrics=initial_metrics,training_metrics=training_metrics,
                  validation_metrics=validation_metrics,validation_uncertainty=validation_uncertainty,
                  validation_by_year=validation_by_year,
                  selection_candidates=selection_rows,
                  training_samples=len(combined_y),validation_samples=len(yv))
    _write_json(folder/'metrics.json',record)
    return record


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open('w',newline='',encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _winner(records: list[dict]) -> dict:
    return min(records,key=lambda row:(row['validation_metrics']['rmse'],row['validation_metrics']['mae'],
                                       row['n_features'],ARMS.index(row['arm'])))


def train_horizon(cfg: dict, profile: dict, zones: np.ndarray, horizon: int, directory: Path) -> dict:
    directory.mkdir()
    collection = collect_development(cfg,profile,zones,horizon,directory/'samples.npz')
    _write_json(directory/'sampling.json',collection['counts'])
    stage = collection['stages']
    positives = np.concatenate((stage['ajuste'][1],stage['selecao_camadas'][1]))
    positives = positives[positives>0]
    if not len(positives):
        raise ValueError(f'H{horizon} has no positive development targets')
    cutoff = float(np.quantile(positives,cfg['raw']['evaluation']['high_target_positive_quantile']))
    training_mean = float(np.mean(np.concatenate((stage['ajuste'][1],stage['selecao_camadas'][1]))))
    reference_metrics = dict(zero=metrics(stage['validacao_modelo'][1],np.zeros_like(stage['validacao_modelo'][1]),subsets=False),
                             training_mean=metrics(stage['validacao_modelo'][1],
                                                   np.full_like(stage['validacao_modelo'][1],training_mean),subsets=False))
    _write_json(directory/'validation_references.json',reference_metrics)
    records = []
    for arm in ARMS:
        logging.info('H%s: fitting %s',horizon,arm)
        records.append(_training_arm(arm,stage,cutoff,directory,profile['width'],
                                     cfg['raw']['sampling']['seed']+horizon*100))
    winner = _winner(records)
    flat = [dict(horizon=horizon,arm=row['arm'],phenomena=';'.join(row['phenomena']),
                 n_features=row['n_features'],initial_selection_rmse=row['initial_selection_metrics']['rmse'],
                 training_rmse=row['training_metrics']['rmse'],validation_rmse=row['validation_metrics']['rmse'],
                 validation_mae=row['validation_metrics']['mae'],validation_r2=row['validation_metrics']['r2'],
                 validation_nmad=row['validation_metrics']['nmad'],
                 validation_rmse_ci95_low=row['validation_uncertainty']['rmse_ci95'][0],
                 validation_rmse_ci95_high=row['validation_uncertainty']['rmse_ci95'][1],
                 validation_capture_10pct=row['validation_metrics']['capture_10pct']) for row in records]
    _write_csv(directory/'comparison.csv',flat)
    _write_csv(directory/'selection_candidates.csv',[
        dict(arm=row['arm'],step=row.get('step'),removed_group=row.get('removed_group'),
             accepted=row.get('accepted'),phenomena=';'.join(row['phenomena']),
             features=';'.join(row['features']),n_features=row['n_features'],
             selection_rmse=row['selection_rmse'],selection_mae=row['selection_mae'])
        for record in records for row in record['selection_candidates']])
    selected = dict(horizon=horizon,arm=winner['arm'],phenomena=winner['phenomena'],
                    feature_names=winner['feature_names'],n_features=winner['n_features'],
                    validation_metrics=winner['validation_metrics'],high_target_threshold=cutoff,
                    training_target_mean=training_mean,
                    model_path=str(directory/winner['arm']/'model.joblib'))
    _write_json(directory/'winner.json',selected)
    logging.info('H%s winner: %s, validation RMSE %.6f',horizon,winner['arm'],winner['validation_metrics']['rmse'])
    return dict(winner=selected,records=records,sampling=collection['counts'],references=reference_metrics)


def _external_profile(cfg: dict, year: int) -> dict:
    return _profile(_feature_path(cfg['root'],'teste_final',year,PHENOMENA[0],'original'))


def _bootstrap_block_errors(y: np.ndarray, p: np.ndarray, row: np.ndarray, col: np.ndarray,
                            width: int, block_pixels: int = 384, repeats: int = 1000,
                            seed: int = 4201) -> dict:
    """Paired spatial-block bootstrap of error; blocks are a sensitivity scale."""
    grid_width = math.ceil(width/block_pixels)
    block = (row//block_pixels)*grid_width + (col//block_pixels)
    _, inverse = np.unique(block,return_inverse=True)
    n = np.bincount(inverse).astype(np.float64)
    squared = np.bincount(inverse,weights=np.square(p-y)).astype(np.float64)
    absolute = np.bincount(inverse,weights=np.abs(p-y)).astype(np.float64)
    rng = np.random.default_rng(seed)
    rmse,mae = np.empty(repeats),np.empty(repeats)
    for i in range(repeats):
        selected = rng.integers(0,len(n),size=len(n))
        denominator = n[selected].sum()
        rmse[i] = math.sqrt(squared[selected].sum()/denominator)
        mae[i] = absolute[selected].sum()/denominator
    return dict(block_pixels=block_pixels,approximate_block_km=block_pixels*0.25,
                occupied_blocks=len(n),replicates=repeats,seed=seed,
                rmse_ci95=[float(v) for v in np.quantile(rmse,[0.025,0.975])],
                mae_ci95=[float(v) for v in np.quantile(mae,[0.025,0.975])],
                caveat='Exploratory spatial-block interval; one region and one partition do not establish independent replication.')


def _persistence(cfg: dict, year: int, profile: dict, positions: tuple[np.ndarray,np.ndarray],
                 y: np.ndarray) -> dict | None:
    if year <= 2020:
        return None
    path = _target_path(cfg,'teste_final',year-1)
    rows,cols = positions
    with rasterio.open(path) as ds:
        if ds.nodata == 0:
            raise ValueError(f'External historical target declares zero as NoData: {path}')
        with WarpedVRT(ds,crs=profile['crs'],transform=profile['transform'],
                       width=profile['width'],height=profile['height'],dtype='float32',
                       nodata=float('nan'),resampling=Resampling.nearest) as vrt:
            historical = vrt.read(1,masked=True).filled(np.nan)
            prediction = historical[rows,cols].astype(np.float64)
    finite = np.isfinite(prediction)
    if not np.any(finite):
        return None
    return dict(source_target_year=year-1,common_pixels=int(np.count_nonzero(finite)),
                different_support=bool(np.count_nonzero(finite)!=len(y)),
                metrics=metrics(y[finite],prediction[finite],subsets=False))


def evaluate_external(cfg: dict, selected: dict, horizon: int, directory: Path) -> list[dict]:
    """Open external targets only after both winners have been saved."""
    artifact = joblib.load(selected['model_path'])
    if artifact['arm'] != selected['arm'] or artifact['feature_names'] != selected['feature_names']:
        raise ValueError('Winner metadata and model artifact disagree')
    model,columns = artifact['model'],tuple(artifact['columns'])
    years = cfg['raw']['evaluation'][f'final_h{horizon}_target_years']
    records = []
    for target_year in years:
        input_year = target_year-horizon
        profile = _external_profile(cfg,input_year)
        target_path = _target_path(cfg,'teste_final',target_year)
        target_digest = _sha256(target_path)
        paths = _paths(cfg['root'],'teste_final',input_year)+[target_path]
        out_path = directory/f'external_prediction_target_{target_year}.tif'
        raster_profile = profile.copy()
        raster_profile.update(driver='GTiff',count=1,dtype='float32',nodata=float('nan'),
                              tiled=True,blockxsize=256,blockysize=256,compress='deflate')
        target_parts,pred_parts,row_parts,col_parts = [],[],[],[]
        with rasterio.open(out_path,'w',**raster_profile) as output:
            for index,(window,values) in enumerate(aligned_windows(paths,_methods(cfg),profile,
                                                                     cfg['raw']['sampling']['window_size'])):
                valid = np.isfinite(values).all(axis=-1)
                predicted = np.full(valid.shape,np.nan,dtype=np.float32)
                if np.any(valid):
                    x = values[valid,:-1]
                    y = values[valid,-1].astype(np.float64)
                    p = _predict(model,x,columns)
                    predicted[valid] = p.astype(np.float32)
                    rr,cc = np.where(valid)
                    target_parts.append(y)
                    pred_parts.append(p)
                    row_parts.append(rr.astype(np.int32)+int(window.row_off))
                    col_parts.append(cc.astype(np.int32)+int(window.col_off))
                output.write(predicted,1,window=window)
                if index and index%100 == 0:
                    logging.info('External H%s target %s: %s tiles',horizon,target_year,index)
        if not target_parts:
            raise ValueError(f'No common valid external pixels for H{horizon} year {target_year}')
        y,p = np.concatenate(target_parts),np.concatenate(pred_parts)
        row,col = np.concatenate(row_parts),np.concatenate(col_parts)
        observed = metrics(y,p,selected['high_target_threshold'])
        zero = metrics(y,np.zeros_like(y),subsets=False)
        mean = metrics(y,np.full_like(y,selected['training_target_mean']),subsets=False)
        persistence = _persistence(cfg,target_year,profile,(row,col),y)
        uncertainty = _bootstrap_block_errors(y,p,row,col,profile['width'],
                                               seed=cfg['raw']['sampling']['seed']+horizon*10+target_year)
        record = dict(horizon=horizon,input_year=input_year,target_year=target_year,
                      prediction=str(out_path),target_source=str(target_path),
                      target_sha256=target_digest,common_pixels=len(y),
                      evaluated_fraction=float(len(y)/(profile['height']*profile['width'])),
                      metrics=observed,reference_zero=zero,reference_training_mean=mean,
                      reference_persistence=persistence,uncertainty=uncertainty)
        _write_json(directory/f'external_target_{target_year}_metrics.json',record)
        records.append(record)
        logging.info('External H%s target %s: RMSE %.6f MAE %.6f n=%s',
                     horizon,target_year,observed['rmse'],observed['mae'],len(y))
    return records


def _number(value, digits: int = 4) -> str:
    return 'indefinido' if value is None else f'{value:.{digits}f}'


def write_report(cfg: dict, run_dir: Path, horizons: dict[int,dict], external: dict[int,list[dict]],
                 audit: dict):
    lines = [
        '# Experimento enriquecido 01 — regressão linear simples', '',
        f'Execução: `{run_dir.name}`. Estado: concluído. Modelo: mínimos quadrados ordinários (`sklearn.linear_model.LinearRegression`), ajustado do zero.', '',
        '## Pergunta e desenho', '',
        'H0 estima o target do mesmo ano com as entradas desse ano. H1 estima o target do ano seguinte com as entradas do ano anterior. '
        'H0: zona 1 para ajuste inicial, zona 2 para seleção de camadas e novo ajuste, zona 3 para escolher o conjunto vencedor, '
        'nos targets de 2020–2022. H1: target 2020 para ajuste inicial, 2021 para seleção de camadas e novo ajuste, 2022 para escolher o vencedor; '
        'todos na zona espacial 1. O teste externo é feito somente após salvar ambos os vencedores. H0 usa targets externos de 2020–2022; H1 usa target externo de 2022.', '',
        'A separação espacial em blocos de 96 pixels (24 km), com buffer de 3 pixels, reduz a sobreposição imediata; não prova independência para fenômenos de alcance maior. '
        'A validação H1 é temporal nos mesmos locais de desenvolvimento; a avaliação externa verifica transferência para outra região.', '',
        '## Entradas e preparação', '',
        'Por fenômeno (garimpos, ocorrências minerais, pistas regulares, pistas irregulares): original, magnitude, seno e cosseno da direção, conectividade e suporte do grafo. '
        'Total: 24 canais. Graus brutos não entram porque o ângulo é circular. O suporte distingue ausência de nós próximos de conectividade nula de nós isolados. '
        'Os seis conjuntos usam uma máscara comum em que todos os 24 canais e o target são finitos. '
        'Os inputs contínuos são alinhados por interpolação bilinear, suporte e target por vizinho mais próximo. '
        'Os rasters de origem não são alterados. Cada canal selecionado recebe padronização ajustada apenas nos dados de ajuste. '
        'O target permanece em suas unidades originais; previsões negativas não são truncadas.', '',
        f"Arquivos auditados: {audit['input_and_development_target_files']}; tamanhos e hashes SHA-256 conferidos com o manifesto. "
        'Os targets externos não foram abertos durante a auditoria prévia.', '',
        '## Relevância e seleção', '',
        'Em `originais_filtradas`, todas as 15 combinações não vazias dos quatro fenômenos são treinadas no ajuste inicial e avaliadas na etapa de seleção de camadas. '
        'Em `todas_filtradas`, uma eliminação reversa avalia até 16 grupos fenômeno×produto: original; magnitude; par seno/cosseno; par conectividade/suporte. '
        'A cada passo, remove o grupo cuja retirada melhora mais o RMSE de seleção; para quando nenhuma retirada o melhora. '
        'Essa busca é gulosa, não exaustiva. Preservar os pares mantém a interpretação da direção circular e dos zeros do grafo; importância baixa de um derivado correlacionado não estabelece ausência de contribuição causal. '
        'Após a seleção, cada conjunto é ajustado novamente com a união das amostras de ajuste inicial e seleção de camadas. '
        'O posto da matriz linear e seu número de condição são registrados em cada `metrics.json` para diagnosticar colinearidade. '
        'O vencedor de cada horizonte é o conjunto com menor RMSE na etapa independente de validação; empates são resolvidos por MAE, depois menor número de atributos e, por fim, pela ordem declarada dos seis conjuntos. '
        'Amostragem uniforme sem reposição, até 100.000 pixels por ano e etapa, sobre a máscara comum; métricas internas são amostrais.', '',
        '## Métricas', '',
        'RMSE é o critério primário. MAE, R², viés, percentil 90 do erro absoluto, NMAD dos resíduos e captura da massa do target no top 5/10/20% são diagnósticos. '
        'NMAD = 1,4826022185 × mediana(|erro − mediana(erro)|). CE90 não foi usado porque mede erro posicional horizontal, não erro de intensidade. '
        'RMSE/MAE também são separados para target zero, positivo e intensidades acima do percentil 99 dos positivos de treino. '
        'Previsão constante zero e média do treino são referências; persistência externa usa target histórico adicional, por isso é identificada separadamente. '
        'Captura usa ranking decrescente da previsão, desempate pela ordem dos pixels válidos; não é probabilidade nem acurácia de classificação.', '',
    ]
    for horizon in (0,1):
        item = horizons[horizon]
        lines += [f'## H{horizon} — desenvolvimento', '',
                  f"Vencedor: **{item['winner']['arm']}**, com {item['winner']['n_features']} atributos. "
                  f"RMSE de validação {_number(item['winner']['validation_metrics']['rmse'])}. "
                  f"Fenômenos incluídos: {', '.join(item['winner']['phenomena'])}.", '',
                  '| Conjunto | Nº atributos | RMSE seleção | RMSE treino após novo ajuste | RMSE validação | MAE validação | R² validação | NMAD validação | Captura 10% validação |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for record in item['records']:
            m = record['validation_metrics']
            lines.append(f"| {record['arm']} | {record['n_features']} | {_number(record['initial_selection_metrics']['rmse'])} | "
                         f"{_number(record['training_metrics']['rmse'])} | {_number(m['rmse'])} | {_number(m['mae'])} | "
                         f"{_number(m['r2'])} | {_number(m['nmad'])} | {_number(m['capture_10pct'],3)} |")
        lines += ['',f"Referências na validação: zero, RMSE {_number(item['references']['zero']['rmse'])}; "
                  f"média do treino, RMSE {_number(item['references']['training_mean']['rmse'])}.", '',
                  'Arquivos detalhados: `samples.npz`, `sampling.json`, `selection_candidates.csv`, `comparison.csv`, '
                  '`validation_references.json`, `winner.json` e `metrics.json` de cada conjunto.', '']
        lines += [f'## H{horizon} — região externa', '',
                  '| Target | Pixels comuns | RMSE | MAE | R² | NMAD | P90 erro absoluto | Captura 10% | RMSE zero | RMSE média | RMSE persistência | IC95% RMSE blocos |',
                  '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
        for record in external[horizon]:
            m = record['metrics']
            persistence = record['reference_persistence']
            ci = record['uncertainty']['rmse_ci95']
            lines.append(f"| {record['target_year']} | {record['common_pixels']} | {_number(m['rmse'])} | {_number(m['mae'])} | "
                         f"{_number(m['r2'])} | {_number(m['nmad'])} | {_number(m['p90_absolute_error'])} | "
                         f"{_number(m['capture_10pct'],3)} | {_number(record['reference_zero']['rmse'])} | "
                         f"{_number(record['reference_training_mean']['rmse'])} | "
                         f"{_number(persistence['metrics']['rmse']) if persistence else '—'} | "
                         f"[{_number(ci[0])}, {_number(ci[1])}] |")
        lines += ['', 'Os intervalos são bootstrap pareado exploratório de 1.000 réplicas, com blocos de 384 pixels (~96 km); '
                  'dependem dessa escala e não representam novas regiões independentes. '
                  'A persistência é calculada na interseção válida com o target histórico, que pode ter cobertura diferente. '
                  'Os GeoTIFFs de previsão preservam a grade da região externa e NoData fora da máscara comum.', '']
    lines += ['## Procedência e limites', '',
              'A região externa não participou do ajuste destes modelos, mas seus targets foram analisados em investigações anteriores do projeto. '
              'Os resultados desta rodada são uma nova avaliação de transferência, não evidência confirmatória inteiramente inédita para a pesquisa. '
              'A divisão, os hiperparâmetros, o raio dos grafos e as camadas derivadas não foram otimizados aqui. '
              'Amostras de pixels e anos acumulados têm dependência espacial/temporal; não tratar sua contagem como número de réplicas independentes.', '',
              '## Referências metodológicas', '']
    lines += [f'- [{title}]({url})' for title,url in REFERENCES]
    lines += ['', 'Configuração, código, versões de bibliotecas, manifesto, hashes, máscaras, amostras, modelos, métricas e previsões ficam na pasta desta execução.', '']
    (run_dir/'RELATORIO.md').write_text('\n'.join(lines),encoding='utf-8')


def run(config_path: Path):
    cfg = load_config(config_path)
    cfg['output'].mkdir(parents=True,exist_ok=True)
    run_dir = cfg['output']/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run_dir.mkdir(exist_ok=False)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    file_handler = logging.FileHandler(run_dir/'training.log',encoding='utf-8')
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    status = dict(kind='enriched_linear',status='running',started=_utc(),run=str(run_dir),
                  final_external_test_used=False)
    _write_json(run_dir/'run.json',status)
    try:
        logging.info('Experiment run: %s',run_dir)
        audit = audit_catalog(cfg,verify_hashes=True)
        _write_json(run_dir/'input_audit.json',audit)
        shutil.copy2(cfg['path'],run_dir/'config.yaml')
        project = Path(__file__).parents[2]
        sources = ('run_enriched_linear.py','src/services/enriched_linear.py',
                   'src/services/temporal_service.py','src/splitting/spatial_split.py',
                   'src/geospatial/raster_io.py')
        source_records = []
        for relative in sources:
            original = project/relative
            destination = run_dir/'source_code'/relative
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(original,destination)
            source_records.append(dict(path=relative,sha256=_sha256(original)))
        _write_json(run_dir/'source_files.json',source_records)
        for relative in ('protocols/EXPERIMENTOS_ENRIQUECIDOS_2026.md','docs/ENRICHED_LINEAR.md'):
            original = project/relative
            destination = run_dir/relative
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(original,destination)
        versions = {}
        for name in ('numpy','scipy','scikit-learn','rasterio','joblib','PyYAML','threadpoolctl'):
            versions[name] = importlib.metadata.version(name)
        _write_json(run_dir/'environment.json',dict(python=sys.version,platform=platform.platform(),packages=versions))
        profile = _profile(cfg['reference'])
        zones = split_grid((profile['height'],profile['width']),cfg['raw']['split'])
        write_raster(run_dir/'spatial_zones.tif',zones,profile,categorical=True)
        _write_json(run_dir/'grid.json',audit['reference'])
        horizons = {}
        with threadpool_limits(limits=cfg['raw']['runtime']['cpu_threads']):
            for horizon in (0,1):
                horizons[horizon] = train_horizon(cfg,profile,zones,horizon,run_dir/f'horizon_{horizon}')
            status['selection_finished'] = _utc()
            _write_json(run_dir/'run.json',status)
            # Both winners are frozen before any external target is read.
            external = {}
            for horizon in (0,1):
                external[horizon] = evaluate_external(cfg,horizons[horizon]['winner'],horizon,
                                                      run_dir/f'horizon_{horizon}')
        write_report(cfg,run_dir,horizons,external,audit)
        status.update(status='completed',finished=_utc(),final_external_test_used=True,
                      winners={str(h):horizons[h]['winner']['arm'] for h in (0,1)})
        _write_json(run_dir/'run.json',status)
        logging.info('Completed. Report: %s',run_dir/'RELATORIO.md')
        print(run_dir/'RELATORIO.md')
    except BaseException as exc:
        status.update(status='failed',failed=_utc(),error=repr(exc))
        _write_json(run_dir/'run.json',status)
        logging.exception('Experiment failed; partial files retained at %s',run_dir)
        raise
    finally:
        logger.removeHandler(file_handler)
        logger.removeHandler(stream_handler)
        file_handler.close()
        stream_handler.close()
