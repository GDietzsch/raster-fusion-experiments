"""Experiment 10: dense U-Net regression on frozen enriched raster inputs."""
from __future__ import annotations

from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
import csv
import importlib.metadata
import json
import logging
import math
import platform
import shutil
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
import torch
import yaml

from src.geospatial.raster_io import write_raster
from src.image_pipeline.models.unet import UNet
from src.services import enriched_cnn as cnn
from src.services import enriched_linear as common
from src.splitting.spatial_split import split_grid


KIND = 'enriched_unet'
ARMS = common.ARMS
BASE_ARMS = ('originais', 'gpa', 'grafos', 'todas')
STAGES = cnn.STAGES
CORE = 512
HALO = 224
DEPTH = 5


def load_config(path: Path) -> dict:
    cfg = common.load_config(path)
    raw = cfg['raw']
    family = raw.get('model_family', 'unet')
    if family not in ('unet', 'resunet'):
        raise ValueError('Expected model_family unet or resunet')
    cfg['kind'] = 'enriched_' + family
    settings = raw.get('unet', {})
    source = (cfg['path'].parent / raw['sample_source']).resolve()
    cache_source = (cfg['path'].parent / raw['input_cache_source']).resolve()
    if any(cfg['output'] == protected or cfg['output'].is_relative_to(protected)
           or protected.is_relative_to(cfg['output'])
           for protected in (source, cache_source, cfg['root'])):
        raise ValueError('U-Net output must be separate from sources and data')
    source_status = json.loads((source / 'run.json').read_text(encoding='utf-8'))
    if source_status.get('kind') != 'enriched_mlp' or source_status.get('status') != 'completed':
        raise ValueError('Frozen sample source must be the completed MLP run')
    cache_status = json.loads((cache_source / 'run.json').read_text(encoding='utf-8'))
    if cache_status.get('kind') != 'enriched_cnn' or cache_status.get('status') != 'completed':
        raise ValueError('Input cache source must be the completed two-scale CNN')
    if Path(cache_status['sample_source']).resolve() != source:
        raise ValueError('Input cache and frozen samples differ')
    sample_config = yaml.safe_load((source / 'config.yaml').read_text(encoding='utf-8'))
    cache_config = yaml.safe_load((cache_source / 'config.yaml').read_text(encoding='utf-8'))
    for key in ('data_root', 'reference_target', 'split', 'sampling', 'evaluation'):
        if raw[key] != cache_config[key]:
            raise ValueError(f'U-Net differs from input cache in {key}')
    for key in ('data_root', 'split'):
        if raw[key] != sample_config[key]:
            raise ValueError(f'U-Net differs from frozen samples in {key}')
    if not json.loads((source / 'input_audit.json').read_text(encoding='utf-8'))['hashes_recomputed']:
        raise ValueError('Frozen sample source lacks verified raster hashes')
    integer_keys = ('core_size', 'halo_pixels', 'depth', 'training_tiles_per_year',
                    'selection_tiles_per_year', 'max_epochs', 'min_epochs',
                    'patience', 'random_state')
    if any(type(settings.get(key)) is not int or settings[key] < 1 for key in integer_keys):
        raise ValueError('All U-Net sizes, epoch limits and seeds must be positive integers')
    if (settings['core_size'], settings['halo_pixels'], settings['depth']) != (CORE, HALO, DEPTH):
        raise ValueError('This fixed protocol requires core=512, halo=224 and depth=5')
    if settings['min_epochs'] > settings['max_epochs']:
        raise ValueError('min_epochs exceeds max_epochs')
    if settings.get('device') not in ('auto', 'cpu', 'cuda'):
        raise ValueError('unet.device must be auto, cpu or cuda')
    candidates = settings.get('candidates')
    if not isinstance(candidates, list) or len(candidates) != 2:
        raise ValueError('This U-Net protocol compares two fixed candidates')
    if [item.get('initial_filters') for item in candidates] != [8, 16]:
        raise ValueError('Candidate initial filters must be 8 and 16 in that order')
    for item in candidates:
        expected_keys = {'initial_filters', 'learning_rate', 'weight_decay'}
        if family == 'resunet':
            expected_keys.add('residual_blocks')
        if set(item) != expected_keys:
            raise ValueError('Invalid U-Net candidate keys')
        if family == 'resunet' and item['residual_blocks'] is not True:
            raise ValueError('ResUNet candidates must enable residual_blocks')
        if any(not math.isfinite(float(item[key])) or item[key] < 0
               for key in ('learning_rate', 'weight_decay')) or item['learning_rate'] == 0:
            raise ValueError('Invalid U-Net learning rate or weight decay')
    cfg['sample_source'] = source
    cfg['cache_source'] = cache_source
    cfg['patch_source'] = cache_source  # enforce read-only aligned cache in cnn._aligned_cache
    cfg['_aligned_cache'] = cache_source / 'aligned_inputs'
    return cfg


def check_only(path: Path) -> None:
    cfg = load_config(path)
    audit = common.audit_catalog(cfg, verify_hashes=False)
    for horizon in (0, 1):
        sample = cfg['sample_source'] / f'horizon_{horizon}' / 'samples.npz'
        cnn._verify_reused_patch_bank(cfg, horizon, sample)
        with np.load(sample) as data:
            for stage in STAGES:
                if data[f'{stage}_x'].shape[1] != 24:
                    raise ValueError(f'H{horizon} {stage} lacks 24 channels')
    for region in ('treinamento', 'teste_final'):
        for year in (2020, 2021, 2022):
            stem = cfg['_aligned_cache'] / f'{region}_{year}'
            if not stem.with_suffix('.npy').is_file() or not stem.with_suffix('.json').is_file():
                raise FileNotFoundError(f'Missing aligned input cache: {stem}')
    device = cnn._device(cfg['raw']['unet'])
    print(f"PRECHECK OK: {audit['input_and_development_target_files']} inputs/targets; "
          f'24 channels; H0/H1; 512 core + 224 halo; 5 levels; '
          f'2 candidates per base arm; {device}; external targets unopened.')


def _model_config(filters: int, depth: int | None = None, residual: bool = False) -> dict:
    cfg = {'depth': DEPTH if depth is None else depth, 'initial_filters': filters, 'dropout': 0.0,
            'batch_normalization': True, 'output_activation': 'linear',
            'upsampling': 'bilinear'}
    if residual:
        cfg['residual_blocks'] = True
    return cfg


def _stage_groups(stage: tuple, horizon: int) -> dict[tuple[int, int, int], np.ndarray]:
    positions, target_years = stage[2], stage[3]
    keys = np.column_stack((target_years.astype(np.int32) - horizon,
                            positions[:, 0] // CORE, positions[:, 1] // CORE))
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    sorted_keys = keys[order]
    starts = np.flatnonzero(np.r_[True, np.any(sorted_keys[1:] != sorted_keys[:-1], axis=1)])
    stops = np.r_[starts[1:], len(order)]
    return {tuple(map(int, sorted_keys[start])): order[start:stop]
            for start, stop in zip(starts, stops)}


def _combined_stages(stages: dict) -> tuple:
    return tuple(np.concatenate((stages['ajuste'][i], stages['selecao_camadas'][i]))
                 for i in range(4))


def _choose_keys(groups: dict, cap: int | None, epoch: int, seed: int) -> list[tuple]:
    by_year: dict[int, list[tuple]] = {}
    for key in groups:
        by_year.setdefault(key[0], []).append(key)
    result = []
    for year, keys in sorted(by_year.items()):
        keys.sort()
        order = np.random.default_rng(seed + year).permutation(len(keys))
        count = len(keys) if cap is None else min(cap, len(keys))
        start = ((epoch - 1) * count) % len(keys)
        result.extend(keys[int(order[(start + i) % len(keys)])] for i in range(count))
    return result


class InputCache:
    """Read only the selected channels and tile halo from frozen aligned rasters."""

    def __init__(self, cfg: dict, region: str, profile: dict):
        self.cfg, self.region, self.profile = cfg, region, profile
        self.arrays: dict[int, np.ndarray] = {}

    def year(self, year: int) -> np.ndarray:
        if year not in self.arrays:
            path = cnn._aligned_cache(self.cfg, self.region, year, self.profile)
            array = np.load(path, mmap_mode='r')
            if array.shape != (24, self.profile['height'], self.profile['width']):
                raise ValueError(f'Input cache dimensions changed: {path}')
            self.arrays[year] = array
        return self.arrays[year]

    def patch(self, year: int, row: int, col: int, columns: tuple[int, ...],
              scale: dict, muted: tuple[int, ...] = ()) -> np.ndarray:
        array = self.year(year)
        height, width = self.profile['height'], self.profile['width']
        r0, c0 = max(0, row - HALO), max(0, col - HALO)
        r1, c1 = min(height, row + CORE + HALO), min(width, col + CORE + HALO)
        output = np.full((len(columns), CORE + 2 * HALO, CORE + 2 * HALO),
                         np.nan, dtype=np.float32)
        rr, cc = r0 - row + HALO, c0 - col + HALO
        output[:, rr:rr + r1-r0, cc:cc + c1-c0] = array[list(columns), r0:r1, c0:c1]
        output = (output - scale['input_mean'][:, None, None]) / scale['input_scale'][:, None, None]
        np.nan_to_num(output, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        if muted:
            output[list(muted)] = 0.0
        return output


def _artifact(model: UNet, columns: tuple[int, ...], scale: dict,
              params: dict, epoch: int) -> dict:
    residual = params.get('residual_blocks', False)
    return {'architecture': 'resunet' if residual else 'unet',
            'model_config': _model_config(params['initial_filters'], params.get('model_depth'), residual),
            'state_dict': {key: value.detach().cpu().clone()
                           for key, value in model.state_dict().items()},
            'columns': list(columns),
            'feature_names': [common.FEATURE_NAMES[index] for index in columns],
            'params': params, 'epoch': epoch,
            'input_mean': scale['input_mean'].tolist(),
            'input_scale': scale['input_scale'].tolist(),
            'target_mean': scale['target_mean'],
            'target_scale': scale['target_scale']}


def _restore(artifact: dict, device: torch.device) -> tuple[UNet, tuple[int, ...], dict]:
    if artifact.get('architecture') not in ('unet', 'resunet'):
        raise ValueError('Expected a U-Net artifact')
    if (artifact['architecture'] == 'resunet') != bool(artifact['model_config'].get('residual_blocks', False)):
        raise ValueError('Model architecture and residual configuration disagree')
    columns = tuple(artifact['columns'])
    model = UNet(len(columns), artifact['model_config']).to(device)
    model.load_state_dict(artifact['state_dict'])
    model.eval()
    scale = {'input_mean': np.asarray(artifact['input_mean'], dtype=np.float32),
             'input_scale': np.asarray(artifact['input_scale'], dtype=np.float32),
             'target_mean': float(artifact['target_mean']),
             'target_scale': float(artifact['target_scale'])}
    return model, columns, scale


def _predict_tile(model: UNet, cache: InputCache, key: tuple,
                  columns: tuple[int, ...], scale: dict, device: torch.device,
                  muted_global: tuple[int, ...] = ()) -> np.ndarray:
    year, tile_row, tile_col = key
    muted = tuple(columns.index(index) for index in muted_global if index in columns)
    x = cache.patch(year, tile_row * CORE, tile_col * CORE, columns, scale, muted)
    with torch.inference_mode():
        tensor = torch.from_numpy(x[None]).to(device)
        result = model(tensor)[0, 0, HALO:HALO + CORE,
                               HALO:HALO + CORE].cpu().numpy()
    return result * scale['target_scale'] + scale['target_mean']


def _predict_stage(artifact: dict, stage: tuple, groups: dict, cache: InputCache,
                   device: torch.device, keys: list[tuple] | None = None,
                   muted_global: tuple[int, ...] = ()) -> tuple[np.ndarray, np.ndarray]:
    model, columns, scale = _restore(artifact, device)
    positions = stage[2]
    indices, predictions = [], []
    for key in sorted(groups) if keys is None else keys:
        ids = groups[key]
        result = _predict_tile(model, cache, key, columns, scale, device, muted_global)
        rows, cols = positions[ids, 0] - key[1] * CORE, positions[ids, 1] - key[2] * CORE
        indices.append(ids)
        predictions.append(result[rows, cols].astype(np.float64))
    ids = np.concatenate(indices)
    predicted = np.concatenate(predictions)
    if not np.isfinite(predicted).all():
        raise ValueError('U-Net produced nonfinite predictions')
    return ids, predicted


def _score(artifact: dict, stage: tuple, groups: dict, cache: InputCache,
           device: torch.device, keys: list[tuple],
           muted: tuple[int, ...] = ()) -> tuple[float, float]:
    ids, prediction = _predict_stage(artifact, stage, groups, cache, device, keys, muted)
    return common._rmse(stage[1][ids], prediction), common._mae(stage[1][ids], prediction)


def _save_torch(path: Path, value: dict) -> None:
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def _fit(stage: tuple, groups: dict, cache: InputCache,
         columns: tuple[int, ...], scale: dict, params: dict,
         epochs: int, settings: dict, device: torch.device, folder: Path,
         monitor: tuple[tuple, dict, InputCache, list[tuple]] | None = None) -> tuple[dict, list[dict]]:
    """Train on rotating sparse-label cores, with an epoch-level resume file."""
    folder.mkdir(parents=True, exist_ok=True)
    result_path = folder / 'model.pt'
    history_path = folder / 'training_history.csv'
    if result_path.is_file() and history_path.is_file():
        artifact = torch.load(result_path, map_location='cpu', weights_only=True)
        with history_path.open(encoding='utf-8-sig', newline='') as handle:
            history = list(csv.DictReader(handle))
        if artifact['columns'] != list(columns) or artifact['params'] != params:
            raise ValueError(f'U-Net fit checkpoint differs: {folder}')
        return artifact, history
    torch.manual_seed(settings['random_state'])
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(settings['random_state'])
    model = UNet(len(columns), _model_config(params['initial_filters'], params.get('model_depth'),
                                           params.get('residual_blocks', False))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params['learning_rate'],
                                  weight_decay=params['weight_decay'])
    history: list[dict] = []
    best_rank = None
    stale = 0
    checkpoint_path = folder / 'checkpoint.pt'
    if checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if checkpoint['columns'] != list(columns) or checkpoint['params'] != params or checkpoint['epochs'] != epochs:
            raise ValueError(f'U-Net epoch checkpoint differs: {folder}')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        history = checkpoint['history']
        best_rank = checkpoint['best_rank']
        stale = checkpoint['stale']
    started = time.perf_counter()
    monitor_only = settings.get('monitor_only', False)
    first_epoch = (epochs + 1 if monitor is not None and not monitor_only and stale >= settings['patience']
                   else len(history) + 1)
    for epoch in range(first_epoch, epochs + 1):
        model.train()
        tile_seed = settings.get('tile_seed', settings['random_state'])
        tile_epoch = 1 if settings.get('tile_schedule') == 'fixed' else epoch
        keys = _choose_keys(groups, settings['training_tiles_per_year'], tile_epoch, tile_seed)
        order = np.random.default_rng(tile_seed + epoch).permutation(len(keys))
        weighted_loss = 0.0
        supervised = 0
        for step, offset in enumerate(order, 1):
            key = keys[int(offset)]
            ids = groups[key]
            x = cache.patch(key[0], key[1] * CORE, key[2] * CORE, columns, scale)
            tensor = torch.from_numpy(x[None]).to(device)
            rows = torch.as_tensor(stage[2][ids, 0] - key[1] * CORE,
                                   dtype=torch.long, device=device)
            cols = torch.as_tensor(stage[2][ids, 1] - key[2] * CORE,
                                   dtype=torch.long, device=device)
            target = torch.as_tensor(((stage[1][ids] - scale['target_mean']) /
                                      scale['target_scale']).astype(np.float32), device=device)
            optimizer.zero_grad(set_to_none=True)
            output = model(tensor)[0, 0, HALO:HALO + CORE, HALO:HALO + CORE]
            loss = torch.mean((output[rows, cols] - target) ** 2)
            if not torch.isfinite(loss):
                raise ValueError('U-Net training loss is nonfinite')
            loss.backward()
            optimizer.step()
            weighted_loss += float(loss.detach().cpu()) * len(ids)
            supervised += len(ids)
            if step % 16 == 0:
                logging.info('U-Net %s epoch %s: %s/%s tiles', folder.name,
                             epoch, step, len(keys))
        row = {'epoch': epoch, 'train_loss_scaled': weighted_loss / supervised,
               'optimizer_steps': (history[-1]['optimizer_steps'] if history else 0) + len(keys),
               'training_tiles': len(keys), 'supervised_pixels': supervised,
               'elapsed_seconds': round(time.perf_counter() - started, 2)}
        if monitor is not None:
            monitor_stage, monitor_groups, monitor_cache, monitor_keys = monitor
            candidate = _artifact(model, columns, scale, params, epoch)
            rmse, mae = _score(candidate, monitor_stage, monitor_groups,
                               monitor_cache, device, monitor_keys)
            row['selection_rmse'], row['selection_mae'] = rmse, mae
            if not monitor_only and epoch >= settings['min_epochs']:
                rank = (rmse, mae, epoch)
                if best_rank is None or rank < tuple(best_rank):
                    best_rank, stale = rank, 0
                    _save_torch(folder / 'best.pt', candidate)
                else:
                    stale += 1
        history.append(row)
        common._write_csv(history_path, history)
        _save_torch(checkpoint_path,
                    {'columns': list(columns), 'params': params, 'epochs': epochs,
                     'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                     'history': history, 'best_rank': best_rank, 'stale': stale})
        logging.info('U-Net %s epoch %s/%s loss %.6f%s', folder.name, epoch, epochs,
                     row['train_loss_scaled'],
                     '' if monitor is None else f" selection RMSE {row['selection_rmse']:.6f}")
        if monitor is not None and not monitor_only and stale >= settings['patience']:
            break
    if monitor is not None and not monitor_only:
        if best_rank is None:
            raise ValueError('U-Net selection had no eligible epoch')
        artifact = torch.load(folder / 'best.pt', map_location='cpu', weights_only=True)
    else:
        artifact = _artifact(model, columns, scale, params, len(history))
    _save_torch(result_path, artifact)
    checkpoint_path.unlink(missing_ok=True)
    return artifact, history


def _save_tile_plan(path: Path, groups: dict, settings: dict) -> list[tuple]:
    keys = _choose_keys(groups, settings['selection_tiles_per_year'], 1,
                        settings['random_state'] + 100000)
    payload = {'core_size': CORE, 'halo_pixels': HALO,
               'selection_tiles_per_year': settings['selection_tiles_per_year'],
               'tile_keys': [list(key) for key in keys],
               'total_available_tiles': len(groups),
               'selected_sample_centers': int(sum(len(groups[key]) for key in keys))}
    if path.is_file():
        if json.loads(path.read_text(encoding='utf-8')) != payload:
            raise ValueError('U-Net selection tile plan changed')
    else:
        common._write_json(path, payload)
    return keys


def _search(stages: dict, groups: dict, cache: InputCache, selection_keys: list[tuple],
            settings: dict, device: torch.device, folder: Path) -> dict:
    chosen, rows, epochs = {}, [], []
    for arm in BASE_ARMS:
        columns = common._arm_indices(arm)
        scale = cnn._scales(stages['ajuste'][0], stages['ajuste'][1], columns)
        family = []
        for candidate_id, params in enumerate(settings['candidates']):
            trial_folder = folder / 'trials' / f'{arm}_{candidate_id}'
            artifact, history = _fit(
                stages['ajuste'], groups['ajuste'], cache, columns, scale,
                params, settings['max_epochs'], settings, device, trial_folder,
                (stages['selecao_camadas'], groups['selecao_camadas'], cache, selection_keys))
            best_row = history[artifact['epoch'] - 1]
            row = {'base_arm': arm, 'candidate_id': candidate_id, 'params': params,
                   'selected_epoch': artifact['epoch'], 'actual_epochs': len(history),
                   'selection_rmse': float(best_row['selection_rmse']),
                   'selection_mae': float(best_row['selection_mae']),
                   'parameters': sum(p.numel() for p in
                                     UNet(len(columns), artifact['model_config']).parameters()),
                   'fit_samples_available': len(stages['ajuste'][1]),
                   'selection_samples_scored': sum(len(groups['selecao_camadas'][key])
                                                   for key in selection_keys),
                   'model_path': str(trial_folder / 'model.pt'),
                   'model_sha256': common._sha256(trial_folder / 'model.pt')}
            family.append(row)
            rows.append(row)
            epochs.extend({'base_arm': arm, 'candidate_id': candidate_id, **point}
                          for point in history)
        winner = min(family, key=lambda row: (row['selection_rmse'],
                                              row['selection_mae'],
                                              row['parameters'], row['candidate_id']))
        chosen[arm] = winner
        logging.info('U-Net %s chosen candidate %s, epoch %s, selection RMSE %.6f',
                     arm, winner['candidate_id'], winner['selected_epoch'],
                     winner['selection_rmse'])
    common._write_json(folder / 'hyperparameter_choices.json', chosen)
    common._write_csv(folder / 'hyperparameter_trials.csv',
                      [{key: json.dumps(value) if isinstance(value, (dict, list)) else value
                        for key, value in row.items()} for row in rows])
    common._write_csv(folder / 'hyperparameter_epoch_history.csv', epochs)
    return chosen


def _choose_filters(chosen: dict, stages: dict, groups: dict, cache: InputCache,
                    selection_keys: list[tuple], device: torch.device,
                    folder: Path) -> dict:
    path = folder / 'feature_occlusion.json'
    if path.is_file():
        return json.loads(path.read_text(encoding='utf-8'))
    score_stage, score_groups = stages['selecao_camadas'], groups['selecao_camadas']
    original_artifact = torch.load(chosen['originais']['model_path'],
                                   map_location='cpu', weights_only=True)
    original_columns = common._arm_indices('originais')
    original_rows = []
    for count in range(1, len(common.PHENOMENA) + 1):
        for subset in combinations(common.PHENOMENA, count):
            retained = common._arm_indices('originais', subset)
            muted = tuple(index for index in original_columns if index not in retained)
            rmse, mae = _score(original_artifact, score_stage, score_groups,
                               cache, device, selection_keys, muted)
            original_rows.append({'phenomena': list(subset), 'columns': list(retained),
                                  'features': [common.FEATURE_NAMES[i] for i in retained],
                                  'muted': list(muted), 'n_features': len(retained),
                                  'selection_rmse': rmse, 'selection_mae': mae,
                                  'selected': False})
    original_winner = min(original_rows, key=lambda row: (
        row['selection_rmse'], row['selection_mae'], row['n_features'], row['features']))
    original_winner['selected'] = True

    all_artifact = torch.load(chosen['todas']['model_path'],
                              map_location='cpu', weights_only=True)
    all_columns = common._arm_indices('todas')
    product_groups = common._product_groups()
    baseline_rmse, baseline_mae = _score(all_artifact, score_stage, score_groups,
                                         cache, device, selection_keys)
    all_rows = [{'removed_groups': [], 'columns': list(all_columns),
                 'features': list(common.FEATURE_NAMES), 'muted': [],
                 'n_features': 24, 'selection_rmse': baseline_rmse,
                 'selection_mae': baseline_mae, 'selected': False}]
    favorable = []
    for name, group in product_groups.items():
        rmse, mae = _score(all_artifact, score_stage, score_groups,
                           cache, device, selection_keys, tuple(group))
        retained = tuple(i for i in all_columns if i not in group)
        all_rows.append({'removed_groups': [name], 'columns': list(retained),
                         'features': [common.FEATURE_NAMES[i] for i in retained],
                         'muted': list(group), 'n_features': len(retained),
                         'selection_rmse': rmse, 'selection_mae': mae,
                         'selected': False})
        if rmse < baseline_rmse - 1e-12:
            favorable.append(name)
    if len(favorable) > 1:
        muted = tuple(i for name in favorable for i in product_groups[name])
        retained = tuple(i for i in all_columns if i not in muted)
        if retained:
            rmse, mae = _score(all_artifact, score_stage, score_groups,
                               cache, device, selection_keys, muted)
            all_rows.append({'removed_groups': favorable, 'columns': list(retained),
                             'features': [common.FEATURE_NAMES[i] for i in retained],
                             'muted': list(muted), 'n_features': len(retained),
                             'selection_rmse': rmse, 'selection_mae': mae,
                             'selected': False})
    all_winner = min(all_rows, key=lambda row: (
        row['selection_rmse'], row['selection_mae'], row['n_features'], row['features']))
    all_winner['selected'] = True
    result = {'method': 'zero occlusion on fixed selection tiles; filtered models refit from scratch',
              'originais_filtradas': {'chosen_columns': original_winner['columns'],
                                      'attempts': original_rows},
              'todas_filtradas': {'chosen_columns': all_winner['columns'],
                                  'attempts': all_rows}}
    common._write_json(path, result)
    common._write_csv(folder / 'feature_occlusion.csv', [
        {'arm': arm, 'features': ';'.join(row['features']),
         'muted': ';'.join(common.FEATURE_NAMES[i] for i in row['muted']),
         'removed_groups': ';'.join(row.get('removed_groups', [])),
         'n_features': row['n_features'], 'selection_rmse': row['selection_rmse'],
         'selection_mae': row['selection_mae'], 'selected': row['selected']}
        for arm in ('originais_filtradas', 'todas_filtradas')
        for row in result[arm]['attempts']])
    return result


def _train_arm(arm: str, choice: dict, columns: tuple[int, ...],
               stages: dict, groups: dict, combined: tuple, combined_groups: dict,
               cache: InputCache, selection_keys: list[tuple], threshold: float,
               profile: dict, settings: dict, device: torch.device,
               folder: Path) -> dict:
    arm_folder = folder / arm
    arm_folder.mkdir(exist_ok=True)
    metrics_path, model_path = arm_folder / 'metrics.json', arm_folder / 'model.pt'
    if metrics_path.is_file() and model_path.is_file():
        record = json.loads(metrics_path.read_text(encoding='utf-8'))
        if (record['feature_names'] != [common.FEATURE_NAMES[i] for i in columns] or
                record['hyperparameters'] != choice['params'] or
                record['model_sha256'] != common._sha256(model_path)):
            raise ValueError(f'U-Net completed arm differs: {arm_folder}')
        return record
    if arm in BASE_ARMS:
        initial = torch.load(choice['model_path'], map_location='cpu', weights_only=True)
    else:
        initial_scale = cnn._scales(stages['ajuste'][0], stages['ajuste'][1], columns)
        initial, _ = _fit(stages['ajuste'], groups['ajuste'], cache, columns,
                          initial_scale, choice['params'], choice['selected_epoch'],
                          settings, device, arm_folder / 'initial')
    initial_rmse, initial_mae = _score(initial, stages['selecao_camadas'],
                                      groups['selecao_camadas'], cache,
                                      device, selection_keys)
    scale = cnn._scales(combined[0], combined[1], columns)
    artifact, history = _fit(combined, combined_groups, cache, columns, scale,
                             choice['params'], choice['selected_epoch'],
                             settings, device, arm_folder / 'final')
    _save_torch(model_path, artifact)
    common._write_csv(arm_folder / 'training_history.csv', history)
    train_ids, train_pred = _predict_stage(artifact, combined, combined_groups,
                                           cache, device)
    valid_ids, valid_pred = _predict_stage(artifact, stages['validacao_modelo'],
                                           groups['validacao_modelo'], cache, device)
    valid = stages['validacao_modelo']
    record = {'arm': arm, 'n_features': len(columns),
              'feature_names': [common.FEATURE_NAMES[i] for i in columns],
              'hyperparameters': choice['params'],
              'selected_epoch': choice['selected_epoch'],
              'actual_epochs': len(history),
              'optimizer_steps': int(history[-1]['optimizer_steps']),
              'model_path': str(model_path), 'model_sha256': common._sha256(model_path),
              'parameters': sum(p.numel() for p in
                                UNet(len(columns), artifact['model_config']).parameters()),
              'input_scaler_mean': artifact['input_mean'],
              'input_scaler_scale': artifact['input_scale'],
              'target_scaler_mean': artifact['target_mean'],
              'target_scaler_scale': artifact['target_scale'],
              'initial_selection_metrics': {'rmse': initial_rmse, 'mae': initial_mae},
              'training_metrics': common.metrics(combined[1][train_ids], train_pred, threshold),
              'validation_metrics': common.metrics(valid[1][valid_ids], valid_pred, threshold),
              'validation_uncertainty': common._bootstrap_block_errors(
                  valid[1][valid_ids], valid_pred, valid[2][valid_ids, 0],
                  valid[2][valid_ids, 1], profile['width'],
                  seed=settings['random_state']),
              'validation_by_year': [
                  {'target_year': int(year),
                   'metrics': common.metrics(valid[1][valid_ids][valid[3][valid_ids] == year],
                                             valid_pred[valid[3][valid_ids] == year], threshold)}
                  for year in np.unique(valid[3])],
              'training_samples_available': len(combined[1]),
              'training_samples_evaluated': len(train_ids),
              'validation_samples': len(valid_ids),
              'selection_samples_scored': sum(len(groups['selecao_camadas'][key])
                                               for key in selection_keys)}
    common._write_json(metrics_path, record)
    logging.info('U-Net %s: validation RMSE %.6f with %s channels',
                 arm, record['validation_metrics']['rmse'], len(columns))
    return record


def train_horizon(cfg: dict, profile: dict, zones: np.ndarray,
                  horizon: int, folder: Path, device: torch.device) -> dict:
    folder.mkdir(exist_ok=True)
    sample_path = folder / 'samples.npz'
    source_sample = cfg['sample_source'] / f'horizon_{horizon}' / 'samples.npz'
    if not sample_path.is_file():
        shutil.copy2(source_sample, sample_path)
    if common._sha256(sample_path) != common._sha256(source_sample):
        raise ValueError('Frozen U-Net sample file changed')
    stages = cnn._load_stages(sample_path)
    cnn._check_stages(stages, zones, horizon)
    groups = {stage: _stage_groups(stages[stage], horizon) for stage in STAGES}
    combined = _combined_stages(stages)
    combined_groups = _stage_groups(combined, horizon)
    settings = cfg['raw']['unet']
    cache = InputCache(cfg, 'treinamento', profile)
    selection_keys = _save_tile_plan(folder / 'selection_tiles.json',
                                     groups['selecao_camadas'], settings)
    positives = combined[1][combined[1] > 0]
    threshold = float(np.quantile(
        positives, cfg['raw']['evaluation']['high_target_positive_quantile']))
    chosen = _search(stages, groups, cache, selection_keys, settings, device, folder)
    filters = _choose_filters(chosen, stages, groups, cache,
                              selection_keys, device, folder)
    records = []
    for arm in ARMS:
        base = 'originais' if arm == 'originais_filtradas' else (
               'todas' if arm == 'todas_filtradas' else arm)
        columns = tuple(filters[arm]['chosen_columns']) if arm in filters else common._arm_indices(arm)
        records.append(_train_arm(arm, chosen[base], columns, stages, groups,
                                  combined, combined_groups, cache, selection_keys,
                                  threshold, profile, settings, device, folder))
    winner = min(records, key=lambda row: (row['validation_metrics']['rmse'],
                                            row['validation_metrics']['mae'],
                                            row['n_features'], row['arm']))
    selected = {'horizon': horizon, 'arm': winner['arm'],
                'n_features': winner['n_features'],
                'feature_names': winner['feature_names'],
                'validation_metrics': winner['validation_metrics'],
                'high_target_threshold': threshold,
                'training_target_mean': float(np.mean(combined[1])),
                'model_path': winner['model_path']}
    common._write_json(folder / 'winner.json', selected)
    common._write_csv(folder / 'comparison.csv', [
        {'arm': row['arm'], 'n_features': row['n_features'],
         'selection_rmse': row['initial_selection_metrics']['rmse'],
         'training_rmse': row['training_metrics']['rmse'],
         'validation_rmse': row['validation_metrics']['rmse'],
         'validation_mae': row['validation_metrics']['mae'],
         'winner': row['arm'] == winner['arm']}
        for row in records])
    logging.info('U-Net H%s winner %s, RMSE %.6f', horizon, winner['arm'],
                 winner['validation_metrics']['rmse'])
    return {'winner': selected, 'records': records}


def evaluate_external(cfg: dict, selected: dict, horizon: int,
                      folder: Path, target_years: list[int],
                      device: torch.device) -> list[dict]:
    """Evaluate frozen winner on full external grids, one 512-pixel core at a time."""
    artifact = torch.load(selected['model_path'], map_location='cpu', weights_only=True)
    if artifact['feature_names'] != selected['feature_names']:
        raise ValueError('U-Net winner channels differ from its model artifact')
    model, columns, scale = _restore(artifact, device)
    records = []
    for target_year in target_years:
        record_path = folder / f'external_target_{target_year}_metrics.json'
        if record_path.is_file():
            records.append(json.loads(record_path.read_text(encoding='utf-8')))
            continue
        input_year = target_year - horizon
        profile = common._external_profile(cfg, input_year)
        target_path = common._target_path(cfg, 'teste_final', target_year)
        output_path = folder / f'external_prediction_target_{target_year}.tif'
        output_profile = profile.copy()
        output_profile.update(driver='GTiff', count=1, dtype='float32',
                              nodata=float('nan'), tiled=True,
                              blockxsize=256, blockysize=256, compress='deflate')
        cache = InputCache(cfg, 'teste_final', profile)
        target_parts, pred_parts, row_parts, col_parts = [], [], [], []
        with rasterio.open(target_path) as target_source, \
             rasterio.open(output_path, 'w', **output_profile) as output:
            if target_source.nodata == 0:
                raise ValueError('External target declares zero as NoData')
            with WarpedVRT(target_source, crs=profile['crs'],
                           transform=profile['transform'], width=profile['width'],
                           height=profile['height'], dtype='float32',
                           nodata=float('nan'), resampling=Resampling.nearest) as target:
                count = 0
                for row in range(0, profile['height'], CORE):
                    for col in range(0, profile['width'], CORE):
                        height = min(CORE, profile['height'] - row)
                        width = min(CORE, profile['width'] - col)
                        window = Window(col, row, width, height)
                        all_center = cache.year(input_year)[:, row:row + height,
                                                            col:col + width]
                        y = target.read(1, window=window,
                                        masked=True).astype('float32').filled(np.nan)
                        valid = np.isfinite(all_center).all(axis=0) & np.isfinite(y)
                        predicted = np.full((height, width), np.nan, dtype=np.float32)
                        if np.any(valid):
                            tile_key = (input_year, row // CORE, col // CORE)
                            estimate = _predict_tile(model, cache, tile_key, columns,
                                                      scale, device)[:height, :width]
                            if not np.isfinite(estimate[valid]).all():
                                raise ValueError('U-Net external prediction is nonfinite')
                            predicted[valid] = estimate[valid]
                            rr, cc = np.where(valid)
                            target_parts.append(y[valid].astype(np.float64))
                            pred_parts.append(estimate[valid].astype(np.float64))
                            row_parts.append(rr.astype(np.int32) + row)
                            col_parts.append(cc.astype(np.int32) + col)
                        output.write(predicted, 1, window=window)
                        count += 1
                        if count % 25 == 0:
                            logging.info('U-Net external H%s %s: %s tiles',
                                         horizon, target_year, count)
        if not target_parts:
            raise ValueError('U-Net external evaluation has no common valid pixels')
        y = np.concatenate(target_parts)
        predicted = np.concatenate(pred_parts)
        rows, cols = np.concatenate(row_parts), np.concatenate(col_parts)
        observed = common.metrics(y, predicted, selected['high_target_threshold'])
        zero = common.metrics(y, np.zeros_like(y), subsets=False)
        mean = common.metrics(y, np.full_like(y, selected['training_target_mean']),
                              subsets=False)
        persistence = common._persistence(cfg, target_year, profile, (rows, cols), y)
        uncertainty = common._bootstrap_block_errors(
            y, predicted, rows, cols, profile['width'],
            seed=cfg['raw']['sampling']['seed'] + horizon * 10 + target_year)
        record = {'horizon': horizon, 'input_year': input_year,
                  'target_year': target_year, 'prediction': str(output_path),
                  'target_source': str(target_path),
                  'target_sha256': common._sha256(target_path),
                  'common_pixels': len(y),
                  'evaluated_fraction': float(len(y) / (profile['height'] * profile['width'])),
                  'n_features': len(columns), 'feature_names': artifact['feature_names'],
                  'hyperparameters': artifact['params'],
                  'metrics': observed, 'reference_zero': zero,
                  'reference_training_mean': mean,
                  'reference_persistence': persistence,
                  'uncertainty': uncertainty}
        common._write_json(record_path, record)
        records.append(record)
        logging.info('U-Net external H%s %s RMSE %.6f MAE %.6f n=%s',
                     horizon, target_year, observed['rmse'], observed['mae'], len(y))
    return records


def _fmt(value, digits: int = 4) -> str:
    return '—' if value is None else f'{value:.{digits}f}'


def _plot_histories(run_dir: Path) -> None:
    plots = run_dir / 'graficos_treinamento'
    plots.mkdir(exist_ok=True)
    for horizon in (0, 1):
        folder = run_dir / f'horizon_{horizon}'
        chosen = json.loads((folder / 'hyperparameter_choices.json').read_text(encoding='utf-8'))
        figure, axes = plt.subplots(1, 2, figsize=(14, 5))
        for arm in BASE_ARMS:
            trial = folder / 'trials' / f"{arm}_{chosen[arm]['candidate_id']}" / 'training_history.csv'
            with trial.open(encoding='utf-8-sig', newline='') as handle:
                rows = list(csv.DictReader(handle))
            epochs = [int(row['epoch']) for row in rows]
            axes[0].plot(epochs, [float(row['selection_rmse']) for row in rows], label=arm)
            axes[1].plot(epochs, [float(row['train_loss_scaled']) for row in rows], label=arm)
        axes[0].set(title=f'H{horizon}: RMSE de seleção por época',
                    xlabel='Época', ylabel='RMSE (unidade do target)')
        axes[1].set(title=f'H{horizon}: perda de treino na busca',
                    xlabel='Época', ylabel='MSE padronizado')
        for axis in axes:
            axis.grid(alpha=.2)
            axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(plots / f'h{horizon}_busca.png', dpi=300)
        plt.close(figure)
        figure, axis = plt.subplots(figsize=(10, 5))
        for arm in ARMS:
            path = folder / arm / 'training_history.csv'
            with path.open(encoding='utf-8-sig', newline='') as handle:
                rows = list(csv.DictReader(handle))
            axis.plot([int(row['epoch']) for row in rows],
                      [float(row['train_loss_scaled']) for row in rows], label=arm)
        axis.set(title=f'H{horizon}: perda dos seis reajustes',
                 xlabel='Época', ylabel='MSE padronizado')
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(plots / f'h{horizon}_reajuste.png', dpi=300)
        plt.close(figure)


def rebuild_report(run_dir: Path) -> None:
    from plot_enriched_linear_external import render_one

    run_dir = run_dir.resolve()
    status_path = run_dir / 'run.json'
    status = json.loads(status_path.read_text(encoding='utf-8'))
    if status.get('kind') not in (KIND, 'enriched_resunet') or not status.get('final_external_test_used'):
        raise ValueError('Expected an externally evaluated U-Net run')
    figures = []
    external = []
    for horizon in (0, 1):
        for path in sorted((run_dir / f'horizon_{horizon}').glob('external_target_*_metrics.json')):
            record = json.loads(path.read_text(encoding='utf-8'))
            external.append(record)
            figures.append(render_one(record, run_dir))
    supplement = run_dir / 'supplemental_H1_2021'
    supplemental = json.loads((supplement / 'external_target_2021_metrics.json').read_text(encoding='utf-8'))
    external.append(supplemental)
    figures.append(render_one(supplemental, supplement))
    common._write_json(run_dir / 'figuras_teste_externo.json', figures)
    _plot_histories(run_dir)
    settings = yaml.safe_load((run_dir / 'config.yaml').read_text(encoding='utf-8'))['unet']
    residual = status.get('kind') == 'enriched_resunet'
    model_name = 'ResUNet' if residual else 'U-Net'
    lines = [
        f"# Relatório final — experimento {'11, ResUNet' if residual else '10, U-Net densa'}", '',
        f'Execução: `{run_dir}`. {model_name} com cinco reduções e expansão simétrica; '
        'entrada 960×960, núcleo de previsão 512×512, halo de 224 pixels '
        '(56 km). Saída linear na unidade do target.', '',
        '## Desenho e interpretação', '',
        'Ronneberger et al. (2015) justificam os caminhos de contração, expansão '
        'e conexões entre escalas da U-Net. Aqui a arquitetura foi adaptada '
        'à regressão de intensidade contínua. H0 usa inputs do mesmo ano; H1 '
        'do ano anterior. Target zero representa ausência de registro na fotografia, '
        'não impossibilidade de ocorrência.', '',
        f"Cada época usa até {settings['training_tiles_per_year']} núcleos por ano de input "
        'em rotação determinística e perda MSE apenas nos centros congelados. '
        f"A seleção e a ocultação usam {settings['selection_tiles_per_year']} núcleos "
        'fixos por ano; a validação final usa todos os centros congelados. '
        'Este regime de otimização difere dos modelos por pixel. '
        'O buffer espacial de três pixels não separa halos de 56 km, '
        'então a validação pode ser otimista. A área externa já foi estudada '
        'em experimentos anteriores e é exploratória.', '',
        'A busca comparou 8 e 16 filtros iniciais, AdamW e até '
        f"{settings['max_epochs']} épocas, com mínimo {settings['min_epochs']} "
        f"e paciência {settings['patience']}. A seleção de canais por ocultação "
        'não é importância causal; os braços filtrados foram reajustados do zero. '
        'Os targets externos não orientaram parâmetros, épocas, canais ou vencedor.', '',
        '## Doze braços treinados e validados', '',
        '| Horizonte | Braço | Canais | Filtros | Épocas | Passos AdamW | '
        'RMSE seleção | RMSE treino | RMSE validação | MAE validação | Venceu |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|',
    ]
    if residual:
        lines[lines.index('## Doze braços treinados e validados'):lines.index('## Doze braços treinados e validados')] = [
            '## Justificativa das conexões residuais', '',
            'He et al. (2016) propõem aprender uma função residual somada a um atalho; '
            'Zhang, Liu e Wang (2018) combinam esse princípio à U-Net para extração de estradas. '
            'Este experimento usa uma adaptação para regressão: em cada bloco do encoder, '
            'bottleneck e decoder, duas convoluções 3×3 com BatchNorm/ReLU são somadas '
            'a uma projeção 1×1, seguida de ReLU. As concatenações entre escalas permanecem. '
            'Não é reprodução exata da arquitetura dos artigos, nem utiliza pesos pré-treinados.', '',
            'A hipótese é que os atalhos possam facilitar a otimização. Mantêm-se os dados, '
            'os patches, a rotação, a busca e a semente do experimento 10; as projeções '
            'adicionam parâmetros. Vencedores, canais e épocas podem mudar após a busca, '
            'portanto a comparação final de vencedores não é uma ablação pura das conexões. '
            'Uma execução por configuração não mede estabilidade entre sementes. '
            'H0/H1 designam horizontes temporais, não testes estatísticos nulo/alternativo.', '',
            '[Protocolo ResUNet](protocols/EXPERIMENTO_11_RESUNET.md).', '']
    arms_csv = []
    for horizon in (0, 1):
        folder = run_dir / f'horizon_{horizon}'
        winner = json.loads((folder / 'winner.json').read_text(encoding='utf-8'))
        for arm in ARMS:
            record = json.loads((folder / arm / 'metrics.json').read_text(encoding='utf-8'))
            arms_csv.append({'horizon': horizon, 'arm': arm,
                             'features': ';'.join(record['feature_names']),
                             'n_features': record['n_features'],
                             'filters': record['hyperparameters']['initial_filters'],
                             'selected_epoch': record['selected_epoch'],
                             'optimizer_steps': record['optimizer_steps'],
                             'hyperparameters': json.dumps(record['hyperparameters']),
                             'selection_rmse': record['initial_selection_metrics']['rmse'],
                             'training_rmse': record['training_metrics']['rmse'],
                             'validation_rmse': record['validation_metrics']['rmse'],
                             'validation_mae': record['validation_metrics']['mae'],
                             'winner': arm == winner['arm']})
            lines.append(f"| H{horizon} | {arm} | {record['n_features']} | "
                         f"{record['hyperparameters']['initial_filters']} | "
                         f"{record['selected_epoch']} | {record['optimizer_steps']} | "
                         f"{_fmt(record['initial_selection_metrics']['rmse'])} | "
                         f"{_fmt(record['training_metrics']['rmse'])} | "
                         f"{_fmt(record['validation_metrics']['rmse'])} | "
                         f"{_fmt(record['validation_metrics']['mae'])} | "
                         f"{'sim' if arm == winner['arm'] else 'não'} |")
    common._write_csv(run_dir / 'resultados_12_treinamentos.csv', arms_csv)
    for horizon in (0, 1):
        example = json.loads((run_dir / f'horizon_{horizon}' / 'originais' /
                              'metrics.json').read_text(encoding='utf-8'))
        lines += [f"H{horizon}: {example['training_samples_available']:,} centros "
                  f"de ajuste+seleção disponíveis; {example['validation_samples']:,} "
                  'centros completos na validação. Os núcleos de treino giram '
                  'entre épocas conforme `selection_tiles.json` e a configuração.', '']
    lines += ['', 'Os arquivos `horizon_*/hyperparameter_trials.csv`, '
              '`horizon_*/hyperparameter_epoch_history.csv`, '
              '`horizon_*/feature_occlusion.csv` e os históricos de cada braço '
              'registram todas as tentativas, épocas, grupos ocultados e '
              'modelos reajustados. `selection_tiles.json` fixa os núcleos '
              'usados na seleção.', '', '## Vencedores e canais', '']
    for horizon in (0, 1):
        folder = run_dir / f'horizon_{horizon}'
        winner = json.loads((folder / 'winner.json').read_text(encoding='utf-8'))
        record = json.loads((folder / winner['arm'] / 'metrics.json').read_text(encoding='utf-8'))
        lines += [f"### H{horizon}: {winner['arm']} ({winner['n_features']} canais)", '',
                  'Canais: ' + ', '.join(winner['feature_names']) + '.', '',
                  f"Parâmetros: `{json.dumps(record['hyperparameters'])}`; "
                  f"épocas: {record['selected_epoch']}; passos: {record['optimizer_steps']}; "
                  f"RMSE validação: {_fmt(record['validation_metrics']['rmse'])}; "
                  f"MAE validação: {_fmt(record['validation_metrics']['mae'])}.", '']
    if residual:
        lines += ['## Detalhes de todos os braços e tentativas', '']
        for horizon in (0, 1):
            folder = run_dir / f'horizon_{horizon}'
            lines += [f'### H{horizon}: busca de hiperparâmetros', '',
                      '| Família | Candidato | Filtros | Época escolhida | Épocas realizadas | RMSE seleção | MAE seleção |',
                      '|---|---:|---:|---:|---:|---:|---:|']
            with (folder / 'hyperparameter_trials.csv').open(encoding='utf-8-sig', newline='') as handle:
                for trial in csv.DictReader(handle):
                    p = json.loads(trial['params'])
                    lines.append(f"| {trial['base_arm']} | {trial['candidate_id']} | {p['initial_filters']} | "
                                 f"{trial['selected_epoch']} | {trial['actual_epochs']} | "
                                 f"{float(trial['selection_rmse']):.4f} | {float(trial['selection_mae']):.4f} |")
            for arm in ARMS:
                record = json.loads((folder / arm / 'metrics.json').read_text(encoding='utf-8'))
                retained = record['feature_names']
                base = 'originais' if arm == 'originais_filtradas' else ('todas' if arm == 'todas_filtradas' else arm)
                removed = [common.FEATURE_NAMES[i] for i in common._arm_indices(base) if common.FEATURE_NAMES[i] not in retained]
                lines += ['', f'**H{horizon} · {arm}**', '', 'Canais mantidos: ' + ', '.join(retained) + '.', '',
                          'Removidos em relação à família de origem: ' + (', '.join(removed) or 'nenhum') + '.', '',
                          f"Parâmetros: `{json.dumps(record['hyperparameters'])}`; {record['parameters']:,} parâmetros treináveis; "
                          f"{record['actual_epochs']} épocas; {record['optimizer_steps']} passos.", '',
                          '| Conjunto | RMSE | MAE | R² | Viés | NMAD | P90 absoluto |',
                          '|---|---:|---:|---:|---:|---:|---:|']
                for key, label in (('training_metrics', 'Treino completo'), ('validation_metrics', 'Validação completa')):
                    m = record[key]
                    lines.append('| ' + label + ' | ' + ' | '.join(_fmt(m.get(k)) for k in
                        ('rmse', 'mae', 'r2', 'bias', 'nmad', 'p90_absolute_error')) + ' |')
                lines += ['', f'[Histórico](horizon_{horizon}/{arm}/training_history.csv) · '
                          f'[Métricas completas](horizon_{horizon}/{arm}/metrics.json)', '']
            occlusion = json.loads((folder / 'feature_occlusion.json').read_text(encoding='utf-8'))
            lines += [f'### H{horizon}: todas as combinações por ocultação', '',
                      'São avaliações do modelo base com canais ocultados, não treinamentos independentes. '
                      'Apenas a combinação escolhida de cada família é treinada novamente do zero.', '',
                      '| Família filtrada | Mantidos | Ocultados | RMSE seleção | MAE seleção | Escolhida |',
                      '|---|---|---|---:|---:|---|']
            for arm in ('originais_filtradas', 'todas_filtradas'):
                for trial in occlusion[arm]['attempts']:
                    lines.append(f"| {arm} | {', '.join(trial['features'])} | "
                                 f"{', '.join(common.FEATURE_NAMES[i] for i in trial['muted']) or 'nenhum'} | "
                                 f"{_fmt(trial['selection_rmse'])} | {_fmt(trial['selection_mae'])} | "
                                 f"{'sim' if trial['selected'] else 'não'} |")
            lines += ['']
    lines += ['## Testes externos após congelamento', '',
              'A persistência usa um target histórico e informação adicional. '
              'H1 2020→2021 é o par complementar. Todos os braços não vencedores '
              'ficaram fora deste teste.', '',
              '| Horizonte | Entrada→target | Canais | Pixels | RMSE | MAE | R² | '
              'Viés | NMAD | P90 abs. | Captura 10% | RMSE zero | RMSE persistência |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    external_csv = []
    for record in external:
        metric, persistence = record['metrics'], record['reference_persistence']
        supplementary = record is supplemental
        lines.append(f"| H{record['horizon']} | {record['input_year']}→{record['target_year']}"
                     f"{' (complementar)' if supplementary else ''} | "
                     f"{record['n_features']} | {record['common_pixels']:,} | "
                     f"{_fmt(metric['rmse'])} | {_fmt(metric['mae'])} | "
                     f"{_fmt(metric['r2'])} | {_fmt(metric['bias'])} | "
                     f"{_fmt(metric['nmad'])} | {_fmt(metric['p90_absolute_error'])} | "
                     f"{_fmt(100 * metric['capture_10pct'], 2)}% | "
                     f"{_fmt(record['reference_zero']['rmse'])} | "
                     f"{_fmt(None if persistence is None else persistence['metrics']['rmse'])} |")
        external_csv.append({'horizon': record['horizon'],
                             'input_year': record['input_year'],
                             'target_year': record['target_year'],
                             'supplemental': supplementary,
                             'n_features': record['n_features'],
                             'feature_names': ';'.join(record['feature_names']),
                             'common_pixels': record['common_pixels'],
                             'target_sha256': record['target_sha256'],
                             'prediction': record['prediction'],
                             'metrics': json.dumps(metric),
                             'zero': json.dumps(record['reference_zero']),
                             'mean': json.dumps(record['reference_training_mean']),
                             'persistence': json.dumps(persistence),
                             'uncertainty': json.dumps(record['uncertainty'])})
    common._write_csv(run_dir / 'teste_externo_metricas.csv', external_csv)
    lines += ['', 'Os JSONs externos incluem captura top 5/10/20%, erro em '
              'target zero/positivo/alto, referências e intervalos espaciais '
              'exploratórios. Previsões negativas não foram truncadas. '
              'CE90 posicional não se aplica ao erro de intensidade.', '',
              '## Curvas de treinamento', '',
              'As curvas de busca medem RMSE na seleção fixa por época; '
              'os reajustes finais mostram perda de treino.', '',
              '![Busca H0](graficos_treinamento/h0_busca.png)', '',
              '![Reajuste H0](graficos_treinamento/h0_reajuste.png)', '',
              '![Busca H1](graficos_treinamento/h1_busca.png)', '',
              '![Reajuste H1](graficos_treinamento/h1_reajuste.png)', '',
              '## Mapas externos: target original e previsão', '',
              'Cada figura usa a mesma máscara e escala nos dois painéis. '
              'PNG 600 dpi; os GeoTIFFs preservam georreferência e resolução.', '']
    for record in external:
        horizon, iy, ty = record['horizon'], record['input_year'], record['target_year']
        name = f'teste_externo_H{horizon}_{iy}_{ty}.png'
        figure_path = f'supplemental_H1_2021/{name}' if record is supplemental else name
        lines += [f'**H{horizon} · {iy}→{ty}**', '',
                  f'![Target original e previsão {model_name} H{horizon} {iy}→{ty}]({figure_path})', '',
                  f"[GeoTIFF da previsão]({Path(record['prediction']).relative_to(run_dir).as_posix()})", '']
    lines += ['## Reprodutibilidade e referências', '',
              'Código, configuração, protocolos, hashes das amostras e versões '
              'foram congelados neste run. Os caches de inputs são somente de leitura; '
              'targets externos foram abertos após a escolha dos vencedores.', '',
              '- [Protocolo U-Net](protocols/EXPERIMENTO_10_UNET.md) e '
              '[protocolo comum](protocols/EXPERIMENTOS_ENRIQUECIDOS_2026.md).',
              '- [Ronneberger et al. (2015), U-Net](https://arxiv.org/abs/1505.04597).',
              '- [Ioffe e Szegedy (2015), BatchNorm](https://arxiv.org/abs/1502.03167).',
              '- [Loshchilov e Hutter (2019), AdamW](https://arxiv.org/abs/1711.05101).',
              '- [Valavi et al. (2019), avaliação espacial](https://doi.org/10.1111/2041-210X.13107).', '']
    lines += [
        '- [Métricas de regressão, scikit-learn](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics): RMSE, MAE e R².',
        '- [Desvio absoluto mediano, SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.median_abs_deviation.html): base do NMAD dos resíduos.',
        '- [WarpedVRT e reamostragem, Rasterio](https://rasterio.readthedocs.io/en/stable/api/rasterio.vrt.html): alinhamento para inferência.',
        '- [numpy.gradient](https://numpy.org/doc/stable/reference/generated/numpy.gradient.html) e [cKDTree](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html): construção dos canais GPA e de grafos, detalhada nos documentos de inputs.', '',
    ]
    if residual:
        lines += ['- [He et al. (2016), Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385).',
                  '- [Zhang, Liu e Wang (2018), Road Extraction by Deep Residual U-Net](https://doi.org/10.1109/LGRS.2018.2802944); [preprint 2017](https://arxiv.org/abs/1711.10684).', '',
                  'As figuras são geradas automaticamente; a inspeção visual será realizada na análise após a execução.', '']
    (run_dir / 'RELATORIO.md').write_text('\n'.join(lines), encoding='utf-8')
    status['status'] = 'completed'
    common._write_json(status_path, status)
    print(run_dir / 'RELATORIO.md')


def _snapshot(cfg: dict, run_dir: Path) -> None:
    shutil.copy2(cfg['path'], run_dir / 'config.yaml')
    project = Path(__file__).parents[2]
    sources = ('run_enriched_unet.py', 'src/services/enriched_unet.py',
               'src/image_pipeline/models/unet.py',
               'src/image_pipeline/models/cnn.py',
               'src/services/enriched_cnn.py',
               'src/services/enriched_linear.py',
               'src/services/temporal_service.py',
               'src/splitting/spatial_split.py',
               'src/geospatial/raster_io.py',
               'plot_enriched_linear_external.py',
               'generate_gradient_inputs.py', 'generate_graph_inputs.py')
    if cfg.get('kind') == 'enriched_resunet':
        sources += ('run_enriched_resunet.py', 'src/services/enriched_resunet.py',
                    'src/services/enriched_unet_controls.py')
    entries = []
    for relative in sources:
        source = project / relative
        destination = run_dir / 'source_code' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        entries.append({'path': relative, 'sha256': common._sha256(source)})
    common._write_json(run_dir / 'source_files.json', entries)
    documents = ('protocols/EXPERIMENTO_10_UNET.md',
                     'protocols/EXPERIMENTOS_ENRIQUECIDOS_2026.md',
                     'docs/GRADIENT_INPUTS.md', 'docs/GRAPH_INPUTS.md')
    if cfg.get('kind') == 'enriched_resunet':
        documents += ('protocols/EXPERIMENTO_11_RESUNET.md',)
    for relative in documents:
        source = project / relative
        destination = run_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    packages = {name: importlib.metadata.version(name) for name in
                ('numpy', 'torch', 'rasterio', 'matplotlib', 'scikit-learn',
                 'PyYAML', 'psutil')}
    common._write_json(run_dir / 'environment.json',
                       {'python': sys.version, 'platform': platform.platform(),
                        'packages': packages,
                        'cuda_available': torch.cuda.is_available(),
                        'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
                        'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
                        'cuda_device': torch.cuda.get_device_name(0)
                        if torch.cuda.is_available() else None})


def run(config_path: Path, resume: Path | None = None) -> None:
    cfg = load_config(config_path)
    cfg['output'].mkdir(parents=True, exist_ok=True)
    if resume is None:
        run_dir = cfg['output'] / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        run_dir.mkdir(exist_ok=False)
        status = {'kind': cfg['kind'], 'status': 'running',
                  'started': datetime.now(timezone.utc).isoformat(),
                  'run': str(run_dir), 'sample_source': str(cfg['sample_source']),
                  'input_cache_source': str(cfg['cache_source']),
                  'final_external_test_used': False}
        common._write_json(run_dir / 'run.json', status)
    else:
        run_dir = resume.resolve()
        if not run_dir.is_dir() or run_dir.parent != cfg['output'].resolve():
            raise ValueError('Resume path must be an unfinished U-Net run')
        status = json.loads((run_dir / 'run.json').read_text(encoding='utf-8'))
        if status.get('kind') != cfg['kind'] or status.get('status') == 'completed':
            raise ValueError('Only an unfinished U-Net run can be resumed')
        if (run_dir / 'config.yaml').read_bytes() != cfg['path'].read_bytes():
            raise ValueError('U-Net config changed since run began')
        status['status'] = 'running'
        status['resumed'] = datetime.now(timezone.utc).isoformat()
        common._write_json(run_dir / 'run.json', status)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    file_handler = logging.FileHandler(run_dir / 'training.log', mode='a', encoding='utf-8')
    stream_handler = logging.StreamHandler()
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    device = cnn._device(cfg['raw']['unet'])
    try:
        if not (run_dir / 'source_files.json').is_file():
            _snapshot(cfg, run_dir)
        source_info = run_dir / 'sample_source.json'
        if not source_info.is_file():
            hashes = {f'H{h}': common._sha256(
                cfg['sample_source'] / f'horizon_{h}' / 'samples.npz') for h in (0, 1)}
            source_audit = json.loads((cfg['sample_source'] / 'input_audit.json').read_text(encoding='utf-8'))
            current_audit = common.audit_catalog(cfg, verify_hashes=False)
            if [(x['path'], x['sha256']) for x in source_audit['recorded_files']] != [
                    (x['path'], x['sha256']) for x in current_audit['recorded_files']]:
                raise ValueError('U-Net raster manifest differs from verified source')
            common._write_json(run_dir / 'input_audit.json', current_audit)
            common._write_json(source_info,
                               {'source': str(cfg['sample_source']), 'sha256': hashes,
                                'manifest_matches_verified_source': True})
        hashes = json.loads(source_info.read_text(encoding='utf-8'))['sha256']
        for horizon in (0, 1):
            sample = cfg['sample_source'] / f'horizon_{horizon}' / 'samples.npz'
            if common._sha256(sample) != hashes[f'H{horizon}']:
                raise ValueError('Frozen U-Net sample source changed')
            cnn._verify_reused_patch_bank(cfg, horizon, sample)
        profile = common._profile(cfg['reference'])
        zones = split_grid((profile['height'], profile['width']), cfg['raw']['split'])
        if not (run_dir / 'spatial_zones.tif').is_file():
            write_raster(run_dir / 'spatial_zones.tif', zones, profile, categorical=True)
        torch.set_num_threads(cfg['raw']['runtime']['cpu_threads'])
        if device.type == 'cuda':
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        logging.info('U-Net dense core %s halo %s depth %s device %s',
                     CORE, HALO, DEPTH, device)
        horizons = {}
        for horizon in (0, 1):
            horizons[horizon] = train_horizon(cfg, profile, zones, horizon,
                                              run_dir / f'horizon_{horizon}', device)
        status['selection_finished'] = datetime.now(timezone.utc).isoformat()
        status['winners'] = {str(h): horizons[h]['winner']['arm'] for h in (0, 1)}
        common._write_json(run_dir / 'run.json', status)
        for horizon in (0, 1):
            evaluate_external(cfg, horizons[horizon]['winner'], horizon,
                              run_dir / f'horizon_{horizon}',
                              cfg['raw']['evaluation'][f'final_h{horizon}_target_years'],
                              device)
        supplement = run_dir / 'supplemental_H1_2021'
        supplement.mkdir(exist_ok=True)
        evaluate_external(cfg, horizons[1]['winner'], 1, supplement, [2021], device)
        status['final_external_test_used'] = True
        status['external_finished'] = datetime.now(timezone.utc).isoformat()
        common._write_json(run_dir / 'run.json', status)
        rebuild_report(run_dir)
        status = json.loads((run_dir / 'run.json').read_text(encoding='utf-8'))
        status['finished'] = datetime.now(timezone.utc).isoformat()
        common._write_json(run_dir / 'run.json', status)
    except KeyboardInterrupt:
        status['status'] = 'interrupted'
        status['interrupted'] = datetime.now(timezone.utc).isoformat()
        common._write_json(run_dir / 'run.json', status)
        logging.warning('U-Net interrupted; resume with --resume %s', run_dir)
        raise SystemExit(130)
    except Exception as exc:
        status['status'] = 'failed'
        status['failed'] = datetime.now(timezone.utc).isoformat()
        status['error'] = repr(exc)
        common._write_json(run_dir / 'run.json', status)
        logging.exception('U-Net failed; partial artifacts retained at %s', run_dir)
        raise
    finally:
        logger.removeHandler(file_handler)
        logger.removeHandler(stream_handler)
        file_handler.close()
        stream_handler.close()
