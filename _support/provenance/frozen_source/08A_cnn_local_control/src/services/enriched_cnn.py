"""Experiment 08: center-pixel CNN over 5x5 input-only raster neighborhoods."""
from __future__ import annotations

from contextlib import ExitStack
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
from torch import nn
from torch.nn import functional as F

from src.services import enriched_linear as common
from src.services.enriched_hgb import selection_subsample
from src.splitting.spatial_split import split_grid
from src.geospatial.raster_io import write_raster


ARMS = ('originais', 'originais_filtradas', 'gpa', 'grafos', 'todas', 'todas_filtradas')
BASE_ARMS = ('originais', 'gpa', 'grafos', 'todas')
STAGES = ('ajuste', 'selecao_camadas', 'validacao_modelo')
RADIUS = 2
WIDE_STEP = 82
WIDE_RADIUS = 2 * WIDE_STEP


class CompactCNN(nn.Module):
    """Two-scale CNN, optionally ablated to the original local branch only."""

    def __init__(self, channels: int, filters: int, wide_context: bool = True):
        super().__init__()
        self.wide_context = wide_context
        self.local1 = nn.Conv2d(channels, filters, 3)
        self.local2 = nn.Conv2d(filters, filters, 3)
        if wide_context:
            self.wide1 = nn.Conv2d(channels, filters, 3)
            self.wide2 = nn.Conv2d(filters, filters, 3)
        self.head = nn.Conv2d(2 * filters if wide_context else filters, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = F.relu(self.local2(F.relu(self.local1(x[:, 0]))))
        if not self.wide_context:
            return self.head(local).squeeze(1)
        wide = F.relu(self.wide2(F.relu(self.wide1(x[:, 1]))))
        return self.head(torch.cat((local, wide), dim=1)).squeeze(1)

    def forward_dense(self, x: torch.Tensor, height: int, width: int,
                      step: int = WIDE_STEP) -> torch.Tensor:
        """Dense equivalent of patch inference; x includes a 2*step halo."""
        radius = 2 * step if self.wide_context else RADIUS
        local_x = x[:, :, radius - RADIUS:radius + height + RADIUS,
                    radius - RADIUS:radius + width + RADIUS]
        local = F.relu(self.local2(F.relu(self.local1(local_x))))
        if not self.wide_context:
            return self.head(local).squeeze(1)
        wide = F.relu(F.conv2d(x, self.wide1.weight, self.wide1.bias,
                               dilation=step))
        wide = F.relu(F.conv2d(wide, self.wide2.weight, self.wide2.bias,
                               dilation=step))
        return self.head(torch.cat((local, wide), dim=1)).squeeze(1)


def load_config(path: Path) -> dict:
    cfg = common.load_config(path)
    raw = cfg['raw']
    settings = raw.get('cnn', {})
    source = (cfg['path'].parent / raw['sample_source']).resolve()
    if cfg['output'] == source or cfg['output'].is_relative_to(source):
        raise ValueError('CNN output must be separate from the sample-source run')
    status = json.loads((source / 'run.json').read_text(encoding='utf-8'))
    if status.get('kind') != 'enriched_mlp' or status.get('status') != 'completed':
        raise ValueError('Sample source must be the completed enriched MLP run')
    source_cfg = __import__('yaml').safe_load((source / 'config.yaml').read_text(encoding='utf-8'))
    if source_cfg['data_root'] != raw['data_root'] or source_cfg['split'] != raw['split']:
        raise ValueError('Data root or split differs from the frozen sample source')
    for key in ('per_stage_year', 'seed'):
        if source_cfg['sampling'][key] != raw['sampling'][key]:
            raise ValueError(f'Sampling {key} differs from frozen source')
    if not (source / 'input_audit.json').is_file():
        raise FileNotFoundError('Sample source lacks its input audit')
    if not json.loads((source / 'input_audit.json').read_text(encoding='utf-8'))['hashes_recomputed']:
        raise ValueError('Sample source did not verify raster hashes')
    integer_keys = ('patch_size', 'wide_step_pixels', 'tile_size', 'selection_seed',
                    'selection_samples_per_stage_year', 'max_epochs',
                    'min_epochs', 'patience', 'batch_size', 'random_state')
    if any(type(settings.get(k)) is not int or settings[k] < 1 for k in integer_keys):
        raise ValueError('All CNN size, epoch, seed and batch settings must be positive integers')
    if (settings['patch_size'] != 5 or settings['wide_step_pixels'] != WIDE_STEP or
            settings['min_epochs'] > settings['max_epochs']):
        raise ValueError('This CNN requires patch_size=5, wide_step_pixels=82 and min_epochs<=max_epochs')
    if raw['split']['buffer_pixels'] < RADIUS:
        raise ValueError('Spatial buffer must cover the local CNN receptive radius')
    if settings.get('device') not in ('auto', 'cpu', 'cuda'):
        raise ValueError('cnn.device must be auto, cpu or cuda')
    candidates = settings.get('candidates', [])
    if not candidates or len({(v.get('filters'), v.get('learning_rate'), v.get('weight_decay'))
                              for v in candidates}) != len(candidates):
        raise ValueError('CNN candidates must be a nonempty unique list')
    for item in candidates:
        if set(item) != {'filters', 'learning_rate', 'weight_decay'}:
            raise ValueError('CNN candidate keys must be filters, learning_rate, weight_decay')
        if type(item['filters']) is not int or item['filters'] < 1:
            raise ValueError('CNN filters must be a positive integer')
        if not math.isfinite(float(item['learning_rate'])) or item['learning_rate'] <= 0:
            raise ValueError('CNN learning rate must be finite and positive')
        if not math.isfinite(float(item['weight_decay'])) or item['weight_decay'] < 0:
            raise ValueError('CNN weight decay must be finite and nonnegative')
    for h in (0, 1):
        if not (source / f'horizon_{h}' / 'samples.npz').is_file():
            raise FileNotFoundError(f'Frozen sample file H{h} is missing')
    cfg['sample_source'] = source
    return cfg


def check_only(path: Path) -> None:
    cfg = load_config(path)
    audit = common.audit_catalog(cfg, verify_hashes=False)
    for h in (0, 1):
        with np.load(cfg['sample_source'] / f'horizon_{h}' / 'samples.npz') as data:
            for stage in STAGES:
                if data[f'{stage}_x'].shape[1] != len(common.FEATURE_NAMES):
                    raise ValueError(f'H{h} {stage} does not have 24 input channels')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if cfg['raw']['cnn']['device'] == 'cuda' and device != 'cuda':
        raise RuntimeError('CUDA requested but unavailable')
    print(f"PRECHECK OK: {audit['input_and_development_target_files']} inputs/targets; "
          f"24 channels, H0/H1, {len(cfg['raw']['cnn']['candidates'])} CNN candidates "
          f"per base arm; {device} available; external targets not opened.")


def _device(settings: dict) -> torch.device:
    mode = settings['device']
    if mode == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    return torch.device('cuda' if mode == 'cuda' or
                        (mode == 'auto' and torch.cuda.is_available()) else 'cpu')


def _load_stages(path: Path) -> dict:
    with np.load(path) as sample:
        return {stage: tuple(sample[f'{stage}_{part}'] for part in
                             ('x', 'y', 'positions', 'target_year')) for stage in STAGES}


def _check_stages(stages: dict, zones: np.ndarray, horizon: int) -> None:
    for stage, (x, y, positions, years) in stages.items():
        if x.ndim != 2 or x.shape[1] != 24 or len(x) != len(y) or len(y) != len(positions):
            raise ValueError(f'H{horizon} {stage} sample shapes disagree')
        if not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(y < 0):
            raise ValueError(f'H{horizon} {stage} has invalid center values')
        if len(y) == 0 or np.any(positions < 0) or np.any(positions[:, 0] >= zones.shape[0]) or np.any(positions[:, 1] >= zones.shape[1]):
            raise ValueError(f'H{horizon} {stage} has invalid positions')
        got_zone = zones[positions[:, 0], positions[:, 1]]
        expected_zone = {'ajuste': 1, 'selecao_camadas': 2, 'validacao_modelo': 3}[stage]
        if horizon == 0 and not np.all(got_zone == expected_zone):
            raise ValueError(f'H0 {stage} positions violate spatial zone')
        if horizon == 1 and (not np.all(got_zone == 1) or
                             not np.all(years == {'ajuste': 2020, 'selecao_camadas': 2021,
                                                   'validacao_modelo': 2022}[stage])):
            raise ValueError(f'H1 {stage} positions violate temporal protocol')


class AlignedInputs:
    """Read 24 feature channels on one grid; no target is opened here."""

    def __init__(self, cfg: dict, region: str, year: int, profile: dict,
                 cache: Path | None = None):
        self.stack = ExitStack()
        self.profile = profile
        self.readers = []
        self.cache = None
        if cache is not None:
            self.cache = np.load(cache, mmap_mode='r')
            if self.cache.shape != (24, profile['height'], profile['width']):
                raise ValueError(f'Aligned input cache has wrong dimensions: {cache}')
            return
        try:
            paths = common._paths(cfg['root'], region, year)
            for path, method in zip(paths, common._methods(cfg)[:-1]):
                src = self.stack.enter_context(rasterio.open(path))
                self.readers.append(self.stack.enter_context(WarpedVRT(
                    src, crs=profile['crs'], transform=profile['transform'],
                    width=profile['width'], height=profile['height'],
                    dtype='float32', nodata=float('nan'),
                    resampling=Resampling[method], warp_mem_limit=64)))
        except BaseException:
            self.stack.close()
            raise

    def close(self) -> None:
        self.stack.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def tile(self, row: int, col: int, height: int, width: int,
             radius: int = WIDE_RADIUS) -> np.ndarray:
        """Return C x (height+2r) x (width+2r), NaN outside raster."""
        r0, c0 = max(0, row - radius), max(0, col - radius)
        r1 = min(self.profile['height'], row + height + radius)
        c1 = min(self.profile['width'], col + width + radius)
        window = Window(c0, r0, c1 - c0, r1 - r0)
        out = np.full((24, height + 2 * radius, width + 2 * radius),
                      np.nan, dtype=np.float32)
        rr, cc = r0 - (row - radius), c0 - (col - radius)
        if self.cache is not None:
            out[:, rr:rr + r1-r0, cc:cc + c1-c0] = self.cache[:, r0:r1, c0:c1]
        else:
            for channel, reader in enumerate(self.readers):
                value = reader.read(1, window=window, masked=True).astype('float32').filled(np.nan)
                out[channel, rr:rr + value.shape[0], cc:cc + value.shape[1]] = value
        return out


def _aligned_cache(cfg: dict, region: str, year: int, profile: dict) -> Path:
    """Warp a year's 24 inputs once, sequentially, before random halo reads."""
    folder = cfg['_aligned_cache']
    folder.mkdir(exist_ok=True)
    path = folder / f'{region}_{year}.npy'
    marker = folder / f'{region}_{year}.json'
    if marker.is_file() and path.is_file():
        metadata = json.loads(marker.read_text(encoding='utf-8'))
        if metadata['shape'] != [24, profile['height'], profile['width']]:
            raise ValueError(f'Cached aligned grid changed: {path}')
        return path
    path.unlink(missing_ok=True)
    bank = np.lib.format.open_memmap(path, mode='w+', dtype='float32',
                                     shape=(24, profile['height'], profile['width']))
    paths = common._paths(cfg['root'], region, year)
    methods = common._methods(cfg)[:-1]
    started = time.perf_counter()
    for index, (window, values) in enumerate(common.aligned_windows(
            paths, methods, profile, cfg['raw']['sampling']['window_size']), 1):
        row, col = int(window.row_off), int(window.col_off)
        bank[:, row:row + int(window.height), col:col + int(window.width)] = \
            values.transpose(2, 0, 1)
        if index % 100 == 0:
            logging.info('Aligned input cache %s %s: %s tiles', region, year, index)
    bank.flush()
    del bank
    metadata = {'region': region, 'year': year,
                'shape': [24, profile['height'], profile['width']],
                'paths': [str(p) for p in paths], 'methods': methods,
                'seconds': round(time.perf_counter() - started, 2),
                'source': 'inputs only; no target raster opened'}
    common._write_json(marker, metadata)
    logging.info('Aligned input cache %s %s completed in %.1f s',
                 region, year, metadata['seconds'])
    return path


def extract_two_scales(block: np.ndarray, row: int, col: int,
                       step: int = WIDE_STEP) -> np.ndarray:
    """Extract local 5x5 and sparse 5x5 from a haloed tile."""
    radius = 2 * step
    local_r, local_c = row + radius - RADIUS, col + radius - RADIUS
    local = block[:, local_r:local_r + 5, local_c:local_c + 5]
    sample_r = row + radius + np.arange(-2, 3) * step
    sample_c = col + radius + np.arange(-2, 3) * step
    wide = block[:, sample_r[:, None], sample_c[None, :]]
    return np.stack((local, wide)).astype(np.float32, copy=False)


def _fill_patch_bank(cfg: dict, profile: dict, stages: dict, horizon: int,
                     directory: Path) -> dict:
    """Read each input tile once per input year for all three sampled stages."""
    marker = directory / 'patch_bank.json'
    if marker.is_file():
        info = json.loads(marker.read_text(encoding='utf-8'))
        if (info.get('sample_source_sha256') == common._sha256(directory / 'samples.npz')
                and info.get('wide_step_pixels') == WIDE_STEP
                and all((directory / f'{stage}_patches.npy').is_file() for stage in STAGES)):
            return info
    for stage in STAGES:
        (directory / f'{stage}_patches.npy').unlink(missing_ok=True)
    banks = {stage: np.lib.format.open_memmap(
        directory / f'{stage}_patches.npy', mode='w+', dtype='float32',
        shape=(len(stages[stage][1]), 2, 24, 5, 5)) for stage in STAGES}
    tile = cfg['raw']['cnn']['tile_size']
    cols_per_row = math.ceil(profile['width'] / tile)
    total_tiles = 0
    for input_year in sorted({int(year - horizon) for stage in STAGES
                              for year in stages[stage][3]}):
        groups: dict[int, list[tuple[str, int, int, int]]] = {}
        for stage in STAGES:
            _, _, positions, years = stages[stage]
            for idx in np.flatnonzero(years == input_year + horizon):
                row, col = map(int, positions[idx])
                key = (row // tile) * cols_per_row + col // tile
                groups.setdefault(key, []).append((stage, int(idx), row, col))
        cache = _aligned_cache(cfg, 'treinamento', input_year, profile)
        with AlignedInputs(cfg, 'treinamento', input_year, profile, cache) as reader:
            for count, key in enumerate(sorted(groups), 1):
                row = (key // cols_per_row) * tile
                col = (key % cols_per_row) * tile
                height = min(tile, profile['height'] - row)
                width = min(tile, profile['width'] - col)
                block = reader.tile(row, col, height, width)
                for stage, idx, sample_row, sample_col in groups[key]:
                    dr, dc = sample_row - row, sample_col - col
                    banks[stage][idx] = extract_two_scales(block, dr, dc)
                if count % 200 == 0:
                    logging.info('H%s input year %s: patch tiles %s/%s',
                                 horizon, input_year, count, len(groups))
        total_tiles += len(groups)
        logging.info('H%s input year %s: %s patch tiles complete',
                     horizon, input_year, len(groups))
    for bank in banks.values():
        bank.flush()
    info = {'horizon': horizon, 'shape': {s: list(banks[s].shape) for s in STAGES},
            'total_tiles': total_tiles, 'patch_size': 5,
            'wide_step_pixels': WIDE_STEP,
            'sample_source_sha256': common._sha256(directory / 'samples.npz')}
    common._write_json(marker, info)
    return info


class PatchView:
    """Indexed view over one or more memory-mapped patch banks."""

    def __init__(self, segments: list[tuple[np.ndarray, np.ndarray]],
                 labels: list[np.ndarray]):
        self.segments = segments
        self.lengths = np.array([len(indices) for _, indices in segments], dtype=np.int64)
        self.offsets = np.cumsum(self.lengths)
        self.y = np.concatenate(labels).astype(np.float64)
        if len(self.y) != int(self.offsets[-1]):
            raise ValueError('Patch indices and labels disagree')

    def __len__(self):
        return len(self.y)

    def batch(self, ids: np.ndarray) -> np.ndarray:
        out = np.empty((len(ids), 2, 24, 5, 5), dtype=np.float32)
        start = 0
        for (bank, indices), end in zip(self.segments, self.offsets):
            mask = (ids >= start) & (ids < end)
            if np.any(mask):
                out[mask] = bank[indices[ids[mask] - start]]
            start = int(end)
        return out


def _view(bank: np.ndarray, labels: np.ndarray, indices: np.ndarray | None = None) -> PatchView:
    if indices is None:
        indices = np.arange(len(labels))
    return PatchView([(bank, indices)], [labels[indices]])


def _two_stage_view(banks: dict, stages: dict) -> PatchView:
    return PatchView([(banks[stage], np.arange(len(stages[stage][1])))
                      for stage in ('ajuste', 'selecao_camadas')],
                     [stages[stage][1] for stage in ('ajuste', 'selecao_camadas')])


def _scales(x: np.ndarray, y: np.ndarray, columns: tuple[int, ...]) -> dict:
    mean = np.mean(x[:, columns], axis=0, dtype=np.float64)
    scale = np.std(x[:, columns], axis=0, dtype=np.float64)
    scale[scale == 0] = 1.0
    return {'input_mean': mean.astype(np.float32),
            'input_scale': scale.astype(np.float32),
            'target_mean': float(np.mean(y)),
            'target_scale': float(np.std(y)) or 1.0}


def _tensor(raw: np.ndarray, columns: tuple[int, ...], scale: dict,
            device: torch.device, muted: tuple[int, ...] = (),
            wide_context: bool = True) -> torch.Tensor:
    if not wide_context:
        raw = raw[:, :1]
    x = (raw[:, :, columns] - scale['input_mean'][None, None, :, None, None]) / scale['input_scale'][None, None, :, None, None]
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if muted:
        x[:, :, muted] = 0.0
    return torch.from_numpy(np.ascontiguousarray(x)).to(device)


def _artifact(model: CompactCNN, columns: tuple[int, ...], params: dict,
              scale: dict, epoch: int) -> dict:
    return {'state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            'columns': list(columns), 'feature_names': [common.FEATURE_NAMES[i] for i in columns],
            'params': params, 'epoch': epoch,
            'wide_context': model.wide_context,
            'input_mean': scale['input_mean'].tolist(),
            'input_scale': scale['input_scale'].tolist(),
            'target_mean': scale['target_mean'], 'target_scale': scale['target_scale']}


def _restore(artifact: dict, device: torch.device) -> tuple[CompactCNN, tuple[int, ...], dict]:
    columns = tuple(artifact['columns'])
    model = CompactCNN(len(columns), int(artifact['params']['filters']),
                       wide_context=artifact.get('wide_context', True)).to(device)
    model.load_state_dict(artifact['state_dict'])
    model.eval()
    scale = {'input_mean': np.asarray(artifact['input_mean'], dtype=np.float32),
             'input_scale': np.asarray(artifact['input_scale'], dtype=np.float32),
             'target_mean': float(artifact['target_mean']),
             'target_scale': float(artifact['target_scale'])}
    return model, columns, scale


def predict_view(artifact: dict, view: PatchView, batch_size: int,
                 device: torch.device, muted_global: tuple[int, ...] = ()) -> np.ndarray:
    model, columns, scale = _restore(artifact, device)
    muted = tuple(columns.index(i) for i in muted_global if i in columns)
    result = np.empty(len(view), dtype=np.float64)
    with torch.inference_mode():
        for start in range(0, len(view), batch_size):
            ids = np.arange(start, min(start + batch_size, len(view)))
            x = _tensor(view.batch(ids), columns, scale, device, muted,
                        wide_context=artifact.get('wide_context', True))
            p = model(x).reshape(-1).detach().cpu().numpy()
            result[ids] = p * scale['target_scale'] + scale['target_mean']
    if not np.isfinite(result).all():
        raise ValueError('CNN produced nonfinite predictions')
    return result


def fit_cnn(view: PatchView, columns: tuple[int, ...], scale: dict, params: dict,
            epochs: int, settings: dict, device: torch.device,
            monitor: PatchView | None = None, *,
            wide_context: bool = True) -> tuple[dict, list[dict], dict | None]:
    torch.manual_seed(settings['random_state'])
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(settings['random_state'])
    model = CompactCNN(len(columns), params['filters'],
                       wide_context=wide_context).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params['learning_rate'],
                                  weight_decay=params['weight_decay'])
    rng = np.random.default_rng(settings['random_state'])
    history = []
    best = None
    best_artifact = None
    stale = 0
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(view))
        model.train()
        loss_sum = 0.0
        for offset in range(0, len(order), settings['batch_size']):
            ids = order[offset:offset + settings['batch_size']]
            x = _tensor(view.batch(ids), columns, scale, device,
                        wide_context=wide_context)
            y = torch.from_numpy(((view.y[ids] - scale['target_mean']) /
                                  scale['target_scale']).astype(np.float32)).to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(x).reshape(-1)
            loss = torch.mean((output - y) ** 2)
            if not torch.isfinite(loss):
                raise ValueError('CNN training loss is nonfinite')
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(ids)
        row = {'epoch': epoch, 'train_loss_scaled': loss_sum / len(view),
               'optimizer_steps': math.ceil(len(view) / settings['batch_size']) * epoch,
               'elapsed_seconds': round(time.perf_counter() - started, 2)}
        if monitor is not None:
            candidate = _artifact(model, columns, params, scale, epoch)
            prediction = predict_view(candidate, monitor, settings['batch_size'], device)
            row['selection_rmse'] = common._rmse(monitor.y, prediction)
            row['selection_mae'] = common._mae(monitor.y, prediction)
            if epoch >= settings['min_epochs']:
                rank = (row['selection_rmse'], row['selection_mae'], epoch)
                if best is None or rank < best:
                    best, best_artifact, stale = rank, candidate, 0
                else:
                    stale += 1
        history.append(row)
        logging.info('CNN epoch %s/%s loss %.6f%s', epoch, epochs,
                     row['train_loss_scaled'],
                     '' if monitor is None else f" selection RMSE {row['selection_rmse']:.6f}")
        if monitor is not None and stale >= settings['patience']:
            break
    final = _artifact(model, columns, params, scale, len(history))
    return final, history, best_artifact


def _selection_views(stages: dict, banks: dict, directory: Path,
                     settings: dict) -> tuple[dict, dict]:
    path = directory / 'selection_subsample_indices.npz'
    if not path.is_file():
        selection_subsample(stages, settings['selection_samples_per_stage_year'],
                            settings['selection_seed'], directory)
    with np.load(path) as archive:
        indices = {stage: archive[stage].astype(np.int64) for stage in
                   ('ajuste', 'selecao_camadas')}
    selected = {stage: _view(banks[stage], stages[stage][1], indices[stage])
                for stage in indices}
    return selected, indices


def _search(stages: dict, selected: dict, indices: dict, settings: dict,
            device: torch.device, directory: Path) -> dict:
    trials_dir = directory / 'trials'
    trials_dir.mkdir(exist_ok=True)
    candidates = settings['candidates']
    rows, chosen = [], {}
    for arm in BASE_ARMS:
        columns = common._arm_indices(arm)
        x_fit = stages['ajuste'][0][indices['ajuste']]
        scale = _scales(x_fit, selected['ajuste'].y, columns)
        family_rows = []
        for cid, params in enumerate(candidates):
            stem = trials_dir / f'{arm}_{cid}'
            record_path, artifact_path = stem.with_suffix('.json'), stem.with_suffix('.pt')
            if record_path.is_file() and artifact_path.is_file():
                row = json.loads(record_path.read_text(encoding='utf-8'))
                if row['params'] != params or row['base_arm'] != arm:
                    raise ValueError(f'Inconsistent CNN trial checkpoint: {stem}')
            else:
                _, history, best = fit_cnn(
                    selected['ajuste'], columns, scale, params,
                    settings['max_epochs'], settings, device, selected['selecao_camadas'])
                if best is None:
                    raise ValueError('No eligible selection epoch')
                best_row = history[best['epoch'] - 1]
                torch.save(best, artifact_path)
                row = {'base_arm': arm, 'candidate_id': cid, 'params': params,
                       'selected_epoch': best['epoch'], 'actual_epochs': len(history),
                       'selection_rmse': best_row['selection_rmse'],
                       'selection_mae': best_row['selection_mae'],
                       'parameters': sum(v.numel() for v in best['state_dict'].values()),
                       'fit_samples': len(selected['ajuste']),
                       'selection_samples': len(selected['selecao_camadas']),
                       'history': history, 'artifact': str(artifact_path)}
                common._write_json(record_path, row)
                logging.info('CNN H%s %s candidate %s: epoch %s RMSE %.6f',
                             directory.name[-1], arm, cid, best['epoch'],
                             row['selection_rmse'])
            rows.append(row)
            family_rows.append(row)
        best_row = min(family_rows,
                       key=lambda r: (r['selection_rmse'], r['selection_mae'],
                                      r['parameters'], r['selected_epoch'], r['candidate_id']))
        chosen[arm] = {'params': best_row['params'], 'epochs': best_row['selected_epoch'],
                       'candidate_id': best_row['candidate_id'],
                       'artifact': best_row['artifact'],
                       'selection_rmse': best_row['selection_rmse']}
    common._write_json(directory / 'hyperparameter_choices.json', chosen)
    common._write_csv(directory / 'hyperparameter_trials.csv', [
        {k: (json.dumps(v) if isinstance(v, (dict, list)) else v)
         for k, v in row.items() if k != 'history'} for row in rows])
    common._write_csv(directory / 'hyperparameter_epoch_history.csv', [
        {'base_arm': row['base_arm'], 'candidate_id': row['candidate_id'], **point}
        for row in rows for point in row['history']])
    return chosen


def _score_mask(artifact: dict, view: PatchView, muted: tuple[int, ...],
                settings: dict, device: torch.device) -> tuple[float, float]:
    pred = predict_view(artifact, view, settings['batch_size'], device, muted)
    return common._rmse(view.y, pred), common._mae(view.y, pred)


def _choose_filters(chosen: dict, score: PatchView, settings: dict,
                    device: torch.device, directory: Path) -> dict:
    path = directory / 'feature_occlusion.json'
    if path.is_file():
        return json.loads(path.read_text(encoding='utf-8'))
    originals = torch.load(chosen['originais']['artifact'], map_location='cpu', weights_only=True)
    original_full = common._arm_indices('originais')
    original_rows = []
    for count in range(1, len(common.PHENOMENA) + 1):
        for subset in combinations(common.PHENOMENA, count):
            retained = common._arm_indices('originais', subset)
            muted = tuple(i for i in original_full if i not in retained)
            rmse, mae = _score_mask(originals, score, muted, settings, device)
            original_rows.append({'phenomena': list(subset),
                                  'features': [common.FEATURE_NAMES[i] for i in retained],
                                  'columns': list(retained), 'muted': list(muted),
                                  'n_features': len(retained),
                                  'selection_rmse': rmse, 'selection_mae': mae,
                                  'selected': False})
    original_winner = min(original_rows,
                          key=lambda r: (r['selection_rmse'], r['selection_mae'],
                                         r['n_features'], r['features']))
    original_winner['selected'] = True

    all_artifact = torch.load(chosen['todas']['artifact'], map_location='cpu', weights_only=True)
    all_columns = common._arm_indices('todas')
    groups = common._product_groups()
    baseline_rmse, baseline_mae = _score_mask(all_artifact, score, (), settings, device)
    all_rows = [{'removed_groups': [], 'features': list(common.FEATURE_NAMES),
                 'columns': list(all_columns), 'muted': [], 'n_features': 24,
                 'selection_rmse': baseline_rmse, 'selection_mae': baseline_mae,
                 'selected': False}]
    favorable = []
    for name, group in groups.items():
        rmse, mae = _score_mask(all_artifact, score, group, settings, device)
        retained = tuple(i for i in all_columns if i not in group)
        row = {'removed_groups': [name],
               'features': [common.FEATURE_NAMES[i] for i in retained],
               'columns': list(retained), 'muted': list(group),
               'n_features': len(retained), 'selection_rmse': rmse,
               'selection_mae': mae, 'selected': False}
        all_rows.append(row)
        if rmse < baseline_rmse - 1e-12:
            favorable.append(name)
    if len(favorable) > 1:
        muted = tuple(i for name in favorable for i in groups[name])
        retained = tuple(i for i in all_columns if i not in muted)
        rmse, mae = _score_mask(all_artifact, score, muted, settings, device)
        all_rows.append({'removed_groups': favorable,
                         'features': [common.FEATURE_NAMES[i] for i in retained],
                         'columns': list(retained), 'muted': list(muted),
                         'n_features': len(retained),
                         'selection_rmse': rmse, 'selection_mae': mae,
                         'selected': False})
    all_winner = min(all_rows,
                     key=lambda r: (r['selection_rmse'], r['selection_mae'],
                                    r['n_features'], r['features']))
    all_winner['selected'] = True
    result = {'method': 'zero occlusion in fitted base CNN; filtered arms refit from scratch',
              'originais_filtradas': {'chosen_columns': original_winner['columns'],
                                      'attempts': original_rows},
              'todas_filtradas': {'chosen_columns': all_winner['columns'],
                                  'attempts': all_rows}}
    common._write_json(path, result)
    common._write_csv(directory / 'feature_occlusion.csv', [
        {'arm': arm, 'features': ';'.join(row['features']),
         'muted': ';'.join(common.FEATURE_NAMES[i] for i in row['muted']),
         'removed_groups': ';'.join(row.get('removed_groups', [])),
         'n_features': row['n_features'], 'selection_rmse': row['selection_rmse'],
         'selection_mae': row['selection_mae'], 'selected': row['selected']}
        for arm in ('originais_filtradas', 'todas_filtradas')
        for row in result[arm]['attempts']])
    return result


def _train_arm(arm: str, choice: dict, columns: tuple[int, ...],
               stages: dict, banks: dict, selected: dict, indices: dict,
               threshold: float, width: int, seed: int, settings: dict,
               device: torch.device, directory: Path) -> dict:
    folder = directory / arm
    folder.mkdir(exist_ok=True)
    record_path, model_path = folder / 'metrics.json', folder / 'model.pt'
    if record_path.is_file() and model_path.is_file():
        record = json.loads(record_path.read_text(encoding='utf-8'))
        if record['feature_names'] != [common.FEATURE_NAMES[i] for i in columns] or record['hyperparameters'] != choice['params']:
            raise ValueError(f'CNN completed arm differs from current selection: {folder}')
        if common._sha256(model_path) != record['model_sha256']:
            raise ValueError(f'CNN completed arm checkpoint hash differs: {folder}')
        return record
    if arm in BASE_ARMS:
        initial = torch.load(choice['artifact'], map_location='cpu', weights_only=True)
    else:
        scale = _scales(stages['ajuste'][0][indices['ajuste']],
                        selected['ajuste'].y, columns)
        initial, _, _ = fit_cnn(selected['ajuste'], columns, scale,
                                choice['params'], choice['epochs'], settings, device)
        # No epoch choice here: it was inherited from the base arm.
    initial_pred = predict_view(initial, selected['selecao_camadas'],
                                settings['batch_size'], device)
    initial_metrics = common.metrics(selected['selecao_camadas'].y,
                                     initial_pred, threshold)

    combined_x = np.concatenate((stages['ajuste'][0], stages['selecao_camadas'][0]))
    combined_y = np.concatenate((stages['ajuste'][1], stages['selecao_camadas'][1]))
    full_scale = _scales(combined_x, combined_y, columns)
    train_view = _two_stage_view(banks, stages)
    final, history, _ = fit_cnn(train_view, columns, full_scale, choice['params'],
                                choice['epochs'], settings, device)
    torch.save(final, model_path)
    common._write_csv(folder / 'training_history.csv', history)
    train_pred = predict_view(final, train_view, settings['batch_size'], device)
    valid_view = _view(banks['validacao_modelo'], stages['validacao_modelo'][1])
    valid_pred = predict_view(final, valid_view, settings['batch_size'], device)
    validation_positions = stages['validacao_modelo'][2]
    validation_years = stages['validacao_modelo'][3]
    record = {
        'arm': arm, 'n_features': len(columns),
        'feature_names': [common.FEATURE_NAMES[i] for i in columns],
        'hyperparameters': choice['params'], 'selected_epoch': choice['epochs'],
        'actual_epochs': len(history), 'optimizer_steps': history[-1]['optimizer_steps'],
        'model_sha256': common._sha256(model_path), 'model_path': str(model_path),
        'input_scaler_mean': final['input_mean'],
        'input_scaler_scale': final['input_scale'],
        'target_scaler_mean': final['target_mean'],
        'target_scaler_scale': final['target_scale'],
        'initial_selection_metrics': initial_metrics,
        'training_metrics': common.metrics(combined_y, train_pred, threshold),
        'validation_metrics': common.metrics(valid_view.y, valid_pred, threshold),
        'validation_uncertainty': common._bootstrap_block_errors(
            valid_view.y, valid_pred, validation_positions[:, 0],
            validation_positions[:, 1], width, seed=seed),
        'validation_by_year': [
            {'target_year': int(year),
             'metrics': common.metrics(valid_view.y[validation_years == year],
                                       valid_pred[validation_years == year], threshold)}
            for year in np.unique(validation_years)],
        'selection_fit_samples': len(selected['ajuste']),
        'selection_score_samples': len(selected['selecao_camadas']),
        'training_samples': len(combined_y), 'validation_samples': len(valid_view),
    }
    common._write_json(record_path, record)
    logging.info('CNN %s: validation RMSE %.6f with %s channels',
                 arm, record['validation_metrics']['rmse'], len(columns))
    return record


def train_horizon(cfg: dict, profile: dict, zones: np.ndarray,
                  horizon: int, directory: Path, device: torch.device) -> dict:
    directory.mkdir(exist_ok=True)
    sample_path = directory / 'samples.npz'
    if not sample_path.is_file():
        shutil.copy2(cfg['sample_source'] / f'horizon_{horizon}' / 'samples.npz', sample_path)
    stages = _load_stages(sample_path)
    _check_stages(stages, zones, horizon)
    _fill_patch_bank(cfg, profile, stages, horizon, directory)
    banks = {stage: np.load(directory / f'{stage}_patches.npy', mmap_mode='r')
             for stage in STAGES}
    settings = cfg['raw']['cnn']
    selected, indices = _selection_views(stages, banks, directory, settings)
    positive = np.concatenate((stages['ajuste'][1], stages['selecao_camadas'][1]))
    positive = positive[positive > 0]
    threshold = float(np.quantile(positive,
                      cfg['raw']['evaluation']['high_target_positive_quantile']))
    chosen = _search(stages, selected, indices, settings, device, directory)
    filters = _choose_filters(chosen, selected['selecao_camadas'],
                              settings, device, directory)
    records = []
    for arm in ARMS:
        base = 'originais' if arm == 'originais_filtradas' else (
               'todas' if arm == 'todas_filtradas' else arm)
        columns = tuple(filters[arm]['chosen_columns']) if arm in filters else common._arm_indices(arm)
        records.append(_train_arm(arm, chosen[base], columns, stages, banks,
                                  selected, indices, threshold, profile['width'],
                                  cfg['raw']['sampling']['seed'] + horizon,
                                  settings, device, directory))
    winner = min(records, key=lambda r: (r['validation_metrics']['rmse'],
                                          r['validation_metrics']['mae'],
                                          r['n_features'], r['arm']))
    selected_winner = {'horizon': horizon, 'arm': winner['arm'],
                       'n_features': winner['n_features'],
                       'feature_names': winner['feature_names'],
                       'validation_metrics': winner['validation_metrics'],
                       'high_target_threshold': threshold,
                       'training_target_mean': float(np.mean(np.concatenate(
                           (stages['ajuste'][1], stages['selecao_camadas'][1])))),
                       'model_path': winner['model_path']}
    common._write_json(directory / 'winner.json', selected_winner)
    common._write_csv(directory / 'comparison.csv', [
        {'arm': r['arm'], 'n_features': r['n_features'],
         'selection_rmse': r['initial_selection_metrics']['rmse'],
         'training_rmse': r['training_metrics']['rmse'],
         'validation_rmse': r['validation_metrics']['rmse'],
         'validation_mae': r['validation_metrics']['mae'],
         'winner': r['arm'] == winner['arm']} for r in records])
    return {'winner': selected_winner, 'records': records}


def _dense_predict(artifact: dict, raw: np.ndarray, height: int, width: int,
                   device: torch.device) -> np.ndarray:
    model, columns, scale = _restore(artifact, device)
    x = (raw[list(columns)] - scale['input_mean'][:, None, None]) / scale['input_scale'][:, None, None]
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    with torch.inference_mode():
        tensor = torch.from_numpy(np.ascontiguousarray(x[None])).to(device)
        predicted = model.forward_dense(tensor, height, width).squeeze(0).cpu().numpy()
    return predicted * scale['target_scale'] + scale['target_mean']


def evaluate_external(cfg: dict, selected: dict, horizon: int, directory: Path,
                      target_years: list[int], device: torch.device) -> list[dict]:
    """Open external targets only after H0 and H1 winners are frozen."""
    artifact = torch.load(selected['model_path'], map_location='cpu', weights_only=True)
    if artifact['feature_names'] != selected['feature_names']:
        raise ValueError('CNN winner channels differ from model artifact')
    radius = WIDE_RADIUS if artifact.get('wide_context', True) else RADIUS
    records = []
    for target_year in target_years:
        record_path = directory / f'external_target_{target_year}_metrics.json'
        if record_path.is_file():
            records.append(json.loads(record_path.read_text(encoding='utf-8')))
            continue
        input_year = target_year - horizon
        profile = common._external_profile(cfg, input_year)
        target_path = common._target_path(cfg, 'teste_final', target_year)
        out_path = directory / f'external_prediction_target_{target_year}.tif'
        output_profile = profile.copy()
        output_profile.update(driver='GTiff', count=1, dtype='float32',
                              nodata=float('nan'), tiled=True,
                              blockxsize=256, blockysize=256, compress='deflate')
        target_parts, pred_parts, row_parts, col_parts = [], [], [], []
        tile_size = cfg['raw']['cnn']['tile_size']
        cache = _aligned_cache(cfg, 'teste_final', input_year, profile)
        with AlignedInputs(cfg, 'teste_final', input_year, profile, cache) as reader, \
             rasterio.open(target_path) as target_source, \
             rasterio.open(out_path, 'w', **output_profile) as output:
            if target_source.nodata == 0:
                raise ValueError('External target declares zero as NoData')
            with WarpedVRT(target_source, crs=profile['crs'],
                           transform=profile['transform'], width=profile['width'],
                           height=profile['height'], dtype='float32',
                           nodata=float('nan'), resampling=Resampling.nearest) as target:
                tile_count = 0
                for row in range(0, profile['height'], tile_size):
                    for col in range(0, profile['width'], tile_size):
                        height = min(tile_size, profile['height'] - row)
                        width = min(tile_size, profile['width'] - col)
                        window = Window(col, row, width, height)
                        raw = reader.tile(row, col, height, width, radius=radius)
                        center = raw[:, radius:radius + height,
                                     radius:radius + width]
                        y = target.read(1, window=window, masked=True).astype('float32').filled(np.nan)
                        valid = np.isfinite(center).all(axis=0) & np.isfinite(y)
                        predicted = np.full((height, width), np.nan, dtype=np.float32)
                        if np.any(valid):
                            p = _dense_predict(artifact, raw, height, width, device)
                            if not np.isfinite(p[valid]).all():
                                raise ValueError('CNN external prediction contains nonfinite values')
                            predicted[valid] = p[valid]
                            rr, cc = np.where(valid)
                            target_parts.append(y[valid].astype(np.float64))
                            pred_parts.append(p[valid].astype(np.float64))
                            row_parts.append(rr.astype(np.int32) + row)
                            col_parts.append(cc.astype(np.int32) + col)
                        output.write(predicted, 1, window=window)
                        tile_count += 1
                        if tile_count % 100 == 0:
                            logging.info('CNN external H%s %s: %s tiles', horizon,
                                         target_year, tile_count)
        if not target_parts:
            raise ValueError('CNN external evaluation has no common valid pixels')
        y = np.concatenate(target_parts)
        p = np.concatenate(pred_parts)
        rows = np.concatenate(row_parts)
        cols = np.concatenate(col_parts)
        observed = common.metrics(y, p, selected['high_target_threshold'])
        zero = common.metrics(y, np.zeros_like(y), subsets=False)
        mean = common.metrics(y, np.full_like(y, selected['training_target_mean']), subsets=False)
        persistence = common._persistence(cfg, target_year, profile, (rows, cols), y)
        uncertainty = common._bootstrap_block_errors(
            y, p, rows, cols, profile['width'],
            seed=cfg['raw']['sampling']['seed'] + horizon * 10 + target_year)
        record = {'horizon': horizon, 'input_year': input_year,
                  'target_year': target_year, 'prediction': str(out_path),
                  'target_source': str(target_path),
                  'target_sha256': common._sha256(target_path),
                  'common_pixels': len(y),
                  'evaluated_fraction': float(len(y) / (profile['height'] * profile['width'])),
                  'metrics': observed, 'reference_zero': zero,
                  'reference_training_mean': mean,
                  'reference_persistence': persistence, 'uncertainty': uncertainty}
        common._write_json(record_path, record)
        records.append(record)
        logging.info('CNN external H%s %s: RMSE %.6f MAE %.6f n=%s',
                     horizon, target_year, observed['rmse'], observed['mae'], len(y))
    return records


def _plot_histories(run_dir: Path) -> None:
    folder = run_dir / 'graficos_treinamento'
    folder.mkdir(exist_ok=True)
    for horizon in (0, 1):
        source = run_dir / f'horizon_{horizon}'
        choices = json.loads((source / 'hyperparameter_choices.json').read_text(encoding='utf-8'))
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for arm in BASE_ARMS:
            cid = choices[arm]['candidate_id']
            trial = json.loads((source / 'trials' / f'{arm}_{cid}.json').read_text(encoding='utf-8'))
            epochs = [point['epoch'] for point in trial['history']]
            axes[0].plot(epochs, [point['selection_rmse'] for point in trial['history']],
                         label=arm)
            axes[1].plot(epochs, [point['train_loss_scaled'] for point in trial['history']],
                         label=arm)
        axes[0].set(title=f'H{horizon}: RMSE da seleção por época',
                    xlabel='Época', ylabel='RMSE (unidade do target)')
        axes[1].set(title=f'H{horizon}: perda no ajuste de busca',
                    xlabel='Época', ylabel='MSE padronizado')
        for ax in axes:
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(folder / f'h{horizon}_busca.png', dpi=300)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(10, 5))
        for arm in ARMS:
            with (source / arm / 'training_history.csv').open(encoding='utf-8-sig', newline='') as handle:
                rows = list(csv.DictReader(handle))
            ax.plot([int(row['epoch']) for row in rows],
                    [float(row['train_loss_scaled']) for row in rows], label=arm)
        ax.set(title=f'H{horizon}: perda dos seis reajustes finais',
               xlabel='Época', ylabel='MSE padronizado')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(folder / f'h{horizon}_reajuste.png', dpi=300)
        plt.close(fig)


def _fmt(value, digits: int = 4) -> str:
    return '—' if value is None else f'{value:.{digits}f}'


def rebuild_report(run_dir: Path) -> None:
    from plot_enriched_linear_external import render_one

    run_dir = run_dir.resolve()
    status_path = run_dir / 'run.json'
    status = json.loads(status_path.read_text(encoding='utf-8'))
    if status.get('kind') != 'enriched_cnn' or not status.get('final_external_test_used'):
        raise ValueError('Expected an externally evaluated CNN run')
    figures = []
    for h in (0, 1):
        for path in sorted((run_dir / f'horizon_{h}').glob('external_target_*_metrics.json')):
            figures.append(render_one(json.loads(path.read_text(encoding='utf-8')), run_dir))
    supplement = run_dir / 'supplemental_H1_2021'
    supplemental_record = json.loads((supplement / 'external_target_2021_metrics.json').read_text(encoding='utf-8'))
    figures.append(render_one(supplemental_record, supplement))
    common._write_json(run_dir / 'figuras_teste_externo.json', figures)
    _plot_histories(run_dir)
    arms_csv, external_csv = [], []
    lines = [
        '# Relatório final — experimento 08, CNN espacial de duas escalas', '',
        f'Execução: `{run_dir}`. Um ramo observa vizinhança local 5×5; o outro '
        'amostra 5×5 até ±41 km em cada eixo. Ambos usam somente inputs. '
        'O alvo é intensidade de registros; zero não significa impossibilidade de ocorrência.', '',
        '## Protocolo e leitura', '',
        'H0 usa entrada e target do mesmo ano; H1 usa entrada do ano anterior. '
        'São os mesmos centros amostrados do experimento MLP, com 24 canais possíveis '
        'e seis braços por horizonte. Hiperparâmetros, épocas e ocultação de canais '
        'foram escolhidos na seleção interna; os modelos filtrados foram reajustados '
        'do zero. A validação escolheu somente o braço vencedor. Os targets externos '
        'foram abertos depois de congelar H0 e H1. A região externa já foi estudada '
        'antes e a comparação entre modelos é exploratória.', '',
        'A seleção de filtros por ocultação zera canais em um modelo base já treinado. '
        'Ela não é importância causal nem equivale a treinar cada combinação. '
        'O halo de ±41 km pode atravessar zonas de validação com o mesmo contexto de '
        'covariáveis; o buffer espacial de três pixels não garante contextos disjuntos. '
        'Nenhum target vizinho entra na CNN.', '',
        '## Doze braços treinados e validados', '',
        '| Horizonte | Braço | Canais | Épocas | Passos AdamW | RMSE seleção | RMSE treino | RMSE validação | MAE validação | Venceu |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---|',
    ]
    for h in (0, 1):
        folder = run_dir / f'horizon_{h}'
        winner = json.loads((folder / 'winner.json').read_text(encoding='utf-8'))
        for arm in ARMS:
            record = json.loads((folder / arm / 'metrics.json').read_text(encoding='utf-8'))
            arms_csv.append({'horizon': h, 'arm': arm,
                             'features': ';'.join(record['feature_names']),
                             'n_features': record['n_features'],
                             'selected_epoch': record['selected_epoch'],
                             'optimizer_steps': record['optimizer_steps'],
                             'hyperparameters': json.dumps(record['hyperparameters']),
                             'selection_rmse': record['initial_selection_metrics']['rmse'],
                             'training_rmse': record['training_metrics']['rmse'],
                             'validation_rmse': record['validation_metrics']['rmse'],
                             'validation_mae': record['validation_metrics']['mae'],
                             'winner': arm == winner['arm']})
            lines.append(f"| H{h} | {arm} | {record['n_features']} | {record['selected_epoch']} | "
                         f"{record['optimizer_steps']} | {_fmt(record['initial_selection_metrics']['rmse'])} | "
                         f"{_fmt(record['training_metrics']['rmse'])} | "
                         f"{_fmt(record['validation_metrics']['rmse'])} | "
                         f"{_fmt(record['validation_metrics']['mae'])} | "
                         f"{'sim' if arm == winner['arm'] else 'não'} |")
    lines += ['', 'Os CSVs `resultados_12_treinamentos.csv`, `feature_occlusion.csv`, '
              '`horizon_*/hyperparameter_trials.csv` e `horizon_*/hyperparameter_epoch_history.csv` '
              'contêm todas as configurações, épocas e testes de ocultação. '
              'Cada braço também guarda `metrics.json`, `training_history.csv` e `model.pt`.', '']
    common._write_csv(run_dir / 'resultados_12_treinamentos.csv', arms_csv)
    lines += ['## Vencedores, canais e teste externo', '']
    for h in (0, 1):
        folder = run_dir / f'horizon_{h}'
        winner = json.loads((folder / 'winner.json').read_text(encoding='utf-8'))
        record = json.loads((folder / winner['arm'] / 'metrics.json').read_text(encoding='utf-8'))
        lines += [f"### H{h}: {winner['arm']} ({winner['n_features']} canais)", '',
                  'Canais: ' + ', '.join(winner['feature_names']) + '.', '',
                  f"Parâmetros: `{json.dumps(record['hyperparameters'])}`; "
                  f"épocas: {record['selected_epoch']}; passos: {record['optimizer_steps']}; "
                  f"RMSE validação: {_fmt(record['validation_metrics']['rmse'])}; "
                  f"MAE validação: {_fmt(record['validation_metrics']['mae'])}.", '']
    lines += [
        'Cada par externo usa a mesma máscara para target e previsão; métricas são '
        'calculadas na unidade original do raster. A persistência usa target histórico '
        'e portanto informação adicional. H1 2020→2021 é avaliação complementar, '
        'após congelamento.', '',
        '| Horizonte | Entrada→target | Pixels | RMSE | MAE | R² | Viés | NMAD | P90 abs. | Captura 10% | RMSE zero | RMSE persistência |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    external = []
    for h in (0, 1):
        for path in sorted((run_dir / f'horizon_{h}').glob('external_target_*_metrics.json')):
            external.append(json.loads(path.read_text(encoding='utf-8')))
    external.append(supplemental_record)
    for item in external:
        metric = item['metrics']
        persistence = item['reference_persistence']
        lines.append(f"| H{item['horizon']} | {item['input_year']}→{item['target_year']}"
                     f"{' (complementar)' if item is supplemental_record else ''} | "
                     f"{item['common_pixels']:,} | {_fmt(metric['rmse'])} | {_fmt(metric['mae'])} | "
                     f"{_fmt(metric['r2'])} | {_fmt(metric['bias'])} | {_fmt(metric['nmad'])} | "
                     f"{_fmt(metric['p90_absolute_error'])} | "
                     f"{_fmt(100 * metric['capture_10pct'], 2)}% | "
                     f"{_fmt(item['reference_zero']['rmse'])} | "
                     f"{_fmt(None if persistence is None else persistence['metrics']['rmse'])} |")
        external_csv.append({
            'horizon': item['horizon'], 'input_year': item['input_year'],
            'target_year': item['target_year'], 'supplemental': item is supplemental_record,
            'common_pixels': item['common_pixels'], 'prediction': item['prediction'],
            'target_sha256': item['target_sha256'],
            'metrics': json.dumps(metric), 'zero': json.dumps(item['reference_zero']),
            'mean': json.dumps(item['reference_training_mean']),
            'persistence': json.dumps(persistence),
            'uncertainty': json.dumps(item['uncertainty'])})
    common._write_csv(run_dir / 'teste_externo_metricas.csv', external_csv)
    lines += ['', 'Os JSONs de cada par contêm captura top 5/10/20%, erro em target '
              'zero/positivo/alto, referências e intervalos espaciais exploratórios. '
              'Previsões negativas não foram truncadas. A métrica CE90 posicional '
              'não se aplica a erros de intensidade.', '',
              '## Curvas de treinamento', '',
              'As curvas de busca são medidas a cada época; as curvas finais são '
              'perda de ajuste, pois a validação não escolhe época.', '',
              '![Busca H0](graficos_treinamento/h0_busca.png)', '',
              '![Reajuste H0](graficos_treinamento/h0_reajuste.png)', '',
              '![Busca H1](graficos_treinamento/h1_busca.png)', '',
              '![Reajuste H1](graficos_treinamento/h1_reajuste.png)', '',
              '## Mapas externos: target original e previsão', '',
              'Cada imagem usa mesma máscara e escala nos dois painéis. PNG 600 dpi; '
              'os GeoTIFFs preservam a resolução e georreferência.', '']
    for item in external:
        h, iy, ty = item['horizon'], item['input_year'], item['target_year']
        name = f'teste_externo_H{h}_{iy}_{ty}.png'
        path = f'supplemental_H1_2021/{name}' if item is supplemental_record else name
        lines += [f'**H{h} · {iy}→{ty}**', '',
                  f'![Target original e previsão CNN H{h} {iy}→{ty}]({path})', '',
                  f"[GeoTIFF da previsão]({Path(item['prediction']).relative_to(run_dir).as_posix()})", '']
    lines += [
        '## Reprodutibilidade e fontes', '',
        f"Fonte das posições de desenvolvimento: `{status['sample_source']}`; "
        'hashes dos dois arquivos em `sample_source.json`. Catálogo, versões, '
        'configuração, código e protocolos estão congelados neste run. '
        'Os dados externos não orientaram a busca.', '',
        '- [PyTorch Conv2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html): operações convolucionais.',
        '- [PyTorch AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html): otimizador.',
        '- [Valavi et al. (2019)](https://doi.org/10.1111/2041-210X.13107): separação espacial.',
        '- [Protocolo CNN](protocols/EXPERIMENTO_08_CNN.md) e [protocolo comum](protocols/EXPERIMENTOS_ENRIQUECIDOS_2026.md): canais, métricas e limites.', '',
    ]
    (run_dir / 'RELATORIO.md').write_text('\n'.join(lines), encoding='utf-8')
    status['status'] = 'completed'
    common._write_json(status_path, status)
    print(run_dir / 'RELATORIO.md')


def _snapshot(cfg: dict, run_dir: Path) -> None:
    shutil.copy2(cfg['path'], run_dir / 'config.yaml')
    project = Path(__file__).parents[2]
    sources = ('run_enriched_cnn.py', 'src/services/enriched_cnn.py',
               'src/services/enriched_linear.py', 'src/services/enriched_hgb.py',
               'src/services/temporal_service.py', 'src/splitting/spatial_split.py',
               'src/geospatial/raster_io.py', 'plot_enriched_linear_external.py',
               'generate_gradient_inputs.py', 'generate_graph_inputs.py')
    source_records = []
    for relative in sources:
        source = project / relative
        dest = run_dir / 'source_code' / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        source_records.append({'path': relative, 'sha256': common._sha256(source)})
    common._write_json(run_dir / 'source_files.json', source_records)
    for relative in ('protocols/EXPERIMENTOS_ENRIQUECIDOS_2026.md',
                     'protocols/EXPERIMENTO_08_CNN.md',
                     'docs/GRADIENT_INPUTS.md', 'docs/GRAPH_INPUTS.md'):
        source = project / relative
        dest = run_dir / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
    packages = {name: importlib.metadata.version(name) for name in
                ('numpy', 'torch', 'rasterio', 'matplotlib', 'scikit-learn',
                 'PyYAML', 'psutil')}
    common._write_json(run_dir / 'environment.json',
                       {'python': sys.version, 'platform': platform.platform(),
                        'packages': packages, 'cuda_available': torch.cuda.is_available(),
                        'cuda_device': torch.cuda.get_device_name(0)
                        if torch.cuda.is_available() else None})


def run(config_path: Path, resume: Path | None = None) -> None:
    cfg = load_config(config_path)
    cfg['output'].mkdir(parents=True, exist_ok=True)
    if resume is None:
        run_dir = cfg['output'] / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        run_dir.mkdir(exist_ok=False)
        status = {'kind': 'enriched_cnn', 'status': 'running',
                  'started': datetime.now(timezone.utc).isoformat(),
                  'run': str(run_dir), 'sample_source': str(cfg['sample_source']),
                  'final_external_test_used': False}
        common._write_json(run_dir / 'run.json', status)
    else:
        run_dir = resume.resolve()
        if not run_dir.is_dir() or run_dir.parent != cfg['output'].resolve():
            raise ValueError('Resume path must be an unfinished CNN run')
        status = json.loads((run_dir / 'run.json').read_text(encoding='utf-8'))
        if status.get('kind') != 'enriched_cnn' or status.get('status') == 'completed':
            raise ValueError('Only an unfinished CNN run can be resumed')
        if (run_dir / 'config.yaml').read_bytes() != cfg['path'].read_bytes():
            raise ValueError('CNN config changed since run began')
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
    device = _device(cfg['raw']['cnn'])
    try:
        cfg['_aligned_cache'] = run_dir / 'aligned_inputs'
        if not (run_dir / 'source_files.json').is_file():
            _snapshot(cfg, run_dir)
        sample_info_path = run_dir / 'sample_source.json'
        if not sample_info_path.is_file():
            hashes = {f'H{h}': common._sha256(
                cfg['sample_source'] / f'horizon_{h}' / 'samples.npz') for h in (0, 1)}
            source_audit = json.loads((cfg['sample_source'] / 'input_audit.json').read_text(encoding='utf-8'))
            current = common.audit_catalog(cfg, verify_hashes=False)
            if [(x['path'], x['sha256']) for x in source_audit['recorded_files']] != [
                    (x['path'], x['sha256']) for x in current['recorded_files']]:
                raise ValueError('CNN raster manifest differs from verified MLP source')
            common._write_json(run_dir / 'input_audit.json', current)
            common._write_json(sample_info_path,
                               {'source': str(cfg['sample_source']),
                                'sha256': hashes, 'manifest_matches_verified_source': True})
        sample_info = json.loads(sample_info_path.read_text(encoding='utf-8'))
        for h in (0, 1):
            if common._sha256(cfg['sample_source'] / f'horizon_{h}' / 'samples.npz') != sample_info['sha256'][f'H{h}']:
                raise ValueError('Frozen sample source changed')
        profile = common._profile(cfg['reference'])
        zones = split_grid((profile['height'], profile['width']), cfg['raw']['split'])
        if not (run_dir / 'spatial_zones.tif').is_file():
            write_raster(run_dir / 'spatial_zones.tif', zones, profile, categorical=True)
        device = _device(cfg['raw']['cnn'])
        logging.info('CNN two scales, wide radius %s pixels / %.1f km; device %s',
                     WIDE_RADIUS, WIDE_RADIUS * abs(profile['transform'].a) / 1000, device)
        torch.set_num_threads(cfg['raw']['runtime']['cpu_threads'])
        if device.type == 'cuda':
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
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
                              cfg['raw']['evaluation'][f'final_h{horizon}_target_years'], device)
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
        logging.warning('CNN interrupted; resume with --resume %s', run_dir)
        raise SystemExit(130)
    except Exception as exc:
        status['status'] = 'failed'
        status['failed'] = datetime.now(timezone.utc).isoformat()
        status['error'] = repr(exc)
        common._write_json(run_dir / 'run.json', status)
        logging.exception('CNN run failed; partial artifacts kept at %s', run_dir)
        raise
    finally:
        logger.removeHandler(file_handler)
        logger.removeHandler(stream_handler)
        file_handler.close()
        stream_handler.close()
