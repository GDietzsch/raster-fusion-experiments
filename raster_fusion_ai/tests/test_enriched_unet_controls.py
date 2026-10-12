"""Checks for paired controls, resume, independent validation and reporting."""
from pathlib import Path
import json

import numpy as np
import pytest
import torch

from src.services import enriched_unet as u
from src.services import enriched_unet_controls as c
from src.services import enriched_cnn as cnn
from src.services import enriched_linear as common


def config():
    return {'raw': {'seeds': [4210, 4211, 4212], 'tile_seed': 4210,
                    'epochs': 2, 'training_tiles_per_year': 2,
                    'monitor_tiles_per_year': 1, 'context_depth': 3},
            'base': {'raw': {'unet': {'min_epochs': 1, 'patience': 1}}},
            'winners': {h: {'columns': list(common._arm_indices('gpa')) if h else [1, 4, 5, 12, 13, 16, 17, 19],
                            'params': {'initial_filters': 2, 'learning_rate': .001, 'weight_decay': .01},
                            'high_target_threshold': 1.5,
                            'feature_names': [common.FEATURE_NAMES[i] for i in (common._arm_indices('gpa') if h else [1, 4, 5, 12, 13, 16, 17, 19])]}
                        for h in (0, 1)}}


class Cache:
    def patch(self, year, row, col, columns, scale, muted=()):
        yy, xx = np.indices((u.CORE + 2*u.HALO, u.CORE + 2*u.HALO))
        result = np.stack([np.sin((yy + row) / 13 + i) + np.cos((xx + col) / 17 + i)
                           for i in columns]).astype('float32')
        if muted:
            result[list(muted)] = 0
        return result


def data():
    rng = np.random.default_rng(12)
    stages = {}
    for name in u.STAGES:
        x = rng.normal(size=(24, 24)).astype('float32')
        positions = np.array([[row, col] for row in (3, 10, 35, 42, 67, 74)
                              for col in (3, 10, 35, 42)], dtype='int32')
        stages[name] = (x, np.abs(x[:, 0]), positions, np.full(24, 2021, dtype='int16'))
    return stages


def test_plan_has_paired_seeds_shared_references_and_57_external_pairs():
    jobs = c.job_plan(config())
    assert len(jobs) == len({j['id'] for j in jobs}) == 24
    assert sum(3 if j['horizon'] == 0 else 2 for j in jobs) == 57
    assert len([j for j in jobs if c.belongs(j, '10B')]) == 12
    assert len([j for j in jobs if c.belongs(j, '10C')]) == 9
    assert {len(j['columns']) for j in jobs if j['variant'] == 'magnitude'} == {4}
    assert {len(j['columns']) for j in jobs if j['variant'] == 'direction'} == {8}
    assert [j['experiment'] for j in jobs] == sorted(j['experiment'] for j in jobs)


def test_fixed_tile_plan_shared_across_seeds_and_paired_first_epoch(monkeypatch):
    monkeypatch.setattr(u, 'CORE', 32)
    cfg = config()
    stage = data()['ajuste']
    groups = u._stage_groups(stage, 0)
    fixed = {'schedule': 'fixed', 'seed': 4210}
    first = c.tile_plan(stage, groups, cfg, fixed, groups)
    second = c.tile_plan(stage, groups, cfg, dict(fixed, seed=4211), groups)
    rotated = c.tile_plan(stage, groups, cfg, {'schedule': 'rotating'}, groups)
    assert first == second
    assert first['epoch_keys'][0] == first['epoch_keys'][1] == rotated['epoch_keys'][0]
    assert rotated['epoch_keys'][0] != rotated['epoch_keys'][1]
    changed_target = (stage[0], stage[1] * 100, stage[2], stage[3])
    assert c.tile_plan(changed_target, groups, cfg, fixed, groups)['epoch_keys'] == first['epoch_keys']


def test_monitor_only_keeps_last_epoch_and_exact_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(u, 'CORE', 32)
    monkeypatch.setattr(u, 'HALO', 16)
    torch.set_num_threads(1)
    stage = data()['ajuste']
    groups = u._stage_groups(stage, 0)
    columns = (0, 1)
    scale = cnn._scales(stage[0], stage[1], columns)
    params = {'initial_filters': 2, 'model_depth': 3, 'learning_rate': .001, 'weight_decay': .01}
    settings = {'random_state': 12, 'tile_seed': 42, 'tile_schedule': 'fixed',
                'training_tiles_per_year': 2, 'min_epochs': 1, 'patience': 1, 'monitor_only': True}
    keys = u._choose_keys(groups, 1, 1, 42)
    monitor = (stage, groups, Cache(), keys)
    # A worsening diagnostic must neither early-stop nor select the first epoch.
    monkeypatch.setattr(u, '_score', lambda artifact, *args, **kwargs: (float(artifact['epoch']), 1.))
    full, history = u._fit(stage, groups, Cache(), columns, scale, params, 3, settings,
                           torch.device('cpu'), tmp_path / 'full', monitor)
    assert full['epoch'] == len(history) == 3
    assert full['model_config']['depth'] == 3
    original = u._save_torch
    def stop(path, value):
        original(path, value)
        if path.name == 'checkpoint.pt':
            raise KeyboardInterrupt
    monkeypatch.setattr(u, '_save_torch', stop)
    with pytest.raises(KeyboardInterrupt):
        u._fit(stage, groups, Cache(), columns, scale, params, 3, settings,
               torch.device('cpu'), tmp_path / 'resume', monitor)
    monkeypatch.setattr(u, '_save_torch', original)
    resumed, history = u._fit(stage, groups, Cache(), columns, scale, params, 3, settings,
                              torch.device('cpu'), tmp_path / 'resume', monitor)
    assert resumed['epoch'] == 3
    for name, tensor in full['state_dict'].items():
        torch.testing.assert_close(tensor, resumed['state_dict'][name], rtol=0, atol=0)


def test_training_and_report_preserve_all_seeds_and_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(u, 'CORE', 32)
    monkeypatch.setattr(u, 'HALO', 16)
    torch.set_num_threads(1)
    cfg = config()
    jobs = [j for j in c.job_plan(cfg) if j['experiment'] == '10A' and j['horizon'] == 1]
    # Synthetic miniature has one initialization but both paired schedules.
    jobs = [dict(j, params=dict(j['params'], model_depth=2)) for j in jobs if j['seed'] == 4210]
    c.write_json(tmp_path / 'run.json', {'status': 'running'})
    c.write_json(tmp_path / 'plan.json', jobs)
    c.write_json(tmp_path / 'source_winners.json', cfg['winners'])
    for job in jobs:
        folder = tmp_path / 'jobs' / job['id']
        record = c.train_job(cfg, job, data(), {'width': 96}, Cache(), folder, torch.device('cpu'))
        assert record['epochs'] == 2
        assert record['validation_metrics']['n'] == 24
        assert record['training_metrics']['n'] == 48
        assert record['actual_training_centers'] <= 48
        assert 'monitor_validation_rmse' in (folder / 'training_history.csv').read_text()
        assert c.train_job(cfg, job, data(), {'width': 96}, Cache(), folder, torch.device('cpu')) == record
        metric = record['validation_metrics']
        c.write_json(folder / 'external' / 'external_target_2022_metrics.json', {
            'input_year': 2021, 'target_year': 2022, 'common_pixels': 24,
            'metrics': metric, 'reference_zero': metric, 'reference_training_mean': metric,
            'reference_persistence': {'metrics': metric}})
    c.rebuild_reports(tmp_path)
    doc = (tmp_path / '10A' / 'RELATORIO.md').read_text(encoding='utf-8')
    assert 'rotating' in doc and 'fixed' in doc and 'RMSE persistência' in doc
    assert 'Ronneberger' in doc and 'curvas.png' in doc
    assert (tmp_path / '10A' / 'paired_validation_deltas.csv').is_file()
    assert (tmp_path / '10C' / 'RELATORIO.md').is_file()  # shared GPA reference
    assert not list(tmp_path.rglob('*.tmp'))


def test_source_fingerprint_rejects_mixed_resume(tmp_path):
    cfg = {'output': tmp_path}
    folder = tmp_path / 'run'
    c.write_json(folder / 'run.json', {'kind': c.KIND, 'requested_experiment': 'all', 'status': 'failed'})
    c.write_json(folder / 'fingerprint.json', {'code': 'before'})
    assert c.choose_run(cfg, 'all', None, False, {'code': 'before'}) == folder
    with pytest.raises(ValueError, match='changed'):
        c.choose_run(cfg, 'all', folder, False, {'code': 'after'})


def test_external_evaluations_start_only_after_every_training(tmp_path, monkeypatch):
    cfg = config()
    cfg.update(output=tmp_path, source=tmp_path)
    cfg['base'].update(sample_source=tmp_path, reference=tmp_path / 'reference.tif')
    cfg['base']['raw'].update(runtime={'cpu_threads': 1}, split={})
    monkeypatch.setattr(c, 'load_config', lambda path: cfg)
    monkeypatch.setattr(c, 'fingerprint', lambda cfg: {})
    monkeypatch.setattr(c, 'choose_run', lambda *args: tmp_path)
    c.write_json(tmp_path / 'run.json', {'status': 'prepared'})
    c.write_json(tmp_path / 'input_audit.json', {'recorded_files': []})
    monkeypatch.setattr(common, 'audit_catalog', lambda *args, **kwargs: {'recorded_files': []})
    monkeypatch.setattr(common, '_profile', lambda path: {'width': 96, 'height': 96})
    monkeypatch.setattr(c, 'split_grid', lambda *args: np.zeros((96, 96)))
    monkeypatch.setattr(cnn, '_device', lambda cfg: torch.device('cpu'))
    monkeypatch.setattr(cnn, '_verify_reused_patch_bank', lambda *args: None)
    monkeypatch.setattr(cnn, '_load_stages', lambda *args: data())
    monkeypatch.setattr(cnn, '_check_stages', lambda *args: None)
    monkeypatch.setattr(u, 'InputCache', lambda *args: Cache())
    monkeypatch.setattr(c, 'rebuild_reports', lambda *args: None)
    events = []
    def train(cfg, job, stages, profile, cache, folder, device):
        events.append('train')
        c.write_json(folder / 'metrics.json', job)
    def external(cfg, record, folder, device):
        assert events.count('train') == 24
        events.append('external')
    monkeypatch.setattr(c, 'train_job', train)
    monkeypatch.setattr(c, 'external_job', external)
    c.run(tmp_path / 'cfg')
    assert events == ['train'] * 24 + ['external'] * 24
    assert c.read_json(tmp_path / 'run.json')['status'] == 'completed'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU seam verification')
def test_shallow_context_keeps_tile_seams_consistent():
    torch.manual_seed(14)
    model = u.UNet(3, u._model_config(8, 3)).cuda().eval()
    raster = torch.randn(1, 3, 960, 704, device='cuda')
    with torch.inference_mode():
        full = model(raster)[0, 0, 224:736, 224:480]
        tiled = torch.cat([model(raster[:, :, row:row + 704])[0, 0, 224:480, 224:480]
                           for row in (0, 256)])
    torch.testing.assert_close(full, tiled, rtol=0, atol=1e-4)
