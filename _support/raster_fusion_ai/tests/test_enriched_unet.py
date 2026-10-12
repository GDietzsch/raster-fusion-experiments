"""Focused checks for the sparse-label dense U-Net experiment."""
import json
import numpy as np
import pytest
import torch
import yaml

from src.image_pipeline.models.unet import UNet
from src.services import enriched_cnn as cnn
from src.services import enriched_unet as unet


class TinyCache:
    def patch(self, year, row, col, columns, scale, muted=()):
        yy, xx = np.indices((unet.CORE + 2 * unet.HALO,
                             unet.CORE + 2 * unet.HALO))
        output = np.stack([np.sin((yy + row + year) / 13 + index) +
                           np.cos((xx + col) / 17 + index)
                           for index in columns]).astype(np.float32)
        if muted:
            output[list(muted)] = 0
        return output


@pytest.mark.skipif(not torch.cuda.is_available(), reason='full-scale seam test needs a GPU')
def test_224_pixel_halo_matches_full_grid_across_tile_seam():
    torch.manual_seed(16)
    model = UNet(3, unet._model_config(8)).cuda().eval()
    raster = torch.randn(1, 3, 960, 704, device='cuda')
    with torch.inference_mode():
        full = model(raster)[0, 0, 224:736, 224:480]
        tiled = torch.cat([
            model(raster[:, :, row:row + 704])[0, 0, 224:480, 224:480]
            for row in (0, 256)], dim=0)
    assert torch.max(torch.abs(full - tiled)).item() < 1e-4


def test_rotating_tiles_cover_every_key_without_using_target_values():
    groups = {(2020, index, 0): np.array([index]) for index in range(5)}
    first = unet._choose_keys(groups, 2, 1, 42)
    rounds = [unet._choose_keys(groups, 2, epoch, 42) for epoch in range(1, 4)]
    assert first == rounds[0]
    assert len(set(key for round_keys in rounds for key in round_keys)) == 5


@pytest.mark.parametrize('residual', [False, True, 'inception'])
def test_sparse_label_fit_predicts_only_frozen_centers(tmp_path, monkeypatch, residual):
    monkeypatch.setattr(unet, 'CORE', 32)
    monkeypatch.setattr(unet, 'HALO', 16)
    monkeypatch.setattr(unet, 'DEPTH', 2)
    torch.set_num_threads(1)
    rng = np.random.default_rng(14)
    positions = np.column_stack((rng.integers(0, 64, size=24),
                                 rng.integers(0, 64, size=24))).astype(np.int32)
    x = rng.normal(size=(24, 24)).astype(np.float32)
    y = np.abs(x[:, 0]).astype(np.float64)
    stage = (x, y, positions, np.full(24, 2020, dtype=np.int16))
    groups = unet._stage_groups(stage, 0)
    columns = (0, 1)
    scale = cnn._scales(x, y, columns)
    params = {'initial_filters': 2, 'learning_rate': .001, 'weight_decay': .01}
    if residual == 'inception':
        params['block_type'] = 'inception'
    elif residual:
        params['residual_blocks'] = True
    settings = {'training_tiles_per_year': 2, 'selection_tiles_per_year': 2,
                'random_state': 14, 'min_epochs': 1, 'patience': 2}
    cache = TinyCache()
    keys = unet._choose_keys(groups, 2, 1, 14)
    original_save = unet._save_torch
    interrupted = {'done': False}

    def stop_after_first_epoch(path, value):
        original_save(path, value)
        if path.name == 'checkpoint.pt' and not interrupted['done']:
            interrupted['done'] = True
            raise KeyboardInterrupt

    monkeypatch.setattr(unet, '_save_torch', stop_after_first_epoch)
    with pytest.raises(KeyboardInterrupt):
        unet._fit(stage, groups, cache, columns, scale, params, 2, settings,
                  torch.device('cpu'), tmp_path / 'trial',
                  (stage, groups, cache, keys))
    monkeypatch.setattr(unet, '_save_torch', original_save)
    artifact, history = unet._fit(
        stage, groups, cache, columns, scale, params, 2, settings,
        torch.device('cpu'), tmp_path / 'trial', (stage, groups, cache, keys))
    assert artifact['architecture'] == ('inception_dense' if residual == 'inception' else ('resunet' if residual else 'unet'))
    assert artifact['epoch'] in (1, 2)
    assert len(history) == 2
    assert (tmp_path / 'trial' / 'model.pt').is_file()
    assert not (tmp_path / 'trial' / 'checkpoint.pt').exists()
    ids, prediction = unet._predict_stage(artifact, stage, groups, cache,
                                           torch.device('cpu'))
    assert sorted(ids.tolist()) == list(range(24))
    assert np.isfinite(prediction).all()
    reused, reused_history = unet._fit(
        stage, groups, cache, columns, scale, params, 2, settings,
        torch.device('cpu'), tmp_path / 'trial', (stage, groups, cache, keys))
    assert reused['epoch'] == artifact['epoch']
    assert len(reused_history) == len(history)


@pytest.mark.parametrize('residual', [False, True, 'inception'])
def test_horizon_trains_all_six_arms_without_external_target(tmp_path, monkeypatch, residual):
    monkeypatch.setattr(unet, 'CORE', 32)
    monkeypatch.setattr(unet, 'HALO', 16)
    monkeypatch.setattr(unet, 'DEPTH', 2)
    monkeypatch.setattr(unet, 'InputCache', lambda *_: TinyCache())
    torch.set_num_threads(1)
    rng = np.random.default_rng(15)
    source = tmp_path / 'source' / 'horizon_0'
    source.mkdir(parents=True)
    zones = np.zeros((96, 96), dtype=np.uint8)
    zones[:32], zones[32:64], zones[64:] = 1, 2, 3
    payload = {}
    for stage, row in zip(unet.STAGES, (10, 45, 75)):
        x = rng.normal(size=(24, 24)).astype(np.float32)
        y = np.abs(x[:, 0]).astype(np.float64) + .1
        positions = np.column_stack((np.full(24, row),
                                     np.arange(24) * 4)).astype(np.int32)
        payload.update({f'{stage}_x': x, f'{stage}_y': y,
                        f'{stage}_positions': positions,
                        f'{stage}_target_year': np.full(24, 2020, dtype=np.int16)})
    np.savez_compressed(source / 'samples.npz', **payload)
    cfg = {'sample_source': source.parent,
           'raw': {'unet': {'training_tiles_per_year': 2,
                            'selection_tiles_per_year': 2,
                            'random_state': 15, 'max_epochs': 2,
                            'min_epochs': 1, 'patience': 1,
                            'candidates': [
                                {'initial_filters': 2, 'learning_rate': .001,
                                 'weight_decay': .01},
                                {'initial_filters': 3, 'learning_rate': .001,
                                 'weight_decay': .01}]},
                   'evaluation': {'high_target_positive_quantile': .99}}}
    if residual == 'inception':
        cfg['raw']['unet']['extended_arms'] = True
        for params in cfg['raw']['unet']['candidates']:
            params['block_type'] = 'inception'
    elif residual:
        for params in cfg['raw']['unet']['candidates']:
            params['residual_blocks'] = True
    result = unet.train_horizon(cfg, {'height': 96, 'width': 96}, zones, 0,
                                tmp_path / 'run', torch.device('cpu'))
    assert len(result['records']) == (8 if residual == 'inception' else 6)
    assert result['winner']['arm'] in unet._arms(cfg['raw']['unet'])
    assert (tmp_path / 'run' / 'feature_occlusion.csv').is_file()
    assert (tmp_path / 'run' / 'hyperparameter_trials.csv').is_file()
    assert sum(1 for _ in (tmp_path / 'run').glob('*/model.pt')) == (8 if residual == 'inception' else 6)
    artifact = torch.load(result['winner']['model_path'], weights_only=True)
    assert artifact['architecture'] == ('inception_dense' if residual == 'inception' else ('resunet' if residual else 'unet'))


@pytest.mark.parametrize('residual', [False, True, 'inception'])
def test_external_prediction_writes_geotiff_and_scores_only_common_pixels(
        tmp_path, monkeypatch, residual):
    import rasterio
    from rasterio.transform import from_origin

    monkeypatch.setattr(unet, 'CORE', 32)
    monkeypatch.setattr(unet, 'HALO', 16)
    monkeypatch.setattr(unet, 'DEPTH', 2)
    torch.set_num_threads(1)
    profile = {'driver': 'GTiff', 'width': 64, 'height': 64, 'count': 1,
               'dtype': 'float32', 'crs': 'EPSG:3857',
               'transform': from_origin(0, 16000, 250, 250), 'nodata': -9999.0}
    target_path = tmp_path / 'target.tif'
    target = np.ones((64, 64), dtype=np.float32)
    target[0, 0] = -9999
    with rasterio.open(target_path, 'w', **profile) as output:
        output.write(target, 1)
    cache_path = tmp_path / 'inputs.npy'
    inputs = np.ones((24, 64, 64), dtype=np.float32)
    inputs[:, 0, 1] = np.nan
    np.save(cache_path, inputs)
    monkeypatch.setattr(unet.cnn, '_aligned_cache', lambda *_: cache_path)
    monkeypatch.setattr(unet.common, '_external_profile', lambda *_: profile)
    monkeypatch.setattr(unet.common, '_target_path', lambda *_: target_path)
    monkeypatch.setattr(unet.common, '_persistence', lambda *_: None)
    monkeypatch.setattr(unet.common, '_bootstrap_block_errors',
                        lambda *_args, **_kwargs: {'rmse_ci95': [0.0, 1.0]})
    columns = (0, 1)
    params = {'initial_filters': 2, 'learning_rate': .001, 'weight_decay': .01}
    if residual == 'inception':
        params['block_type'] = 'inception'
    elif residual:
        params['residual_blocks'] = True
    scale = {'input_mean': np.zeros(2, dtype=np.float32),
             'input_scale': np.ones(2, dtype=np.float32),
             'target_mean': 0.0, 'target_scale': 1.0}
    model = UNet(2, unet._model_config(2, residual=residual is True, block_type='inception' if residual == 'inception' else 'conv'))
    artifact = unet._artifact(model, columns, scale, params, 1)
    model_path = tmp_path / 'model.pt'
    torch.save(artifact, model_path)
    folder = tmp_path / 'external'
    folder.mkdir()
    cfg = {'raw': {'sampling': {'seed': 42}}, '_aligned_cache': tmp_path}
    selected = {'model_path': str(model_path),
                'feature_names': artifact['feature_names'],
                'high_target_threshold': 1.0, 'training_target_mean': 1.0}
    records = unet.evaluate_external(cfg, selected, 0, folder, [2021],
                                      torch.device('cpu'))
    assert records[0]['common_pixels'] == 64 * 64 - 2
    assert records[0]['n_features'] == 2
    with rasterio.open(records[0]['prediction']) as prediction:
        values = prediction.read(1)
    assert np.isnan(values[0, 0]) and np.isnan(values[0, 1])
    assert np.isfinite(values[1:, 1:]).all()


@pytest.mark.parametrize('residual', [False, True, 'inception'])
def test_report_includes_twelve_arms_five_pairs_and_images(tmp_path, monkeypatch, residual):
    import plot_enriched_linear_external

    monkeypatch.setattr(plot_enriched_linear_external, 'render_one',
                        lambda record, folder: {'target_year': record['target_year']})
    run_dir = tmp_path / 'run'
    run_dir.mkdir()

    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding='utf-8')

    save(run_dir / 'run.json', {'kind': 'enriched_inception' if residual == 'inception' else ('enriched_resunet' if residual else unet.KIND),
                               'final_external_test_used': True})
    (run_dir / 'config.yaml').write_text(yaml.safe_dump({
        'unet': {'training_tiles_per_year': 2, 'selection_tiles_per_year': 2,
                 'max_epochs': 2, 'min_epochs': 1, 'patience': 1, 'extended_arms': residual == 'inception'}}), encoding='utf-8')
    metric = {'rmse': 2.0, 'mae': 1.0, 'r2': .3, 'bias': .1,
              'nmad': .5, 'p90_absolute_error': 3.0,
              'capture_10pct': .25}
    for horizon in (0, 1):
        folder = run_dir / f'horizon_{horizon}'
        save(folder / 'winner.json', {'arm': 'originais', 'n_features': 4,
                                      'feature_names': ['a', 'b', 'c', 'd']})
        save(folder / 'hyperparameter_choices.json',
             {arm: {'candidate_id': 0} for arm in unet._base_arms({'extended_arms': residual == 'inception'})})
        for arm in unet._arms({'extended_arms': residual == 'inception'}):
            save(folder / arm / 'metrics.json',
                 {'n_features': 4, 'feature_names': ['a', 'b', 'c', 'd'],
                  'hyperparameters': {'initial_filters': 2,
                                      'learning_rate': .001, 'weight_decay': .01},
                  'selected_epoch': 2, 'actual_epochs': 2, 'parameters': 200, 'optimizer_steps': 4,
                  'initial_selection_metrics': metric,
                  'training_metrics': metric, 'validation_metrics': metric,
                  'training_samples_available': 20, 'validation_samples': 10})
            history = folder / arm / 'training_history.csv'
            history.write_text('epoch,train_loss_scaled\n1,2.0\n2,1.0\n',
                               encoding='utf-8')
        for arm in unet._base_arms({'extended_arms': residual == 'inception'}):
            path = folder / 'trials' / f'{arm}_0' / 'training_history.csv'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('epoch,selection_rmse,train_loss_scaled\n1,3.0,2.0\n2,2.0,1.0\n',
                            encoding='utf-8')
    for horizon, input_year, target_year, supplemental in (
            (0, 2020, 2020, False), (0, 2021, 2021, False),
            (0, 2022, 2022, False), (1, 2021, 2022, False),
            (1, 2020, 2021, True)):
        folder = run_dir / ('supplemental_H1_2021' if supplemental
                            else f'horizon_{horizon}')
        save(folder / f'external_target_{target_year}_metrics.json',
             {'horizon': horizon, 'input_year': input_year,
              'target_year': target_year, 'n_features': 4,
              'feature_names': ['a', 'b', 'c', 'd'],
              'common_pixels': 100, 'target_sha256': 'fixed',
              'prediction': str(folder / 'prediction.tif'),
              'metrics': metric, 'reference_zero': metric,
              'reference_training_mean': metric,
              'reference_persistence': None, 'uncertainty': {}})
    if residual:
        for horizon in (0, 1):
            folder = run_dir / f'horizon_{horizon}'
            unet.common._write_csv(folder / 'hyperparameter_trials.csv', [
                {'base_arm': arm, 'candidate_id': 0, 'params': json.dumps({'initial_filters': 8}),
                 'selected_epoch': 2, 'actual_epochs': 3, 'selection_rmse': 2., 'selection_mae': 1.}
                for arm in unet._base_arms({'extended_arms': residual == 'inception'})])
            save(folder / 'feature_occlusion.json', {arm: {'attempts': [
                {'features': ['a'], 'muted': [0], 'selection_rmse': 2., 'selection_mae': 1., 'selected': True}]}
                for arm in ('originais_filtradas', 'todas_filtradas')})
    unet.rebuild_report(run_dir)
    report = (run_dir / 'RELATORIO.md').read_text(encoding='utf-8')
    assert report.count('| H0 |') >= 6 and report.count('| H1 |') >= 6
    assert report.count('![Target original e previsão ' + ('Inception densa' if residual == 'inception' else ('ResUNet' if residual else 'U-Net'))) == 5
    if residual == 'inception':
        assert 'experimento 12' in report and 'Szegedy' in report
        import pandas as pd
        assert len(pd.read_csv(run_dir / 'contrastes_gpa_grafos.csv')) == 8
    elif residual:
        assert 'experimento 11' in report and 'He et al.' in report and 'Zhang' in report
        assert 'Removidos em relação' in report and 'todas as combinações por ocultação' in report
    assert 'Ronneberger' in report
    assert (run_dir / f"resultados_{16 if residual == 'inception' else 12}_treinamentos.csv").is_file()
    assert (run_dir / 'teste_externo_metricas.csv').is_file()
