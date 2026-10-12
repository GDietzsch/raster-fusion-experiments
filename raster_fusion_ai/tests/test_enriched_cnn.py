"""Focused checks for the new spatial context and dense CNN inference."""
import numpy as np
import pytest
import torch
from rasterio.windows import Window

from src.services import enriched_cnn as cnn
from src.services.enriched_cnn import (
    CompactCNN, PatchView, WIDE_RADIUS, _artifact, _dense_predict,
    _view, extract_two_scales, fit_cnn, predict_view,
)


def test_local_and_41km_samples_have_correct_positions():
    size = 5 + 2 * WIDE_RADIUS
    block = np.zeros((24, size, size), dtype=np.float32)
    rows, cols = np.indices((size, size))
    block[0] = rows
    block[1] = cols
    patch = extract_two_scales(block, 2, 2)
    assert patch.shape == (2, 24, 5, 5)
    assert patch[0, 0, 2, 2] == WIDE_RADIUS + 2
    assert patch[1, 0, 0, 2] == 2
    assert patch[1, 0, 4, 2] == 2 * WIDE_RADIUS + 2
    assert patch[1, 1, 2, 0] == 2
    assert patch[1, 1, 2, 4] == 2 * WIDE_RADIUS + 2


def test_dense_inference_equals_center_patch_including_raster_edge():
    torch.set_num_threads(1)
    rng = np.random.default_rng(42)
    raw = rng.normal(size=(24, 9, 10)).astype(np.float32)
    block = np.pad(raw, ((0, 0), (WIDE_RADIUS, WIDE_RADIUS),
                         (WIDE_RADIUS, WIDE_RADIUS)), constant_values=np.nan)
    model = CompactCNN(3, 4)
    scale = {'input_mean': np.zeros(3, dtype=np.float32),
             'input_scale': np.ones(3, dtype=np.float32),
             'target_mean': 1.7, 'target_scale': 2.3}
    artifact = _artifact(model, (0, 1, 2),
                         {'filters': 4, 'learning_rate': .001, 'weight_decay': .01},
                         scale, 1)
    dense = _dense_predict(artifact, block, 9, 10, torch.device('cpu'))
    positions = ((0, 0), (4, 5), (8, 9))
    patches = np.stack([extract_two_scales(block, row, col)
                        for row, col in positions])
    patch = _view(patches, np.zeros(len(positions)))
    single = predict_view(artifact, patch, 2, torch.device('cpu'))
    assert dense.shape == (9, 10)
    np.testing.assert_allclose(single, [dense[row, col] for row, col in positions],
                               atol=1e-5, rtol=1e-5)


def test_fit_has_observed_epoch_history_and_finite_predictions():
    torch.set_num_threads(1)
    rng = np.random.default_rng(43)
    fit_bank = rng.normal(size=(16, 2, 24, 5, 5)).astype(np.float32)
    score_bank = rng.normal(size=(8, 2, 24, 5, 5)).astype(np.float32)
    fit_y = fit_bank[:, 0, 0, 2, 2].astype(np.float64)
    score_y = score_bank[:, 0, 0, 2, 2].astype(np.float64)
    fit_view = _view(fit_bank, fit_y)
    score_view = _view(score_bank, score_y)
    scale = {'input_mean': np.zeros(1, dtype=np.float32),
             'input_scale': np.ones(1, dtype=np.float32),
             'target_mean': 0.0, 'target_scale': 1.0}
    settings = {'random_state': 10, 'batch_size': 4,
                'min_epochs': 1, 'patience': 2}
    params = {'filters': 4, 'learning_rate': .001, 'weight_decay': .01}
    final, history, best = fit_cnn(fit_view, (0,), scale, params, 3,
                                   settings, torch.device('cpu'), score_view)
    assert len(history) in (2, 3)
    assert best is not None and 1 <= best['epoch'] <= len(history)
    assert history[-1]['optimizer_steps'] == 4 * len(history)
    assert np.isfinite(predict_view(final, score_view, 4, torch.device('cpu'))).all()


@pytest.mark.parametrize('architecture', ('compact_cnn', 'resnet_compact'))
def test_six_arm_horizon_trains_and_selects_without_external_targets(
        tmp_path, monkeypatch, architecture):
    torch.set_num_threads(1)
    rng = np.random.default_rng(44)
    source = tmp_path / 'source' / 'horizon_0'
    source.mkdir(parents=True)
    zones = np.zeros((12, 12), dtype=np.uint8)
    zones[:4] = 1
    zones[4:8] = 2
    zones[8:] = 3
    payload = {}
    for stage, zone in zip(cnn.STAGES, (1, 2, 3)):
        x = rng.normal(size=(8, 24)).astype(np.float32)
        y = np.abs(x[:, 0]).astype(np.float64) + .1
        position = np.column_stack((np.full(8, (zone - 1) * 4 + 1),
                                    np.arange(8))).astype(np.int32)
        payload.update({f'{stage}_x': x, f'{stage}_y': y,
                        f'{stage}_positions': position,
                        f'{stage}_target_year': np.full(8, 2020, dtype=np.int16)})
    np.savez_compressed(source / 'samples.npz', **payload)

    def fake_bank(cfg, profile, stages, horizon, directory):
        for stage in cnn.STAGES:
            centers = stages[stage][0]
            bank = np.broadcast_to(centers[:, None, :, None, None],
                                   (len(centers), 2, 24, 5, 5)).copy()
            np.save(directory / f'{stage}_patches.npy', bank)
        return {'synthetic': True}

    monkeypatch.setattr(cnn, '_fill_patch_bank', fake_bank)
    cfg = {'sample_source': source.parent,
           'raw': {'cnn': {'architecture': architecture,
                           'selection_samples_per_stage_year': 4,
                           'selection_seed': 4203, 'max_epochs': 2,
                           'min_epochs': 1, 'patience': 1,
                           'batch_size': 4, 'random_state': 4210,
                           'candidates': [{'filters': 4, 'learning_rate': .001,
                                           'weight_decay': .01}]},
                   'evaluation': {'high_target_positive_quantile': .99},
                   'sampling': {'seed': 42}}}
    result = cnn.train_horizon(cfg, {'width': 12}, zones, 0,
                               tmp_path / 'output', torch.device('cpu'))
    assert len(result['records']) == 6
    assert result['winner']['arm'] in cnn.ARMS
    assert (tmp_path / 'output' / 'feature_occlusion.csv').is_file()
    assert (tmp_path / 'output' / 'comparison.csv').is_file()
    assert sum(1 for _ in (tmp_path / 'output').glob('*/model.pt')) == 6


def test_aligned_input_cache_supports_halo_and_missing_borders(tmp_path, monkeypatch):
    values = np.arange(24 * 6 * 7, dtype=np.float32).reshape(6, 7, 24)
    monkeypatch.setattr(cnn.common, '_paths', lambda *args: [tmp_path / f'{i}.tif' for i in range(24)])
    monkeypatch.setattr(cnn.common, '_methods', lambda *args: ['nearest'] * 25)
    monkeypatch.setattr(cnn.common, 'aligned_windows',
                        lambda *args: [(Window(0, 0, 7, 6), values)])
    cfg = {'_aligned_cache': tmp_path, 'root': tmp_path,
           'raw': {'sampling': {'window_size': 7}}}
    profile = {'width': 7, 'height': 6}
    path = cnn._aligned_cache(cfg, 'treinamento', 2020, profile)
    with cnn.AlignedInputs(cfg, 'treinamento', 2020, profile, path) as reader:
        tile = reader.tile(0, 0, 2, 3, radius=2)
    assert tile.shape == (24, 6, 7)
    assert np.isnan(tile[:, 0, 0]).all()
    np.testing.assert_array_equal(tile[:, 2, 2], values[0, 0])
    np.testing.assert_array_equal(tile[:, 3, 4], values[1, 2])
