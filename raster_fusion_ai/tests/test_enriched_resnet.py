"""Focused checks for the adapted residual branches and tiled inference."""
import numpy as np
import torch

from src.services import enriched_cnn as cnn


def test_residual_patch_and_dense_predictions_match_after_batchnorm_training():
    torch.set_num_threads(1)
    rng = np.random.default_rng(91)
    fit_bank = rng.normal(size=(16, 2, 24, 5, 5)).astype(np.float32)
    score_bank = rng.normal(size=(8, 2, 24, 5, 5)).astype(np.float32)
    fit_view = cnn._view(fit_bank, fit_bank[:, 0, 0, 2, 2].astype(np.float64))
    score_view = cnn._view(score_bank, score_bank[:, 0, 0, 2, 2].astype(np.float64))
    scale = {'input_mean': np.zeros(3, dtype=np.float32),
             'input_scale': np.ones(3, dtype=np.float32),
             'target_mean': 0.0, 'target_scale': 1.0}
    params = {'filters': 4, 'learning_rate': .001, 'weight_decay': .01}
    settings = {'random_state': 91, 'batch_size': 4,
                'min_epochs': 1, 'patience': 2}
    _, history, best = cnn.fit_cnn(
        fit_view, (0, 1, 2), scale, params, 3, settings,
        torch.device('cpu'), score_view, architecture='resnet_compact')
    assert best is not None and best['architecture'] == 'resnet_compact'
    assert len(history) >= 2
    assert np.isfinite(cnn.predict_view(best, score_view, 4, torch.device('cpu'))).all()
    assert any(name.endswith('running_mean') for name in best['state_dict'])

    raw = rng.normal(size=(24, 9, 10)).astype(np.float32)
    block = np.pad(raw, ((0, 0), (cnn.WIDE_RADIUS, cnn.WIDE_RADIUS),
                         (cnn.WIDE_RADIUS, cnn.WIDE_RADIUS)), constant_values=np.nan)
    dense = cnn._dense_predict(best, block, 9, 10, torch.device('cpu'))
    positions = ((0, 0), (4, 5), (8, 9))
    patches = np.stack([cnn.extract_two_scales(block, row, col)
                        for row, col in positions])
    view = cnn._view(patches, np.zeros(len(positions)))
    patch = cnn.predict_view(best, view, 3, torch.device('cpu'))
    np.testing.assert_allclose(patch, [dense[row, col] for row, col in positions],
                               atol=1e-5, rtol=1e-5)


def test_residual_model_has_projection_shortcuts_and_keeps_5x5_support():
    model = cnn.ResidualCNN(24, 8)
    names = tuple(model.state_dict())
    assert any(name.startswith('local.shortcut.') for name in names)
    assert any(name.startswith('wide.shortcut.') for name in names)
    with torch.inference_mode():
        prediction = model.eval()(torch.zeros(2, 2, 24, 5, 5))
    assert prediction.shape == (2, 1, 1)
