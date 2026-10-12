"""Checks that the paired control never consumes the wide input branch."""

import numpy as np
import torch

from src.services import enriched_cnn as cnn


def test_local_control_ignores_wide_patch_and_has_no_wide_weights():
    torch.set_num_threads(1)
    rng = np.random.default_rng(75)
    x = torch.from_numpy(rng.normal(size=(4, 2, 3, 5, 5)).astype(np.float32))
    model = cnn.CompactCNN(3, 4, wide_context=False).eval()
    altered = x.clone()
    altered[:, 1] = 10000
    with torch.inference_mode():
        np.testing.assert_array_equal(model(x).numpy(), model(altered).numpy())
    assert all('wide' not in name for name in model.state_dict())


def test_local_dense_inference_matches_patches_and_fits():
    torch.set_num_threads(1)
    rng = np.random.default_rng(76)
    raw = rng.normal(size=(24, 9, 10)).astype(np.float32)
    padded = np.pad(raw, ((0, 0), (2, 2), (2, 2)), constant_values=np.nan)
    patches = np.zeros((3, 2, 24, 5, 5), dtype=np.float32)
    positions = ((0, 0), (4, 5), (8, 9))
    for index, (row, col) in enumerate(positions):
        patches[index, 0] = padded[:, row:row + 5, col:col + 5]
        patches[index, 1] = 9999
    view = cnn._view(patches, np.zeros(len(patches)))
    scale = {'input_mean': np.zeros(3, dtype=np.float32),
             'input_scale': np.ones(3, dtype=np.float32),
             'target_mean': 0.0, 'target_scale': 1.0}
    params = {'filters': 4, 'learning_rate': .001, 'weight_decay': .01}
    settings = {'random_state': 20, 'batch_size': 2, 'min_epochs': 1, 'patience': 2}
    artifact, history, _ = cnn.fit_cnn(view, (0, 1, 2), scale, params, 2,
                                        settings, torch.device('cpu'),
                                        wide_context=False)
    assert artifact['wide_context'] is False and len(history) == 2
    dense = cnn._dense_predict(artifact, padded, 9, 10, torch.device('cpu'))
    single = cnn.predict_view(artifact, view, 2, torch.device('cpu'))
    np.testing.assert_allclose(single, [dense[row, col] for row, col in positions],
                               atol=1e-5, rtol=1e-5)
