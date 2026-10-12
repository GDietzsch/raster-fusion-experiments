"""Local residual branch must ignore the wide input and tile exactly."""
import json
import numpy as np
import torch

from src.services import enriched_cnn as cnn
from src.services import enriched_cnn_local_control as control


def test_local_resnet_ignores_wide_branch_and_matches_dense_edges():
    torch.set_num_threads(1)
    rng = np.random.default_rng(127)
    model = cnn.ResidualCNN(3, 4, wide_context=False).eval()
    x = torch.from_numpy(rng.normal(size=(4, 2, 3, 5, 5)).astype(np.float32))
    changed = x.clone()
    changed[:, 1] = 10000
    with torch.inference_mode():
        np.testing.assert_array_equal(model(x).numpy(), model(changed).numpy())
    assert all('wide' not in name for name in model.state_dict())

    raw = rng.normal(size=(24, 9, 10)).astype(np.float32)
    padded = np.pad(raw, ((0, 0), (2, 2), (2, 2)), constant_values=np.nan)
    positions = ((0, 0), (4, 5), (8, 9), (2, 3))
    patches = np.zeros((4, 2, 24, 5, 5), dtype=np.float32)
    for index, (row, col) in enumerate(positions):
        patches[index, 0] = padded[:, row:row + 5, col:col + 5]
        patches[index, 1] = 9999
    view = cnn._view(patches, np.zeros(len(patches)))
    scale = {'input_mean': np.zeros(3, dtype=np.float32),
             'input_scale': np.ones(3, dtype=np.float32),
             'target_mean': 0.0, 'target_scale': 1.0}
    params = {'filters': 4, 'learning_rate': .001, 'weight_decay': .01}
    settings = {'random_state': 127, 'batch_size': 2,
                'min_epochs': 1, 'patience': 2}
    artifact, history, _ = cnn.fit_cnn(
        view, (0, 1, 2), scale, params, 2, settings, torch.device('cpu'),
        wide_context=False, architecture='resnet_compact')
    assert artifact['wide_context'] is False
    assert artifact['architecture'] == 'resnet_compact'
    assert len(history) == 2
    dense = cnn._dense_predict(artifact, padded, 9, 10, torch.device('cpu'))
    patch = cnn.predict_view(artifact, view, 2, torch.device('cpu'))
    np.testing.assert_allclose(patch, [dense[row, col] for row, col in positions],
                               atol=1e-5, rtol=1e-5)


def test_local_resnet_report_records_fixed_reference_and_all_five_pairs(
        tmp_path, monkeypatch):
    import plot_enriched_linear_external

    monkeypatch.setattr(plot_enriched_linear_external, 'render_one',
                        lambda record, folder: {'pair': record['target_year']})
    run_dir, reference = tmp_path / 'local', tmp_path / 'reference'
    run_dir.mkdir()
    reference.mkdir()

    def save(path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding='utf-8')

    save(run_dir / 'run.json',
         {'kind': control.RESNET_KIND, 'final_external_test_used': True,
          'reference_run': str(reference)})
    metric = {'rmse': 2.0, 'mae': 1.0, 'r2': .2, 'nmad': .5,
              'capture_10pct': .3}
    for horizon, arm, channels in ((0, 'grafos', 8), (1, 'gpa', 12)):
        folder = run_dir / f'horizon_{horizon}'
        save(folder / 'metrics.json',
             {'horizon': horizon, 'reference_arm': arm, 'n_features': channels,
              'feature_names': [f'channel_{i}' for i in range(channels)],
              'hyperparameters': {'filters': 16, 'learning_rate': .001,
                                  'weight_decay': .01},
              'epochs': 2, 'optimizer_steps': 4, 'n_parameters': 50,
              'reference_n_parameters': 100,
              'training_samples': 10, 'validation_samples': 5,
              'training_metrics': metric, 'validation_metrics': metric,
              'reference_validation_metrics': dict(metric, rmse=3.0)})
        (folder / 'training_history.csv').write_text(
            'epoch,train_loss_scaled\n1,2.0\n2,1.0\n', encoding='utf-8')
    for horizon, input_year, target_year, supplemental in control.PAIRS:
        name = 'supplemental_H1_2021' if supplemental else f'horizon_{horizon}'
        for root, rmse in ((run_dir, 2.0), (reference, 3.0)):
            record = {'horizon': horizon, 'input_year': input_year,
                      'target_year': target_year, 'target_sha256': 'fixed-hash',
                      'common_pixels': 100, 'metrics': dict(metric, rmse=rmse),
                      'reference_zero': {'rmse': 1.0},
                      'prediction': str(run_dir / name / 'prediction.tif')}
            save(root / name / f'external_target_{target_year}_metrics.json', record)
    control.rebuild_report(run_dir)
    report = (run_dir / 'RELATORIO.md').read_text(encoding='utf-8')
    assert report.count('## Treinamento e validação pareada') == 1
    assert 'experimento 09A' in report
    assert 'grafos / 8' in report and 'gpa / 12' in report
    assert report.count('![Target original e previsão ResNet local') == 5
    assert 'EXPERIMENTO_09A_RESNET_CONTROLE_LOCAL.md' in report
