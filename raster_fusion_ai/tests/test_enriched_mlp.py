"""Check MLP scaling, epoch search, resume and the complete enriched protocol."""
import csv
import json

import joblib
import numpy as np
import yaml

import src.services.enriched_mlp as mlp_service
from src.services.enriched_mlp import fit_mlp, run, search_hyperparameters
from test_enriched_linear import _synthetic_dataset


SETTINGS = dict(loss='squared_error', activation='relu', solver='adam',
                early_stopping=False, random_state=4209, selection_seed=4203,
                selection_samples_per_stage_year=8, max_epochs=4,
                patience=2, min_epochs=2, batch_size=8,
                candidates=[dict(hidden_layer_sizes=[4], alpha=0.001,
                                 learning_rate_init=0.001),
                            dict(hidden_layer_sizes=[8], alpha=0.01,
                                 learning_rate_init=0.003)])


def test_scaler_uses_fit_only_and_partial_fit_keeps_epoch_history():
    x = np.arange(24 * 30, dtype=np.float64).reshape(30, 24) / 100
    y = x[:, 0] * 2
    score = x[:5] + 100
    model, history = fit_mlp(x, y, (0, 1), SETTINGS['candidates'][0], 4,
                             SETTINGS, monitor=(score, y[:5]))
    assert np.allclose(model.scaler.mean_, x[:, (0, 1)].mean(axis=0))
    assert len(history) >= 2
    assert len(model.estimator.loss_curve_) == len(history)
    assert all(np.isfinite(row['selection_rmse']) for row in history)
    assert model.predict(x[:3, (0, 1)]).shape == (3,)


def test_fifteen_stale_epochs_stop_the_500_epoch_budget(monkeypatch):
    class ConstantRegressor:
        def __init__(self, **kwargs):
            self.loss_ = 1.0

        def partial_fit(self, x, y):
            return self

        def predict(self, x):
            return np.zeros(len(x))

    monkeypatch.setattr(mlp_service, 'MLPRegressor', ConstantRegressor)
    x = np.arange(30 * 24, dtype=float).reshape(30, 24)
    y = np.arange(30, dtype=float)
    settings = dict(SETTINGS, max_epochs=500, min_epochs=10, patience=15)
    _, history = fit_mlp(x, y, (0,), SETTINGS['candidates'][0], 500,
                         settings, monitor=(x, y))
    assert len(history) == 25
    assert history[-1]['epoch'] == 25


def test_search_checkpoint_is_idempotent(tmp_path):
    rng = np.random.default_rng(7)
    x0, x1 = rng.normal(size=(24, 24)), rng.normal(size=(16, 24))
    selection = {'ajuste': (x0, x0[:, 0], None, None),
                 'selecao_camadas': (x1, x1[:, 0], None, None)}
    first = search_hyperparameters(selection, SETTINGS, tmp_path)
    checkpoint = (tmp_path / 'hyperparameter_trials.json').read_bytes()
    second = search_hyperparameters(selection, SETTINGS, tmp_path)
    assert first == second
    assert checkpoint == (tmp_path / 'hyperparameter_trials.json').read_bytes()
    rows = json.loads(checkpoint)
    assert len(rows) == 4 * len(SETTINGS['candidates'])
    assert all(SETTINGS['min_epochs'] <= row['chosen_epochs'] <= SETTINGS['max_epochs']
               for row in rows)
    assert all(len(row['history']) >= SETTINGS['min_epochs'] for row in rows)


def test_full_synthetic_mlp_run_and_report(tmp_path):
    config_path = _synthetic_dataset(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    config['mlp'] = SETTINGS.copy()
    config['output_root'] = 'mlp_runs'
    config_path.write_text(yaml.safe_dump(config), encoding='utf-8')
    run(config_path)
    output = next((tmp_path / 'mlp_runs').iterdir())
    status = json.loads((output / 'run.json').read_text(encoding='utf-8'))
    assert status['status'] == 'completed'
    assert status['final_external_test_used']
    report = (output / 'RELATORIO.md').read_text(encoding='utf-8')
    assert 'experimento 07, MLP' in report
    assert '## As 15 combinações de originais' in report
    assert 'Avaliação complementar pré-definida' in report
    assert '## MLP: arquitetura, épocas e histórico real' in report
    diagnostics = output / 'graficos_treinamento'
    assert (diagnostics / 'rmse_selecao_por_epoca.png').is_file()
    assert (diagnostics / 'perda_treino_por_epoca.png').is_file()
    for horizon, years in ((0, (2020, 2021, 2022)), (1, (2022,))):
        folder = output / f'horizon_{horizon}'
        with (folder / 'hyperparameter_trials.csv').open(encoding='utf-8-sig', newline='') as source:
            assert len(list(csv.DictReader(source))) == 8
        assert (folder / 'hyperparameter_epoch_history.csv').is_file()
        assert (folder / 'selection_subsample_indices.npz').is_file()
        assert (folder / 'comparison.csv').is_file()
        winner = json.loads((folder / 'winner.json').read_text(encoding='utf-8'))
        saved = joblib.load(folder / winner['arm'] / 'model.joblib')
        assert saved['selected_epoch'] >= SETTINGS['min_epochs']
        assert saved['model'].estimator.solver == 'adam'
        for year in years:
            assert (folder / f'external_prediction_target_{year}.tif').is_file()
            assert (output / f'teste_externo_H{horizon}_{year-horizon}_{year}.png').is_file()
    supplement = output / 'supplemental_H1_2021'
    assert (supplement / 'external_target_2021_metrics.json').is_file()
    assert (supplement / 'teste_externo_H1_2020_2021.png').is_file()
