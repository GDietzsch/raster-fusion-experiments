"""Verify XGBoost's internal selection and the complete enriched report."""
import csv
import json

import joblib
import numpy as np
import yaml
from xgboost import XGBRegressor

from src.services.enriched_xgboost import fit_xgboost, run, tune_estimators
from test_enriched_linear import _synthetic_dataset


SETTINGS = dict(objective='reg:squarederror', tree_method='hist', device='cpu',
                eval_metric='rmse', learning_rate=0.08,
                n_estimators_candidates=[2, 4], max_depth=2,
                min_child_weight=1, reg_lambda=1.0, reg_alpha=0.0,
                gamma=0.0, subsample=1.0, colsample_bytree=1.0,
                max_bin=16, selection_samples_per_stage_year=8,
                selection_seed=4203, random_state=4204,
                early_stopping_rounds=None)


def test_fixed_rounds_and_internal_search(tmp_path):
    rng = np.random.default_rng(8)
    x0, x1 = rng.normal(size=(50, 24)), rng.normal(size=(30, 24))
    selection = {'ajuste': (x0, x0[:, 0], None, None),
                 'selecao_camadas': (x1, x1[:, 0], None, None)}
    model = fit_xgboost(x0, x0[:, 0], (0,), 2, SETTINGS, 1)
    assert model.get_booster().num_boosted_rounds() == 2
    chosen = tune_estimators(selection, [2, 4], SETTINGS, 1, tmp_path)
    assert set(chosen) == {'originais', 'gpa', 'grafos', 'todas'}
    with (tmp_path / 'n_estimators_search.csv').open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 8
    assert sum(row['chosen'] == 'True' for row in rows) == 4


def test_full_synthetic_xgboost_run_with_maps_and_native_models(tmp_path):
    config_path = _synthetic_dataset(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    config['xgboost'] = SETTINGS.copy()
    config['output_root'] = 'xgboost_runs'
    config_path.write_text(yaml.safe_dump(config), encoding='utf-8')
    run(config_path)
    output = next((tmp_path / 'xgboost_runs').iterdir())
    status = json.loads((output / 'run.json').read_text(encoding='utf-8'))
    assert status['status'] == 'completed'
    assert status['final_external_test_used']
    report = (output / 'RELATORIO.md').read_text(encoding='utf-8')
    assert 'experimento 04, XGBoost' in report
    assert 'Chen e Guestrin (2016)' in report
    assert '### Imagens do teste externo' in report
    assert '## As 15 combinações de originais' in report
    assert '## Auditoria complementar dos treinamentos' in report
    diagnostics = output / 'graficos_treinamento'
    assert (diagnostics / 'parametros_efetivos.csv').is_file()
    assert (diagnostics / 'curvas_prefixo_vencedores.png').is_file()
    assert (diagnostics / 'curvas_prefixo_procedencia.json').is_file()
    for horizon, years in ((0, (2020, 2021, 2022)), (1, (2022,))):
        folder = output / f'horizon_{horizon}'
        assert (folder / 'n_estimators_search.csv').is_file()
        assert (folder / 'selection_subsample.json').is_file()
        assert (folder / 'comparison.csv').is_file()
        winner = json.loads((folder / 'winner.json').read_text(encoding='utf-8'))
        assert winner['n_estimators'] in (2, 4)
        model_folder = folder / winner['arm']
        assert (model_folder / 'model.ubj').is_file()
        saved = joblib.load(model_folder / 'model.joblib')
        native = XGBRegressor()
        native.load_model(model_folder / 'model.ubj')
        with np.load(folder / 'samples.npz') as samples:
            x = samples['validacao_modelo_x'][:5, saved['columns']]
        np.testing.assert_allclose(native.predict(x), saved['model'].predict(x), rtol=1e-6)
        for year in years:
            assert (folder / f'external_prediction_target_{year}.tif').is_file()
            assert (output / f'teste_externo_H{horizon}_{year-horizon}_{year}.png').is_file()
    assert (output / 'teste_externo_canais.csv').is_file()
    assert (output / 'teste_externo_metricas.csv').is_file()
