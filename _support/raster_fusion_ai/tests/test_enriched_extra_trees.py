"""Exercise Extra Trees search, checkpoints, six arms and external report."""
import csv
import json

import joblib
import numpy as np
import yaml

from src.services.enriched_extra_trees import run, search_hyperparameters
from test_enriched_linear import _synthetic_dataset


SETTINGS = dict(criterion='squared_error', bootstrap=False, oob_score=False,
                random_state=4207, candidate_seed=4208,
                selection_samples_per_stage_year=8, n_candidates=3, n_finalists=2,
                screening_trees=2, filtering_trees=3, final_trees=4,
                max_features=[0.5, 0.75], min_samples_leaf=[0.01, 0.02],
                max_leaf_nodes=[4, 8], min_samples_split=[0.02, 0.04],
                reference_candidate=dict(max_features=0.75,
                                         min_samples_leaf=0.01,
                                         max_leaf_nodes=8, min_samples_split=0.02))


def test_search_resumes_without_retraining(tmp_path):
    rng = np.random.default_rng(7)
    x0, x1 = rng.normal(size=(32, 24)), rng.normal(size=(20, 24))
    selection = {'ajuste': (x0, x0[:, 0], None, None),
                 'selecao_camadas': (x1, x1[:, 0], None, None)}
    first = search_hyperparameters(selection, SETTINGS, 1, tmp_path)
    checkpoint = (tmp_path / 'hyperparameter_trials.json').read_bytes()
    second = search_hyperparameters(selection, SETTINGS, 1, tmp_path)
    assert first == second
    assert checkpoint == (tmp_path / 'hyperparameter_trials.json').read_bytes()
    rows = json.loads(checkpoint)
    assert len(rows) == 4 * (SETTINGS['n_candidates'] + SETTINGS['n_finalists'])
    assert all(set(row['prefix_rmse']) == {'2', '3', '4'}
               for row in rows if row['stage'] == 'refinement')


def test_full_synthetic_extra_trees_run_and_report(tmp_path):
    config_path = _synthetic_dataset(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    config['extra_trees'] = SETTINGS.copy()
    config['output_root'] = 'et_runs'
    config_path.write_text(yaml.safe_dump(config), encoding='utf-8')
    run(config_path)
    output = next((tmp_path / 'et_runs').iterdir())
    status = json.loads((output / 'run.json').read_text(encoding='utf-8'))
    assert status['status'] == 'completed'
    assert status['final_external_test_used']
    report = (output / 'RELATORIO.md').read_text(encoding='utf-8')
    assert 'experimento 06, Extra Trees' in report
    assert '## As 15 combinações de originais' in report
    assert 'Avaliação complementar pré-definida' in report
    assert 'Geurts' in report
    assert '## Auditoria complementar dos treinamentos' in report
    diagnostics = output / 'graficos_treinamento'
    assert (diagnostics / 'parametros_efetivos.csv').is_file()
    assert (diagnostics / 'curvas_prefixo_vencedores.png').is_file()
    assert (diagnostics / 'curvas_prefixo_procedencia.json').is_file()
    for horizon, years in ((0, (2020, 2021, 2022)), (1, (2022,))):
        folder = output / f'horizon_{horizon}'
        with (folder / 'hyperparameter_trials.csv').open(encoding='utf-8-sig', newline='') as handle:
            assert len(list(csv.DictReader(handle))) == 20
        assert (folder / 'selection_subsample_indices.npz').is_file()
        assert (folder / 'comparison.csv').is_file()
        winner = json.loads((folder / 'winner.json').read_text(encoding='utf-8'))
        saved = joblib.load(folder / winner['arm'] / 'model.joblib')
        assert saved['n_estimators'] == SETTINGS['final_trees']
        assert saved['model'].bootstrap is False
        for year in years:
            assert (folder / f'external_prediction_target_{year}.tif').is_file()
            assert (output / f'teste_externo_H{horizon}_{year-horizon}_{year}.png').is_file()
    supplement = output / 'supplemental_H1_2021'
    assert (supplement / 'external_target_2021_metrics.json').is_file()
    assert (supplement / 'teste_externo_H1_2020_2021.png').is_file()
