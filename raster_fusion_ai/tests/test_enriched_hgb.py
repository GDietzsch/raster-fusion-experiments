"""Verify the HGB selection boundary and complete report on tiny rasters."""
import csv
import json

import numpy as np
import yaml

from src.services.enriched_hgb import fit_hgb, run, selection_subsample, tune_iterations
from test_enriched_linear import _synthetic_dataset


def test_fixed_yearwise_selection_subset(tmp_path):
    x = np.arange(11 * 24, dtype=np.float32).reshape(11, 24)
    years = np.array([2020] * 5 + [2021] * 6, dtype=np.int16)
    records = (x, np.arange(11), np.zeros((11, 2), dtype=np.int32), years)
    stage = {'ajuste': records, 'selecao_camadas': records,
             'validacao_modelo': (None, None, None, None)}
    chosen = selection_subsample(stage, 2, 42, tmp_path)
    assert len(chosen['ajuste'][1]) == len(chosen['selecao_camadas'][1]) == 4
    assert set(chosen) == {'ajuste', 'selecao_camadas'}
    assert (tmp_path / 'selection_subsample_indices.npz').is_file()
    with np.load(tmp_path / 'selection_subsample_indices.npz') as saved:
        assert len(saved['ajuste']) == 4
        assert len(saved['selecao_camadas']) == 4


def test_iteration_search_uses_fit_and_selection_only(tmp_path):
    rng = np.random.default_rng(6)
    x0, x1 = rng.normal(size=(50, 24)), rng.normal(size=(30, 24))
    selection = {'ajuste': (x0, x0[:, 0], None, None),
                 'selecao_camadas': (x1, x1[:, 0], None, None)}
    settings = dict(learning_rate=0.08, max_leaf_nodes=3,
                    min_samples_leaf=2, l2_regularization=1.0,
                    max_bins=16, random_state=42)
    model = fit_hgb(x0, x0[:, 0], (0,), 2, settings)
    assert model.n_iter_ == 2
    chosen = tune_iterations(selection, [2, 4], settings, tmp_path)
    assert set(chosen) == {'originais', 'gpa', 'grafos', 'todas'}
    with (tmp_path / 'max_iter_search.csv').open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 8
    assert sum(row['chosen'] == 'True' for row in rows) == 4


def test_full_synthetic_hgb_run_with_maps_and_references(tmp_path):
    config_path = _synthetic_dataset(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    config['hgb'] = dict(loss='squared_error', learning_rate=0.08,
                         max_iter_candidates=[2, 4], max_leaf_nodes=3,
                         min_samples_leaf=2, l2_regularization=1.0,
                         max_bins=16, selection_samples_per_stage_year=8,
                         early_stopping=False, random_state=4203)
    config['output_root'] = 'hgb_runs'
    config_path.write_text(yaml.safe_dump(config), encoding='utf-8')
    run(config_path)
    output = next((tmp_path / 'hgb_runs').iterdir())
    status = json.loads((output / 'run.json').read_text(encoding='utf-8'))
    assert status['status'] == 'completed'
    assert status['final_external_test_used']
    report = (output / 'RELATORIO.md').read_text(encoding='utf-8')
    assert 'HistGradientBoosting' in report
    assert 'Friedman (2001)' in report
    assert '### Imagens do teste externo' in report
    assert '## As 15 combinações de originais' in report
    assert '## Auditoria complementar dos treinamentos' in report
    diagnostics = output / 'graficos_treinamento'
    assert (diagnostics / 'parametros_efetivos.csv').is_file()
    assert (diagnostics / 'curvas_prefixo_vencedores.png').is_file()
    assert (diagnostics / 'curvas_prefixo_procedencia.json').is_file()
    for horizon, years in ((0, (2020, 2021, 2022)), (1, (2022,))):
        folder = output / f'horizon_{horizon}'
        assert (folder / 'max_iter_search.csv').is_file()
        assert (folder / 'selection_subsample.json').is_file()
        assert (folder / 'comparison.csv').is_file()
        winner = json.loads((folder / 'winner.json').read_text(encoding='utf-8'))
        assert winner['max_iter'] in (2, 4)
        for year in years:
            assert (folder / f'external_prediction_target_{year}.tif').is_file()
            assert (output / f'teste_externo_H{horizon}_{year-horizon}_{year}.png').is_file()
    assert (output / 'teste_externo_canais.csv').is_file()
    assert (output / 'teste_externo_metricas.csv').is_file()
