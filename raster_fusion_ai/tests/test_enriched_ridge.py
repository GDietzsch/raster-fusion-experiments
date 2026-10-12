"""Verify the Ridge tuning boundary and complete report on tiny rasters."""
import csv
import json
from pathlib import Path

import numpy as np
import yaml

from src.services.enriched_ridge import fit_ridge, run, tune_lambda
from test_enriched_linear import _synthetic_dataset


def test_penalty_scales_with_sample_count():
    x = np.arange(60, dtype=float).reshape(20, 3)
    y = np.arange(20, dtype=float)
    first = fit_ridge(x[:10], y[:10], (0, 1), 0.01)
    second = fit_ridge(x, y, (0, 1), 0.01)
    assert first.named_steps['ridge'].alpha == 0.1
    assert second.named_steps['ridge'].alpha == 0.2


def test_lambda_search_uses_only_fit_and_selection(tmp_path):
    rng = np.random.default_rng(5)
    x0 = rng.normal(size=(30, 24))
    x1 = rng.normal(size=(15, 24))
    stage = {'ajuste': (x0, 2*x0[:, 0], None, None),
             'selecao_camadas': (x1, 2*x1[:, 0], None, None)}
    chosen = tune_lambda(stage, [0.0001, 0.01], tmp_path)
    assert set(chosen) == {'originais', 'gpa', 'grafos', 'todas'}
    with (tmp_path / 'alpha_search.csv').open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 8
    assert sum(row['chosen'] == 'True' for row in rows) == 4
    assert all(float(row['alpha_initial']) == 30*float(row['lambda_per_sample'])
               for row in rows)


def test_full_synthetic_ridge_run_with_maps_and_references(tmp_path):
    config_path = _synthetic_dataset(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    config['ridge'] = dict(solver='cholesky',
                           lambda_per_sample=[0.0001, 0.01])
    config['output_root'] = 'ridge_runs'
    config_path.write_text(yaml.safe_dump(config), encoding='utf-8')
    run(config_path)
    output = next((tmp_path / 'ridge_runs').iterdir())
    status = json.loads((output / 'run.json').read_text(encoding='utf-8'))
    assert status['status'] == 'completed'
    assert status['final_external_test_used']
    report = (output / 'RELATORIO.md').read_text(encoding='utf-8')
    assert 'Regressão Ridge' in report
    assert 'Hoerl e Kennard' in report
    assert 'Stock (2025)' in report
    assert 'Lahiri (2003)' in report
    assert '### Imagens do teste externo' in report
    for horizon, years in ((0, (2020, 2021, 2022)), (1, (2022,))):
        folder = output / f'horizon_{horizon}'
        assert (folder / 'alpha_search.csv').is_file()
        assert (folder / 'comparison.csv').is_file()
        winner = json.loads((folder / 'winner.json').read_text(encoding='utf-8'))
        assert winner['lambda_per_sample'] in (0.0001, 0.01)
        for year in years:
            assert (folder / f'external_prediction_target_{year}.tif').is_file()
            assert (output / f'teste_externo_H{horizon}_{year-horizon}_{year}.png').is_file()
    assert (output / 'teste_externo_canais.csv').is_file()
    assert (output / 'teste_externo_metricas.csv').is_file()
