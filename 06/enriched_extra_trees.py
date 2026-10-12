"""Experiment 06: budgeted Extra Trees search on enriched raster features."""
from __future__ import annotations
from src.utils.source_paths import project_root, source_path

from datetime import datetime, timezone
from itertools import product
from pathlib import Path
import csv
import importlib.metadata
import json
import logging
import math
import platform
import shutil
import sys
import time

import joblib
import numpy as np
import psutil
import yaml
from sklearn.ensemble import ExtraTreesRegressor
from threadpoolctl import threadpool_limits

from src.services import enriched_linear as common
from src.services.enriched_hgb import selection_subsample
from src.splitting.spatial_split import split_grid
from src.geospatial.raster_io import write_raster


BASE_ARMS = ('originais', 'gpa', 'grafos', 'todas')
ARM_BASE = {'originais': 'originais', 'originais_filtradas': 'originais',
            'gpa': 'gpa', 'grafos': 'grafos', 'todas': 'todas',
            'todas_filtradas': 'todas'}
PARAMETERS = ('max_features', 'min_samples_leaf', 'max_leaf_nodes', 'min_samples_split')


def load_config(path: Path) -> dict:
    cfg = common.load_config(path)
    settings = cfg['raw'].get('extra_trees', {})
    if settings.get('criterion') != 'squared_error' or settings.get('bootstrap') is not False:
        raise ValueError('Extra Trees requires squared_error and bootstrap=False')
    if settings.get('oob_score') is not False:
        raise ValueError('OOB cannot replace the spatial/temporal selection stage')
    for key in ('random_state', 'candidate_seed', 'selection_samples_per_stage_year',
                'n_candidates', 'n_finalists', 'screening_trees', 'filtering_trees', 'final_trees'):
        if type(settings.get(key)) is not int or settings[key] < 1:
            raise ValueError(f'extra_trees.{key} must be a positive integer')
    if settings['n_finalists'] > settings['n_candidates']:
        raise ValueError('n_finalists must not exceed n_candidates')
    if not settings['screening_trees'] < settings['final_trees']:
        raise ValueError('The finalist forest must have more trees than the screening forest')
    for key in PARAMETERS:
        values = settings.get(key)
        if not isinstance(values, list) or not values or len(values) != len(set(values)):
            raise ValueError(f'extra_trees.{key} needs distinct candidate values')
        if key == 'max_leaf_nodes':
            if any(type(value) is not int or value < 2 for value in values):
                raise ValueError('max_leaf_nodes must be integers >= 2')
        elif any(not isinstance(value, (int, float)) or
                 not math.isfinite(value) or not 0 < value <= 1 for value in values):
            raise ValueError(f'extra_trees.{key} must contain fractions in (0, 1]')
    reference = settings.get('reference_candidate')
    if not isinstance(reference, dict) or set(reference) != set(PARAMETERS):
        raise ValueError('reference_candidate must contain the four search parameters')
    if any(reference[key] not in settings[key] for key in PARAMETERS):
        raise ValueError('reference_candidate must belong to the search grid')
    total = math.prod(len(settings[key]) for key in PARAMETERS)
    if not 1 <= settings['n_candidates'] <= total:
        raise ValueError('n_candidates exceeds the candidate grid')
    previous = ('enriched_linear', 'enriched_ridge', 'enriched_hgb',
                'enriched_xgboost', 'enriched_random_forest')
    if cfg['output'].resolve() in ((project_root() / 'runs' / item).resolve()
                                   for item in previous):
        raise ValueError('Extra Trees output must not overwrite previous experiments')
    return cfg


def candidates(settings: dict) -> list[dict]:
    reference = tuple(settings['reference_candidate'][key] for key in PARAMETERS)
    grid = [row for row in product(*(settings[key] for key in PARAMETERS)) if row != reference]
    indices = np.random.default_rng(settings['candidate_seed']).permutation(len(grid))
    chosen = [reference] + [grid[int(i)] for i in indices[:settings['n_candidates'] - 1]]
    return [dict(zip(PARAMETERS, row)) for row in chosen]


def check_only(config_path: Path) -> None:
    cfg = load_config(config_path)
    audit = common.audit_catalog(cfg, verify_hashes=False)
    settings = cfg['raw']['extra_trees']
    print(f"PRECHECK OK: {audit['input_and_development_target_files']} inputs/targets; "
          f"24 channels, H0/H1, {len(candidates(settings))} ET candidates per base arm; "
          'external targets not opened.')


def fit_extra_trees(x: np.ndarray, y: np.ndarray, columns: tuple[int, ...],
               params: dict, trees: int, threads: int, seed: int) -> ExtraTreesRegressor:
    if not columns or not len(y):
        raise ValueError('Nonempty feature set and fitting sample required')
    model = ExtraTreesRegressor(
        n_estimators=int(trees), criterion='squared_error', bootstrap=False,
        oob_score=False, n_jobs=int(threads), random_state=int(seed),
        max_features=float(params['max_features']),
        min_samples_leaf=float(params['min_samples_leaf']),
        max_leaf_nodes=int(params['max_leaf_nodes']),
        min_samples_split=float(params['min_samples_split']))
    model.fit(x[:, columns], y)
    return model


def _search_key(stage: str, arm: str, candidate_id: int) -> tuple[str, str, int]:
    return stage, arm, candidate_id


def _write_checkpoint_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def search_hyperparameters(selection: dict, settings: dict, threads: int,
                           directory: Path) -> dict[str, dict]:
    """Checkpoint every trial. Resume completed trials without looking at validation."""
    trial_path = directory / 'hyperparameter_trials.json'
    grid = candidates(settings)
    common._write_json(directory / 'hyperparameter_candidates.json', grid)
    rows = json.loads(trial_path.read_text(encoding='utf-8')) if trial_path.is_file() else []
    known = {}
    for row in rows:
        key = _search_key(row['stage'], row['base_arm'], row['candidate_id'])
        if key in known or row['params'] != grid[row['candidate_id']]:
            raise ValueError('Inconsistent hyperparameter trial checkpoint')
        known[key] = row
    x_fit, y_fit = selection['ajuste'][:2]
    x_score, y_score = selection['selecao_camadas'][:2]

    def evaluate(stage: str, arm: str, candidate_id: int, trees: int) -> dict:
        key = _search_key(stage, arm, candidate_id)
        if key in known:
            return known[key]
        columns = common._arm_indices(arm)
        params = grid[candidate_id]
        started = time.perf_counter()
        model = fit_extra_trees(x_fit, y_fit, columns, params, trees, threads,
                           settings['random_state'])
        rss_after_fit_mb = psutil.Process().memory_info().rss / (1024 ** 2)
        prediction = common._predict(model, x_score, columns)
        prefix_rmse = {}
        if stage == 'refinement':
            prefixes = sorted({settings['screening_trees'],
                               settings['filtering_trees'], settings['final_trees']})
            accumulated = np.zeros(len(y_score), dtype=np.float64)
            features = x_score[:, columns]
            for count, estimator in enumerate(model.estimators_, start=1):
                if count > prefixes[-2]:
                    break
                accumulated += estimator.predict(features)
                if count in prefixes[:-1]:
                    prefix_rmse[str(count)] = common._rmse(y_score, accumulated / count)
            prefix_rmse[str(trees)] = common._rmse(y_score, prediction)
        row = dict(stage=stage, base_arm=arm, candidate_id=candidate_id,
                   params=params, n_estimators=trees,
                   fit_samples=len(y_fit), selection_samples=len(y_score),
                   selection_rmse=common._rmse(y_score, prediction),
                   selection_mae=common._mae(y_score, prediction),
                   fit_and_score_seconds=round(time.perf_counter()-started, 3),
                   rss_after_fit_mb=round(rss_after_fit_mb, 1),
                   mean_tree_nodes=float(np.mean([tree.tree_.node_count
                                                  for tree in model.estimators_])),
                   prefix_rmse=prefix_rmse)
        rows.append(row)
        known[key] = row
        _write_checkpoint_json(trial_path, rows)
        logging.info('ET %s %s candidate %s/%s: RMSE %.6f (%.1f s)',
                     stage, arm, candidate_id+1, len(grid), row['selection_rmse'],
                     row['fit_and_score_seconds'])
        return row

    def rank(row: dict):
        params = row['params']
        return (row['selection_rmse'], row['selection_mae'],
                params['max_leaf_nodes'], -params['min_samples_leaf'],
                row['candidate_id'])

    chosen = {}
    for arm in BASE_ARMS:
        screened = [evaluate('screening', arm, index, settings['screening_trees'])
                    for index in range(len(grid))]
        finalists = sorted(screened, key=rank)[:settings['n_finalists']]
        refined = [evaluate('refinement', arm, row['candidate_id'],
                            settings['final_trees']) for row in finalists]
        best = min(refined, key=rank)
        chosen[arm] = grid[best['candidate_id']]
        logging.info('ET %s chosen candidate %s: RMSE %.6f',
                     arm, best['candidate_id']+1, best['selection_rmse'])
    common._write_json(directory / 'hyperparameter_choices.json', chosen)
    common._write_csv(directory / 'hyperparameter_trials.csv', [
        dict(stage=row['stage'], base_arm=row['base_arm'],
             candidate_id=row['candidate_id'], n_estimators=row['n_estimators'],
             **row['params'], fit_samples=row['fit_samples'],
             selection_samples=row['selection_samples'],
             selection_rmse=row['selection_rmse'], selection_mae=row['selection_mae'],
             fit_and_score_seconds=row['fit_and_score_seconds'],
             rss_after_fit_mb=row.get('rss_after_fit_mb'),
             mean_tree_nodes=row['mean_tree_nodes'],
             prefix_rmse=json.dumps(row.get('prefix_rmse', {}), sort_keys=True))
        for row in rows])
    return chosen


def _training_arm(arm: str, full: dict, selection: dict, cutoff: float,
                  directory: Path, width: int, seed: int, params: dict,
                  settings: dict, threads: int) -> dict:
    folder = directory / arm
    record_path = folder / 'metrics.json'
    if record_path.is_file() and (folder / 'model.joblib').is_file():
        record = json.loads(record_path.read_text(encoding='utf-8'))
        if record['arm'] != arm or record['hyperparameters'] != params:
            raise ValueError(f'Existing ET arm differs from current configuration: {folder}')
        logging.info('ET %s: reused completed arm checkpoint', arm)
        return record
    folder.mkdir(exist_ok=True)
    x0, y0 = full['ajuste'][:2]
    x1, y1 = full['selecao_camadas'][:2]
    xs0, ys0 = selection['ajuste'][:2]
    xs1, ys1 = selection['selecao_camadas'][:2]
    xv, yv, valid_positions, valid_years = full['validacao_modelo']

    def fit_filter(x, y, columns):
        return fit_extra_trees(x, y, columns, params, settings['filtering_trees'],
                          threads, settings['random_state'])

    phenomena = common.PHENOMENA
    attempts = []
    if arm == 'originais_filtradas':
        phenomena, attempts = common.select_phenomena(arm, xs0, ys0, xs1, ys1,
                                                       fit_model=fit_filter)
        columns = common._arm_indices(arm, phenomena)
    elif arm == 'todas_filtradas':
        columns, attempts = common.select_all_features(xs0, ys0, xs1, ys1,
                                                        fit_model=fit_filter)
        phenomena = tuple(name for name in common.PHENOMENA
                          if any(common.FEATURE_NAMES[i].startswith(f'{name}__')
                                 for i in columns))
    else:
        columns = common._arm_indices(arm)

    first_model = fit_filter(xs0, ys0, columns)
    selection_metrics = common.metrics(ys1, common._predict(first_model, xs1, columns), cutoff)
    combined_x = np.concatenate((x0, x1))
    combined_y = np.concatenate((y0, y1))
    model = fit_extra_trees(combined_x, combined_y, columns, params,
                       settings['final_trees'], threads, settings['random_state'])
    rss_after_fit_mb = psutil.Process().memory_info().rss / (1024 ** 2)
    training_prediction = common._predict(model, combined_x, columns)
    training_metrics = common.metrics(combined_y, training_prediction, cutoff)
    valid_prediction = common._predict(model, xv, columns)
    validation_metrics = common.metrics(yv, valid_prediction, cutoff)
    uncertainty = common._bootstrap_block_errors(yv, valid_prediction,
                                                  valid_positions[:, 0],
                                                  valid_positions[:, 1], width,
                                                  seed=seed)
    by_year = []
    for year in sorted(set(valid_years)):
        mask = valid_years == year
        by_year.append(dict(target_year=int(year),
                            metrics=common.metrics(yv[mask], valid_prediction[mask], cutoff)))
    feature_names = [common.FEATURE_NAMES[i] for i in columns]
    joblib.dump(dict(model=model, columns=columns, feature_names=feature_names,
                     phenomena=phenomena, arm=arm, hyperparameters=params,
                     n_estimators=settings['final_trees']), folder / 'model.joblib')
    record = dict(arm=arm, phenomena=list(phenomena), n_features=len(columns),
                  feature_names=feature_names, hyperparameters=params,
                  n_estimators=settings['final_trees'], filtering_trees=settings['filtering_trees'],
                  cpu_threads=threads, model_sha256=common._sha256(folder / 'model.joblib'),
                  rss_after_fit_mb=round(rss_after_fit_mb, 1),
                  mean_tree_nodes=float(np.mean([tree.tree_.node_count
                                                 for tree in model.estimators_])),
                  initial_selection_metrics=selection_metrics,
                  training_metrics=training_metrics,
                  validation_metrics=validation_metrics,
                  validation_uncertainty=uncertainty,
                  validation_by_year=by_year,
                  selection_candidates=attempts,
                  selection_fit_samples=len(ys0),
                  selection_score_samples=len(ys1),
                  training_samples=len(combined_y), validation_samples=len(yv))
    common._write_json(record_path, record)
    logging.info('ET %s: validation RMSE %.6f with %s channels',
                 arm, validation_metrics['rmse'], len(columns))
    return record


def _load_stages(path: Path) -> dict:
    with np.load(path) as sample:
        return {stage: tuple(sample[f'{stage}_{part}'] for part in
                             ('x', 'y', 'positions', 'target_year'))
                for stage in ('ajuste', 'selecao_camadas', 'validacao_modelo')}


def train_horizon(cfg: dict, profile: dict, zones: np.ndarray,
                  horizon: int, directory: Path) -> dict:
    directory.mkdir(exist_ok=True)
    sample_path = directory / 'samples.npz'
    if sample_path.is_file() and (directory / 'sampling.json').is_file():
        full = _load_stages(sample_path)
        logging.info('H%s: reused saved development samples', horizon)
    else:
        collection = common.collect_development(cfg, profile, zones, horizon, sample_path)
        common._write_json(directory / 'sampling.json', collection['counts'])
        full = collection['stages']
    positive = np.concatenate((full['ajuste'][1], full['selecao_camadas'][1]))
    positive = positive[positive > 0]
    if not len(positive):
        raise ValueError(f'H{horizon} has no positive development targets')
    cutoff = float(np.quantile(positive,
                               cfg['raw']['evaluation']['high_target_positive_quantile']))
    training_mean = float(np.mean(np.concatenate((full['ajuste'][1],
                                                  full['selecao_camadas'][1]))))
    valid_target = full['validacao_modelo'][1]
    references = dict(
        zero=common.metrics(valid_target, np.zeros_like(valid_target), subsets=False),
        training_mean=common.metrics(valid_target,
                                     np.full_like(valid_target, training_mean), subsets=False))
    common._write_json(directory / 'validation_references.json', references)
    settings = cfg['raw']['extra_trees']
    selection = selection_subsample(full,
                                    settings['selection_samples_per_stage_year'],
                                    4203 + horizon * 1000, directory)
    chosen = search_hyperparameters(selection, settings,
                                    cfg['raw']['runtime']['cpu_threads'], directory)
    records = []
    for arm in common.ARMS:
        logging.info('H%s: fitting Extra Trees %s', horizon, arm)
        records.append(_training_arm(arm, full, selection, cutoff, directory,
                                     profile['width'],
                                     cfg['raw']['sampling']['seed'] + horizon * 100,
                                     chosen[ARM_BASE[arm]], settings,
                                     cfg['raw']['runtime']['cpu_threads']))
    winner = common._winner(records)
    flat = [dict(horizon=horizon, arm=row['arm'],
                 phenomena=';'.join(row['phenomena']),
                 n_features=row['n_features'],
                 n_estimators=row['n_estimators'],
                 selection_fit_samples=row['selection_fit_samples'],
                 selection_score_samples=row['selection_score_samples'],
                 initial_selection_rmse=row['initial_selection_metrics']['rmse'],
                 training_rmse=row['training_metrics']['rmse'],
                 validation_rmse=row['validation_metrics']['rmse'],
                 validation_mae=row['validation_metrics']['mae'],
                 validation_r2=row['validation_metrics']['r2'],
                 validation_nmad=row['validation_metrics']['nmad'],
                 validation_rmse_ci95_low=row['validation_uncertainty']['rmse_ci95'][0],
                 validation_rmse_ci95_high=row['validation_uncertainty']['rmse_ci95'][1],
                 validation_capture_10pct=row['validation_metrics']['capture_10pct'])
            for row in records]
    common._write_csv(directory / 'comparison.csv', flat)
    common._write_csv(directory / 'selection_candidates.csv', [
        dict(arm=row['arm'], step=row.get('step'), removed_group=row.get('removed_group'),
             accepted=row.get('accepted'), phenomena=';'.join(row['phenomena']),
             features=';'.join(row['features']), n_features=row['n_features'],
             selection_rmse=row['selection_rmse'], selection_mae=row['selection_mae'])
        for record in records for row in record['selection_candidates']])
    selected = dict(horizon=horizon, arm=winner['arm'],
                    phenomena=winner['phenomena'], feature_names=winner['feature_names'],
                    n_features=winner['n_features'], n_estimators=winner['n_estimators'],
                    hyperparameters=winner['hyperparameters'],
                    validation_metrics=winner['validation_metrics'],
                    high_target_threshold=cutoff, training_target_mean=training_mean,
                    model_path=str(directory / winner['arm'] / 'model.joblib'))
    common._write_json(directory / 'winner.json', selected)
    logging.info('H%s ET winner: %s; validation RMSE %.6f',
                 horizon, winner['arm'], winner['validation_metrics']['rmse'])
    return dict(winner=selected, records=records, references=references)


def _methodology_appendix(run_dir: Path) -> None:
    settings = yaml.safe_load((run_dir / 'config.yaml').read_text(encoding='utf-8'))['extra_trees']
    lines = [
        '## Extra Trees e exploração de hiperparâmetros', '',
        'O experimento 06 usa `ExtraTreesRegressor` com critério de erro quadrático, '
        'sem bootstrap e com quatro threads no máximo. As quatro famílias não filtradas '
        'recebem exatamente o mesmo orçamento: configurações sorteadas com semente fixa '
        f"e avaliadas nas etapas internas, primeiro com {settings['screening_trees']} "
        f"e depois com {settings['final_trees']} árvores. "
        f"Os {settings['n_finalists']} melhores candidatos da triagem são reavaliados; o RMSE decide, "
        'seguido por MAE e menor complexidade. Braços filtrados herdam a configuração '
        f"da família correspondente e testam grupos de canais com {settings['filtering_trees']} árvores. "
        f"Os seis braços finais são reajustados com {settings['final_trees']} árvores nas amostras completas.", '',
        f"Semente de árvores: {settings['random_state']}; sorteio de candidatos: "
        f"{settings['candidate_seed']}; máximo de "
        f"{settings['selection_samples_per_stage_year']:,} amostras por ano e etapa "
        'na busca. `hyperparameter_candidates.json` fixa a lista sorteada antes do ajuste; '
        '`hyperparameter_trials.csv` e `.json` preservam cada tentativa, RMSE, MAE, '
        'tempo, RSS do processo após ajuste, complexidade média e RMSE por prefixo '
        'de árvores na etapa de refinamento. RSS não representa pico de memória. '
        '`hyperparameter_choices.json` contém as escolhas. '
        'Índices e hashes da subamostra estão em `selection_subsample_indices.npz` e '
        '`selection_subsample.json`. O erro out-of-bag não substitui a separação '
        'espacial/temporal. A área externa não orienta parâmetros, camadas ou sementes.', '',
        '| Horizonte | Família base | max_features | min_samples_leaf | max_leaf_nodes | min_samples_split |',
        '|---|---|---:|---:|---:|---:|',
    ]
    for horizon in (0, 1):
        chosen = json.loads((run_dir / f'horizon_{horizon}' /
                             'hyperparameter_choices.json').read_text(encoding='utf-8'))
        for arm in BASE_ARMS:
            p = chosen[arm]
            lines.append(f"| H{horizon} | {arm} | {p['max_features']} | "
                         f"{p['min_samples_leaf']} | {p['max_leaf_nodes']} | "
                         f"{p['min_samples_split']} |")
    lines += ['', '### Todas as tentativas de hiperparâmetros', '',
              'Cada horizonte contém triagem e refinamento para as quatro famílias base. '
              'O RMSE e MAE abaixo vêm apenas de `selecao_camadas`; as curvas por '
              'prefixo e os números sem arredondamento estão nos CSVs e JSONs de cada horizonte.', '']
    for horizon in (0, 1):
        trials = json.loads((run_dir / f'horizon_{horizon}' /
                             'hyperparameter_trials.json').read_text(encoding='utf-8'))
        lines += [f'#### H{horizon} — {len(trials)} tentativas', '',
                  '| Família | Etapa | Candidato | Árvores | max_features | min_samples_leaf | max_leaf_nodes | min_samples_split | RMSE seleção | MAE seleção | Tempo (s) | RSS após ajuste (MB) |',
                  '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for row in trials:
            p = row['params']
            lines.append(f"| {row['base_arm']} | {row['stage']} | {row['candidate_id'] + 1} | "
                         f"{row['n_estimators']} | {p['max_features']} | "
                         f"{p['min_samples_leaf']} | {p['max_leaf_nodes']} | "
                         f"{p['min_samples_split']} | {row['selection_rmse']:.4f} | "
                         f"{row['selection_mae']:.4f} | {row['fit_and_score_seconds']:.1f} | "
                         f"{row.get('rss_after_fit_mb', '—')} |")
        lines.append('')
    lines += [
        '', '### Conceitos e fontes', '',
        '| Conceito | Fonte e aplicação |', '|---|---|',
        '| Árvores extremamente aleatorizadas | [Geurts, Ernst e Wehenkel (2006), artigo original](https://doi.org/10.1007/s10994-006-6226-1). |',
        '| Parâmetros, divisões aleatórias e ausência de bootstrap | [ExtraTreesRegressor, scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html). |',
        '| Busca aleatória com orçamento fixo | [RandomizedSearchCV, scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html); as divisões efetivas são as do projeto. |',
        '| Separação espacial e temporal | [GroupKFold, scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html), [Valavi et al. (2019)](https://doi.org/10.1111/2041-210X.13107) e [Stock (2025)](https://doi.org/10.3389/frsen.2025.1531097). |',
        '| Importância de canais correlacionados | [Exemplo oficial de importância por permutação com colinearidade](https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html); a seleção efetiva usa os grupos e o RMSE interno. |',
        '| RMSE, MAE, R² | [Métricas de regressão, scikit-learn](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics). |',
        '| NMAD, blocos, rasters, GPA, grafos e figuras | Definições e referências no [protocolo de experimentos enriquecidos](protocols/EXPERIMENTOS_ENRIQUECIDOS_2026.md) e em [protocolo ET](protocols/EXPERIMENTO_06_EXTRA_TREES.md). |',
        '| CE90 omitido | [USGS (2024), acurácia posicional](https://pubs.usgs.gov/publication/ofr20241023/full); não é métrica de erro de intensidade. |', '',
        'A grade, os seis braços e as regras de seleção são decisões pré-definidas '
        'deste projeto; as referências fundamentam os conceitos gerais. '
        'A avaliação H1 2020→2021, quando presente, aparece em pasta suplementar '
        'e não participa da escolha do vencedor.', '',
    ]
    (run_dir / 'methodology_appendix.md').write_text('\n'.join(lines), encoding='utf-8')


def rebuild_report(run_dir: Path) -> None:
    from plot_enriched_linear_external import render_one
    from report_enriched_linear_all import run_report

    run_dir = run_dir.resolve()
    status_path = run_dir / 'run.json'
    status = json.loads(status_path.read_text(encoding='utf-8'))
    if status.get('kind') != 'enriched_extra_trees' or not status.get('final_external_test_used'):
        raise ValueError('Expected an externally evaluated Extra Trees run')
    items = []
    for horizon in (0, 1):
        paths = sorted((run_dir / f'horizon_{horizon}').glob('external_target_*_metrics.json'))
        if len(paths) != (3 if horizon == 0 else 1):
            raise ValueError('Original external metric files are incomplete')
        for path in paths:
            items.append(render_one(json.loads(path.read_text(encoding='utf-8')), run_dir))
    supplement = run_dir / 'supplemental_H1_2021'
    supplement_record_path = supplement / 'external_target_2021_metrics.json'
    if not supplement_record_path.is_file():
        raise ValueError('Supplemental H1 2020→2021 metrics are missing')
    supplement_record = json.loads(supplement_record_path.read_text(encoding='utf-8'))
    supplement_figure = render_one(supplement_record, supplement)
    items.append(supplement_figure)
    common._write_json(run_dir / 'figuras_teste_externo.json', items)
    _methodology_appendix(run_dir)
    status['status'] = 'completed'
    common._write_json(status_path, status)
    run_report(run_dir, run_dir)
    shutil.copy2(run_dir / 'TODOS_RESULTADOS.md', run_dir / 'RELATORIO.md')
    m = supplement_record['metrics']
    addendum = [
        '## Avaliação complementar pré-definida: target externo de 2021 em H1', '',
        'Depois de congelar o vencedor H1, o mesmo modelo foi aplicado às entradas externas '
        'de 2020 para prever o target externo de 2021. Esta avaliação complementa o par '
        'original H1 2021→2022 e não foi usada para escolher parâmetros ou canais.', '',
        f"Pixels comuns: {supplement_record['common_pixels']:,}; "
        f"positivos: {100*(1-m['target_zero']/m['n']):.2f}%. "
        f"RMSE: {m['rmse']:.4f}; MAE: {m['mae']:.4f}; R²: {m['r2']:.4f}; "
        f"captura top 10%: {100*m['capture_10pct']:.2f}%.", '',
        f"Referência zero: RMSE {supplement_record['reference_zero']['rmse']:.4f}, "
        f"MAE {supplement_record['reference_zero']['mae']:.4f}. "
        f"Persistência: RMSE {supplement_record['reference_persistence']['metrics']['rmse']:.4f}, "
        f"MAE {supplement_record['reference_persistence']['metrics']['mae']:.4f}.", '',
        '[GeoTIFF da previsão H1 2020→2021](supplemental_H1_2021/external_prediction_target_2021.tif)', '',
        '![Target original e previsão H1 2020→2021](supplemental_H1_2021/teste_externo_H1_2020_2021.png)', '',
    ]
    with (run_dir / 'RELATORIO.md').open('a', encoding='utf-8') as handle:
        handle.write('\n' + '\n'.join(addendum))
    if status.get('finished'):
        from audit_enriched_training_reports import try_enrich_run
        try_enrich_run(run_dir)
    print(run_dir / 'RELATORIO.md')


def _snapshot(cfg: dict, run_dir: Path) -> None:
    shutil.copy2(cfg['path'], run_dir / 'config.yaml')
    project = project_root()
    sources = ('run_enriched_extra_trees.py', 'audit_enriched_training_reports.py',
               'src/services/enriched_extra_trees.py',
               'src/services/enriched_hgb.py', 'src/services/enriched_linear.py',
               'src/services/temporal_service.py', 'src/splitting/spatial_split.py',
               'src/geospatial/raster_io.py', 'plot_enriched_linear_external.py',
               'report_enriched_linear_all.py', 'generate_gradient_inputs.py',
               'generate_graph_inputs.py')
    source_records = []
    for relative in sources:
        original = source_path(relative)
        destination = run_dir / 'source_code' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, destination)
        source_records.append(dict(path=relative, sha256=common._sha256(original)))
    common._write_json(run_dir / 'source_files.json', source_records)
    for relative in ('protocols/EXPERIMENTOS_ENRIQUECIDOS_2026.md',
                     'protocols/EXPERIMENTO_06_EXTRA_TREES.md',
                     'docs/GRADIENT_INPUTS.md', 'docs/GRAPH_INPUTS.md'):
        original = source_path(relative)
        destination = run_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, destination)
    packages = {name: importlib.metadata.version(name) for name in
                ('numpy', 'scipy', 'scikit-learn', 'rasterio', 'joblib',
                 'PyYAML', 'threadpoolctl', 'matplotlib', 'psutil')}
    common._write_json(run_dir / 'environment.json',
                       dict(python=sys.version, platform=platform.platform(),
                            packages=packages))


def run(config_path: Path, resume: Path | None = None) -> None:
    cfg = load_config(config_path)
    cfg['output'].mkdir(parents=True, exist_ok=True)
    if resume is None:
        run_dir = cfg['output'] / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        run_dir.mkdir(exist_ok=False)
        status = dict(kind='enriched_extra_trees', status='running',
                      started=datetime.now(timezone.utc).isoformat(), run=str(run_dir),
                      final_external_test_used=False)
        common._write_json(run_dir / 'run.json', status)
    else:
        run_dir = resume.resolve()
        if not run_dir.is_dir() or run_dir.parent != cfg['output'].resolve():
            raise ValueError('Resume path must identify an ET run inside its output root')
        status = json.loads((run_dir / 'run.json').read_text(encoding='utf-8'))
        if status.get('kind') != 'enriched_extra_trees' or status.get('status') == 'completed':
            raise ValueError('Only an unfinished Extra Trees run can be resumed')
        if not (run_dir / 'config.yaml').is_file() or (run_dir / 'config.yaml').read_bytes() != cfg['path'].read_bytes():
            raise ValueError('Configuration changed since the run began')
        status['status'] = 'running'
        status['resumed'] = datetime.now(timezone.utc).isoformat()
        common._write_json(run_dir / 'run.json', status)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    file_handler = logging.FileHandler(run_dir / 'training.log', mode='a', encoding='utf-8')
    stream_handler = logging.StreamHandler()
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    try:
        if not (run_dir / 'config.yaml').is_file() or not (run_dir / 'source_files.json').is_file():
            _snapshot(cfg, run_dir)
        if not (run_dir / 'input_audit.json').is_file():
            total = 2 * 4 * len(common.FEATURE_NAMES) + 3
            logging.info('ET input SHA-256 audit: %s files before training', total)

            def progress(count: int, relative: str) -> None:
                if count == 1 or count % 10 == 0 or count == total:
                    logging.info('ET input audit %s/%s: %s', count, total, relative)

            audit = common.audit_catalog(cfg, verify_hashes=True, on_verified=progress)
            common._write_json(run_dir / 'input_audit.json', audit)
        else:
            audit = json.loads((run_dir / 'input_audit.json').read_text(encoding='utf-8'))
            if not audit['hashes_recomputed']:
                raise ValueError('Saved input audit lacks SHA-256 verification')
        profile = common._profile(cfg['reference'])
        zones = split_grid((profile['height'], profile['width']), cfg['raw']['split'])
        if not (run_dir / 'spatial_zones.tif').is_file():
            write_raster(run_dir / 'spatial_zones.tif', zones, profile, categorical=True)
        common._write_json(run_dir / 'grid.json', audit['reference'])
        horizons = {}
        with threadpool_limits(limits=cfg['raw']['runtime']['cpu_threads']):
            for horizon in (0, 1):
                horizons[horizon] = train_horizon(cfg, profile, zones, horizon,
                                                  run_dir / f'horizon_{horizon}')
            status['selection_finished'] = datetime.now(timezone.utc).isoformat()
            common._write_json(run_dir / 'run.json', status)
            for horizon in (0, 1):
                directory = run_dir / f'horizon_{horizon}'
                years = cfg['raw']['evaluation'][f'final_h{horizon}_target_years']
                if not all((directory / f'external_target_{year}_metrics.json').is_file()
                           for year in years):
                    common.evaluate_external(cfg, horizons[horizon]['winner'], horizon,
                                             directory)
            supplement = run_dir / 'supplemental_H1_2021'
            supplement.mkdir(exist_ok=True)
            if not (supplement / 'external_target_2021_metrics.json').is_file():
                cfg['raw']['evaluation']['final_h1_target_years'] = [2021]
                common.evaluate_external(cfg, horizons[1]['winner'], 1, supplement)
        status.update(status='external_complete',
                      external_finished=datetime.now(timezone.utc).isoformat(),
                      final_external_test_used=True,
                      winners={str(h): horizons[h]['winner']['arm'] for h in (0, 1)})
        common._write_json(run_dir / 'run.json', status)
        rebuild_report(run_dir)
        status['status'] = 'completed'
        status['finished'] = datetime.now(timezone.utc).isoformat()
        common._write_json(run_dir / 'run.json', status)
        from audit_enriched_training_reports import try_enrich_run
        try_enrich_run(run_dir)
    except KeyboardInterrupt:
        status.update(status='interrupted', interrupted=datetime.now(timezone.utc).isoformat())
        common._write_json(run_dir / 'run.json', status)
        logging.warning('ET run interrupted; resume with --resume %s', run_dir)
        raise SystemExit(130)
    except Exception as exc:
        status.update(status='failed', failed=datetime.now(timezone.utc).isoformat(),
                      error=repr(exc))
        common._write_json(run_dir / 'run.json', status)
        logging.exception('ET run failed; partial artifacts kept at %s', run_dir)
        raise
    finally:
        logger.removeHandler(file_handler)
        logger.removeHandler(stream_handler)
        file_handler.close()
        stream_handler.close()
