"""Experiment 04: XGBoost on the fixed enriched raster protocol.

Internal stages choose boosting rounds and feature subsets. All six arms are
refitted on the full development samples. External targets are read only after
both horizon winners have been frozen.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import csv
import importlib.metadata
import json
import logging
import math
import platform
import shutil
import sys

import joblib
import numpy as np
import xgboost
import yaml
from xgboost import XGBRegressor
from threadpoolctl import threadpool_limits

from src.services import enriched_linear as common
from src.services.enriched_hgb import selection_subsample
from src.splitting.spatial_split import split_grid
from src.geospatial.raster_io import write_raster


BASE_ARMS = ('originais', 'gpa', 'grafos', 'todas')
ARM_BASE = {'originais': 'originais', 'originais_filtradas': 'originais',
            'gpa': 'gpa', 'grafos': 'grafos', 'todas': 'todas',
            'todas_filtradas': 'todas'}


def load_config(path: Path) -> dict:
    cfg = common.load_config(path)
    settings = cfg['raw'].get('xgboost', {})
    fixed = {'objective': 'reg:squarederror', 'tree_method': 'hist',
             'device': 'cpu', 'eval_metric': 'rmse'}
    for key, value in fixed.items():
        if settings.get(key) != value:
            raise ValueError(f'xgboost.{key} must be {value}')
    if settings.get('early_stopping_rounds', 'missing') is not None:
        raise ValueError('No internal random early stopping is allowed')
    grid = settings.get('n_estimators_candidates')
    if (not isinstance(grid, list) or len(grid) < 2 or
        any(type(value) is not int or value < 1 for value in grid) or
        grid != sorted(set(grid))):
        raise ValueError('n_estimators_candidates must be sorted distinct positive integers')
    for key in ('max_depth', 'min_child_weight', 'max_bin',
                'selection_samples_per_stage_year'):
        if type(settings.get(key)) is not int or settings[key] < 1:
            raise ValueError(f'xgboost.{key} must be a positive integer')
    if settings['max_bin'] > 255:
        raise ValueError('max_bin must be <=255')
    for key in ('selection_seed', 'random_state'):
        if type(settings.get(key)) is not int:
            raise ValueError(f'xgboost.{key} must be an integer')
    for key in ('learning_rate', 'reg_lambda', 'reg_alpha', 'gamma',
                'subsample', 'colsample_bytree'):
        value = settings.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'xgboost.{key} must be finite')
        if (key == 'learning_rate' and value <= 0 or
            key in ('reg_lambda', 'reg_alpha', 'gamma') and value < 0 or
            key in ('subsample', 'colsample_bytree') and not 0 < value <= 1):
            raise ValueError(f'xgboost.{key} is outside its allowed range')
    previous = ('enriched_linear', 'enriched_ridge', 'enriched_hgb')
    if cfg['output'].resolve() in ((Path(__file__).parents[2] / 'runs' / item).resolve()
                                   for item in previous):
        raise ValueError('XGBoost output must not overwrite earlier experiments')
    return cfg


def check_only(config_path: Path) -> None:
    cfg = load_config(config_path)
    audit = common.audit_catalog(cfg, verify_hashes=False)
    print(f"PRECHECK OK: {audit['input_and_development_target_files']} inputs/targets; "
          f"24 channels, H0/H1, {len(cfg['raw']['xgboost']['n_estimators_candidates'])} "
          f"boosting-round candidates; xgboost {xgboost.__version__}. "
          'External targets not opened.')


def fit_xgboost(x: np.ndarray, y: np.ndarray, columns: tuple[int, ...],
                n_estimators: int, settings: dict, threads: int) -> XGBRegressor:
    if not columns or not len(y):
        raise ValueError('Nonempty features and fitting sample required')
    model = XGBRegressor(
        objective='reg:squarederror', tree_method='hist', device='cpu',
        eval_metric='rmse', learning_rate=float(settings['learning_rate']),
        n_estimators=int(n_estimators), max_depth=int(settings['max_depth']),
        min_child_weight=int(settings['min_child_weight']),
        reg_lambda=float(settings['reg_lambda']),
        reg_alpha=float(settings['reg_alpha']), gamma=float(settings['gamma']),
        subsample=float(settings['subsample']),
        colsample_bytree=float(settings['colsample_bytree']),
        max_bin=int(settings['max_bin']), random_state=int(settings['random_state']),
        n_jobs=int(threads), enable_categorical=False, validate_parameters=True,
        early_stopping_rounds=None, verbosity=0)
    model.fit(x[:, columns], y, verbose=False)
    return model


def tune_estimators(selection: dict, grid: list[int], settings: dict,
                    threads: int, directory: Path) -> dict[str, int]:
    x0, y0 = selection['ajuste'][:2]
    x1, y1 = selection['selecao_camadas'][:2]
    rows, chosen = [], {}
    for arm in BASE_ARMS:
        columns = common._arm_indices(arm)
        candidates = []
        for count in grid:
            model = fit_xgboost(x0, y0, columns, count, settings, threads)
            prediction = common._predict(model, x1, columns)
            row = dict(base_arm=arm, n_estimators=count,
                       actual_rounds=model.get_booster().num_boosted_rounds(),
                       fit_samples=len(y0), selection_samples=len(y1),
                       selection_rmse=common._rmse(y1, prediction),
                       selection_mae=common._mae(y1, prediction), chosen=False)
            rows.append(row)
            candidates.append(row)
        best = min(candidates, key=lambda row: (row['selection_rmse'],
                                                row['selection_mae'], row['n_estimators']))
        best['chosen'] = True
        chosen[arm] = best['n_estimators']
        logging.info('XGBoost %s: %s rounds; selection RMSE %.6f',
                     arm, best['n_estimators'], best['selection_rmse'])
    common._write_csv(directory / 'n_estimators_search.csv', rows)
    common._write_json(directory / 'n_estimators_choices.json', chosen)
    return chosen


def _training_arm(arm: str, full: dict, selection: dict, cutoff: float,
                  directory: Path, width: int, seed: int, n_estimators: int,
                  settings: dict, threads: int) -> dict:
    x0, y0 = full['ajuste'][:2]
    x1, y1 = full['selecao_camadas'][:2]
    xs0, ys0 = selection['ajuste'][:2]
    xs1, ys1 = selection['selecao_camadas'][:2]
    xv, yv, valid_positions, valid_years = full['validacao_modelo']

    def fit(x, y, columns):
        return fit_xgboost(x, y, columns, n_estimators, settings, threads)

    phenomena = common.PHENOMENA
    candidates = []
    if arm == 'originais_filtradas':
        phenomena, candidates = common.select_phenomena(arm, xs0, ys0, xs1, ys1,
                                                         fit_model=fit)
        columns = common._arm_indices(arm, phenomena)
    elif arm == 'todas_filtradas':
        columns, candidates = common.select_all_features(xs0, ys0, xs1, ys1,
                                                          fit_model=fit)
        phenomena = tuple(name for name in common.PHENOMENA
                          if any(common.FEATURE_NAMES[i].startswith(f'{name}__')
                                 for i in columns))
    else:
        columns = common._arm_indices(arm)

    initial = fit(xs0, ys0, columns)
    selection_metrics = common.metrics(ys1, common._predict(initial, xs1, columns), cutoff)
    combined_x = np.concatenate((x0, x1))
    combined_y = np.concatenate((y0, y1))
    model = fit(combined_x, combined_y, columns)
    training_metrics = common.metrics(combined_y,
                                      common._predict(model, combined_x, columns), cutoff)
    valid_prediction = common._predict(model, xv, columns)
    validation_metrics = common.metrics(yv, valid_prediction, cutoff)
    uncertainty = common._bootstrap_block_errors(yv, valid_prediction,
                                                  valid_positions[:, 0], valid_positions[:, 1],
                                                  width, seed=seed)
    by_year = []
    for year in sorted(set(valid_years)):
        mask = valid_years == year
        by_year.append(dict(target_year=int(year),
                            metrics=common.metrics(yv[mask], valid_prediction[mask], cutoff)))
    folder = directory / arm
    folder.mkdir()
    feature_names = [common.FEATURE_NAMES[i] for i in columns]
    model.save_model(folder / 'model.ubj')
    joblib.dump(dict(model=model, columns=columns, feature_names=feature_names,
                     phenomena=phenomena, arm=arm, n_estimators=n_estimators),
                folder / 'model.joblib')
    record = dict(arm=arm, phenomena=list(phenomena), n_features=len(columns),
                  feature_names=feature_names, n_estimators=n_estimators,
                  actual_rounds=model.get_booster().num_boosted_rounds(),
                  estimator_parameters={key: settings[key] for key in settings},
                  cpu_threads=threads, native_model_sha256=common._sha256(folder / 'model.ubj'),
                  initial_selection_metrics=selection_metrics,
                  training_metrics=training_metrics,
                  validation_metrics=validation_metrics,
                  validation_uncertainty=uncertainty,
                  validation_by_year=by_year,
                  selection_candidates=candidates,
                  selection_fit_samples=len(ys0), selection_score_samples=len(ys1),
                  training_samples=len(combined_y), validation_samples=len(yv))
    common._write_json(folder / 'metrics.json', record)
    return record


def train_horizon(cfg: dict, profile: dict, zones: np.ndarray,
                  horizon: int, directory: Path) -> dict:
    directory.mkdir()
    collection = common.collect_development(cfg, profile, zones, horizon,
                                             directory / 'samples.npz')
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
    settings = cfg['raw']['xgboost']
    selection = selection_subsample(full,
                                    settings['selection_samples_per_stage_year'],
                                    settings['selection_seed'] + horizon * 1000, directory)
    rounds = tune_estimators(selection, settings['n_estimators_candidates'],
                             settings, cfg['raw']['runtime']['cpu_threads'], directory)
    records = []
    for arm in common.ARMS:
        logging.info('H%s: fitting XGBoost %s', horizon, arm)
        records.append(_training_arm(arm, full, selection, cutoff, directory,
                                     profile['width'],
                                     cfg['raw']['sampling']['seed'] + horizon * 100,
                                     rounds[ARM_BASE[arm]], settings,
                                     cfg['raw']['runtime']['cpu_threads']))
    winner = common._winner(records)
    flat = [dict(horizon=horizon, arm=row['arm'],
                 phenomena=';'.join(row['phenomena']), n_features=row['n_features'],
                 n_estimators=row['n_estimators'], actual_rounds=row['actual_rounds'],
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
                    validation_metrics=winner['validation_metrics'],
                    high_target_threshold=cutoff, training_target_mean=training_mean,
                    model_path=str(directory / winner['arm'] / 'model.joblib'))
    common._write_json(directory / 'winner.json', selected)
    logging.info('H%s winner: %s; RMSE %.6f; rounds %s',
                 horizon, winner['arm'], winner['validation_metrics']['rmse'],
                 winner['n_estimators'])
    return dict(winner=selected, records=records, references=references)


def _methodology_appendix(run_dir: Path) -> None:
    settings = yaml.safe_load((run_dir / 'config.yaml').read_text(encoding='utf-8'))['xgboost']
    cap = settings['selection_samples_per_stage_year']
    lines = [
        '## XGBoost e referências metodológicas', '',
        'O estimador é `XGBRegressor` (XGBoost) com `reg:squarederror`, método de '
        'árvores `hist` em CPU e avaliação RMSE. Todos os canais são numéricos, '
        'sem padronização. O target mantém suas unidades; previsões negativas '
        'não são truncadas. `early_stopping_rounds=None` e nenhum `eval_set` '
        'impedem que XGBoost crie uma seleção adicional fora das etapas fixadas.', '',
        f'A busca de iterações e de camadas usa até {cap:,} pixels por ano em '
        'cada etapa interna, sorteados sem reposição de forma fixa e compartilhada '
        'por todos os candidatos. A mesma semente e os mesmos índices do '
        'experimento 03 são usados quando os `samples.npz` coincidem. Os índices, '
        'hashes e contagens estão em `selection_subsample_indices.npz` e '
        '`selection_subsample.json`. Os seis modelos finais são reajustados '
        'nas amostras completas de ajuste mais seleção; a validação completa '
        'escolhe o vencedor de cada horizonte. O RMSE de seleção usa uma '
        'subamostra diferente da dos experimentos lineares, mas a validação '
        'usa a mesma amostragem de todos os experimentos.', '',
        'A grade `n_estimators_candidates` é testada separadamente nas quatro '
        'famílias não filtradas (originais, GPA, grafos e todas). O menor RMSE '
        'da etapa interna escolhe as rodadas; empates usam MAE e menos árvores. '
        '`originais_filtradas` herda a escolha de originais, e '
        '`todas_filtradas` a de todas. A filtragem mantém pares de direção '
        'e de grafo juntos. A área externa não participa de nenhuma escolha.', '',
        '| Horizonte | Família base | Árvores escolhidas | RMSE na subamostra de seleção |',
        '|---|---|---:|---:|',
    ]
    for horizon in (0, 1):
        with (run_dir / f'horizon_{horizon}' / 'n_estimators_search.csv').open(
                encoding='utf-8-sig', newline='') as handle:
            for row in csv.DictReader(handle):
                if row['chosen'] == 'True':
                    lines.append(f"| H{horizon} | {row['base_arm']} | {row['n_estimators']} | "
                                 f"{float(row['selection_rmse']):.4f} |")
    lines += [
        '', 'Os demais parâmetros fixos e seus valores estão em `config.yaml` '
        'e em cada `metrics.json`: taxa de aprendizado, profundidade máxima, '
        'peso mínimo de folha, regularizações L1/L2, redução mínima de perda, '
        'amostragem de linhas/colunas, bins e sementes. Os arquivos '
        '`n_estimators_search.csv` preservam todas as candidatas. A grade é '
        'uma escolha experimental restrita, não garantia de ótimo global. '
        'Cada braço salva `model.joblib` para a inferência exata desta execução '
        'e `model.ubj` no formato nativo XGBoost, com SHA-256 no `metrics.json`.', '',
        '### Conceitos e fontes', '',
        '| Conceito empregado | Fonte e aplicação neste experimento |', '|---|---|',
        '| XGBoost, árvores e regularização | [Chen e Guestrin (2016), artigo original](https://doi.org/10.1145/2939672.2939785); [parâmetros oficiais](https://xgboost.readthedocs.io/en/stable/parameter.html). |',
        '| API XGBRegressor e método hist | [Referência oficial da API Python](https://xgboost.readthedocs.io/en/stable/python/python_api.html) e [parâmetros de árvore](https://xgboost.readthedocs.io/en/stable/parameter.html), XGBoost. |',
        '| Salvamento UBJSON do modelo | [Guia oficial de Model IO](https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html), XGBoost. |',
        '| Divisão espacial/temporal e validação separada | [Valavi et al. (2019), blockCV](https://doi.org/10.1111/2041-210X.13107); [Stock (2025)](https://doi.org/10.3389/frsen.2025.1531097); [guia de validação do scikit-learn](https://scikit-learn.org/stable/modules/cross_validation.html). |',
        '| Seleção reversa de grupos | [SequentialFeatureSelector, referência conceitual](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SequentialFeatureSelector.html). A regra efetiva do projeto está em `src/services/enriched_linear.py` e os candidatos em `selection_candidates.csv`. |',
        '| RMSE, MAE e R² | [Métricas de regressão](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics), scikit-learn. Viés, P90 e captura top-k seguem o protocolo local. |',
        '| NMAD dos resíduos | [Median absolute deviation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.median_abs_deviation.html), SciPy; o fator 1,4826 e a aplicação aos resíduos estão no protocolo. |',
        '| Intervalos exploratórios por blocos | [Lahiri (2003), Resampling Methods for Dependent Data](https://link.springer.com/book/10.1007/978-1-4757-3803-2), cap. 12. Uma única região externa não fornece regiões independentes. |',
        '| Alinhamento de rasters | [WarpedVRT](https://rasterio.readthedocs.io/en/stable/api/rasterio.vrt.html) e [reamostragem](https://rasterio.readthedocs.io/en/stable/topics/resampling.html), Rasterio. |',
        '| Gradientes chamados GPA neste projeto | [numpy.gradient](https://numpy.org/doc/stable/reference/generated/numpy.gradient.html) e `docs/GRADIENT_INPUTS.md`; não são G1/G2. |',
        '| Grafos espaciais | [SciPy cKDTree](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html) e `docs/GRAPH_INPUTS.md`; pesos e raio são definições locais. |',
        '| Mapas target/previsão | [SymLogNorm](https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.SymLogNorm.html), Matplotlib; mesma máscara e escala dentro de cada par. |',
        '| CE90 omitido | [USGS (2024), acurácia posicional](https://pubs.usgs.gov/publication/ofr20241023/full). CE90 não mede erro de intensidade. |', '',
        'As fontes fundamentam os conceitos gerais. Os seis braços, os pares '
        'temporais, a grade, a subamostra, os produtos raster e a captura '
        'top-k são decisões documentadas deste experimento.', ''
    ]
    (run_dir / 'methodology_appendix.md').write_text('\n'.join(lines), encoding='utf-8')


def rebuild_report(run_dir: Path) -> None:
    from plot_enriched_linear_external import render_one
    from report_enriched_linear_all import run_report

    run_dir = run_dir.resolve()
    status_path = run_dir / 'run.json'
    status = json.loads(status_path.read_text(encoding='utf-8'))
    if status.get('kind') != 'enriched_xgboost' or not status.get('final_external_test_used'):
        raise ValueError('Expected an external-evaluated XGBoost run')
    items = []
    for horizon in (0, 1):
        paths = sorted((run_dir / f'horizon_{horizon}').glob('external_target_*_metrics.json'))
        if len(paths) != (3 if horizon == 0 else 1):
            raise ValueError('External metric files are incomplete')
        for path in paths:
            items.append(render_one(json.loads(path.read_text(encoding='utf-8')), run_dir))
    common._write_json(run_dir / 'figuras_teste_externo.json', items)
    _methodology_appendix(run_dir)
    status['status'] = 'completed'
    common._write_json(status_path, status)
    run_report(run_dir, run_dir)
    shutil.copy2(run_dir / 'TODOS_RESULTADOS.md', run_dir / 'RELATORIO.md')
    print(run_dir / 'RELATORIO.md')


def run(config_path: Path) -> None:
    cfg = load_config(config_path)
    cfg['output'].mkdir(parents=True, exist_ok=True)
    run_dir = cfg['output'] / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run_dir.mkdir(exist_ok=False)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    file_handler = logging.FileHandler(run_dir / 'training.log', encoding='utf-8')
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    status = dict(kind='enriched_xgboost', status='running',
                  started=datetime.now(timezone.utc).isoformat(), run=str(run_dir),
                  final_external_test_used=False)
    common._write_json(run_dir / 'run.json', status)
    try:
        total_audited = 2 * 4 * len(common.FEATURE_NAMES) + 3
        logging.info('Auditoria SHA-256 iniciada: %d rasters de entrada/target de desenvolvimento; o treinamento começa após a conferência.', total_audited)

        def audit_progress(count: int, relative: str) -> None:
            if count == 1 or count % 10 == 0 or count == total_audited:
                logging.info('Auditoria SHA-256: %d/%d arquivos conferidos (último: %s)', count, total_audited, relative)

        audit = common.audit_catalog(cfg, verify_hashes=True, on_verified=audit_progress)
        logging.info('Auditoria SHA-256 concluída; iniciando o experimento XGBoost.')
        common._write_json(run_dir / 'input_audit.json', audit)
        shutil.copy2(cfg['path'], run_dir / 'config.yaml')
        project = Path(__file__).parents[2]
        sources = ('run_enriched_xgboost.py', 'src/services/enriched_xgboost.py',
                   'src/services/enriched_hgb.py', 'src/services/enriched_linear.py',
                   'src/services/temporal_service.py',
                   'src/splitting/spatial_split.py', 'src/geospatial/raster_io.py',
                   'plot_enriched_linear_external.py', 'report_enriched_linear_all.py',
                   'generate_gradient_inputs.py', 'generate_graph_inputs.py')
        source_records = []
        for relative in sources:
            original = project / relative
            destination = run_dir / 'source_code' / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, destination)
            source_records.append(dict(path=relative, sha256=common._sha256(original)))
        common._write_json(run_dir / 'source_files.json', source_records)
        for relative in ('protocols/EXPERIMENTOS_ENRIQUECIDOS_2026.md',
                         'docs/ENRICHED_XGBOOST.md', 'docs/GRADIENT_INPUTS.md',
                         'docs/GRAPH_INPUTS.md'):
            original = project / relative
            destination = run_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, destination)
        versions = {name: importlib.metadata.version(name) for name in
                    ('numpy', 'scipy', 'scikit-learn', 'xgboost', 'rasterio',
                     'joblib', 'PyYAML', 'threadpoolctl', 'matplotlib')}
        common._write_json(run_dir / 'environment.json',
                           dict(python=sys.version, platform=platform.platform(),
                                packages=versions))
        profile = common._profile(cfg['reference'])
        zones = split_grid((profile['height'], profile['width']), cfg['raw']['split'])
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
                common.evaluate_external(cfg, horizons[horizon]['winner'], horizon,
                                         run_dir / f'horizon_{horizon}')
        status.update(status='external_complete',
                      external_finished=datetime.now(timezone.utc).isoformat(),
                      final_external_test_used=True,
                      winners={str(h): horizons[h]['winner']['arm'] for h in (0, 1)})
        common._write_json(run_dir / 'run.json', status)
        rebuild_report(run_dir)
        status['status'] = 'completed'
        status['finished'] = datetime.now(timezone.utc).isoformat()
        common._write_json(run_dir / 'run.json', status)
    except KeyboardInterrupt:
        status.update(status='interrupted', interrupted=datetime.now(timezone.utc).isoformat())
        common._write_json(run_dir / 'run.json', status)
        logging.warning('Execução interrompida pelo teclado; artefatos parciais preservados em %s', run_dir)
        raise SystemExit(130)
    except Exception as exc:
        status.update(status='failed', failed=datetime.now(timezone.utc).isoformat(),
                      error=repr(exc))
        common._write_json(run_dir / 'run.json', status)
        logging.exception('XGBoost run failed; partial artifacts retained at %s', run_dir)
        raise
    finally:
        logger.removeHandler(file_handler)
        logger.removeHandler(stream_handler)
        file_handler.close()
        stream_handler.close()
