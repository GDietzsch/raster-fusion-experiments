"""Experiment 03: histogram gradient boosting on the fixed enriched raster protocol.

Only internal fit/feature-selection samples choose iterations and feature subsets.
The six final arms use all development samples; external targets are opened after
both winners have been frozen.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import importlib.metadata
import json
import logging
import math
import platform
import shutil
import sys

import joblib
import numpy as np
import yaml
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from src.services import enriched_linear as common
from src.splitting.spatial_split import split_grid
from src.geospatial.raster_io import write_raster


BASE_ARMS = ('originais', 'gpa', 'grafos', 'todas')
ARM_BASE = {'originais': 'originais', 'originais_filtradas': 'originais',
            'gpa': 'gpa', 'grafos': 'grafos', 'todas': 'todas',
            'todas_filtradas': 'todas'}


def load_config(path: Path) -> dict:
    cfg = common.load_config(path)
    hgb = cfg['raw'].get('hgb', {})
    if hgb.get('loss') != 'squared_error' or hgb.get('early_stopping') is not False:
        raise ValueError('This protocol fixes squared_error and early_stopping=false')
    grid = hgb.get('max_iter_candidates')
    if not isinstance(grid, list) or len(grid) < 2 or any(type(v) is not int or v < 1 for v in grid) or grid != sorted(set(grid)):
        raise ValueError('max_iter_candidates must be sorted distinct positive integers')
    for key in ('max_leaf_nodes', 'min_samples_leaf', 'max_bins',
                'selection_samples_per_stage_year'):
        if type(hgb.get(key)) is not int or hgb[key] < 1:
            raise ValueError(f'hgb.{key} must be a positive integer')
    if hgb['max_bins'] > 255 or hgb['max_leaf_nodes'] < 2:
        raise ValueError('max_bins must be <=255 and max_leaf_nodes >=2')
    if type(hgb.get('random_state')) is not int:
        raise ValueError('hgb.random_state must be an integer')
    for key in ('learning_rate', 'l2_regularization'):
        value = hgb.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or (key == 'learning_rate' and value == 0):
            raise ValueError(f'hgb.{key} must be finite and positive (L2 may be zero)')
    if cfg['output'].resolve() in ((Path(__file__).parents[2] / 'runs' / item).resolve()
                                   for item in ('enriched_linear', 'enriched_ridge')):
        raise ValueError('HGB output must not overwrite earlier experiments')
    return cfg


def check_only(config_path: Path) -> None:
    cfg = load_config(config_path)
    audit = common.audit_catalog(cfg, verify_hashes=False)
    print(f"PRECHECK OK: {audit['input_and_development_target_files']} inputs/targets; "
          f"24 channels, H0/H1, {len(cfg['raw']['hgb']['max_iter_candidates'])} iteration candidates. "
          'External targets not opened.')


def fit_hgb(x: np.ndarray, y: np.ndarray, columns: tuple[int, ...],
            max_iter: int, settings: dict):
    if not columns or not len(y):
        raise ValueError('Nonempty features and fitting sample required')
    model = HistGradientBoostingRegressor(
        loss='squared_error', learning_rate=float(settings['learning_rate']),
        max_iter=int(max_iter), max_leaf_nodes=int(settings['max_leaf_nodes']),
        min_samples_leaf=int(settings['min_samples_leaf']),
        l2_regularization=float(settings['l2_regularization']),
        max_bins=int(settings['max_bins']), early_stopping=False,
        categorical_features=None, random_state=int(settings['random_state']))
    model.fit(x[:, columns], y)
    return model


def selection_subsample(stage: dict, cap_per_year: int, seed: int,
                        directory: Path) -> dict:
    """One fixed stratified subset, shared by every tuning and feature candidate."""
    selected = {}
    metadata = []
    indexes = {}
    for stage_name in ('ajuste', 'selecao_camadas'):
        x, y, positions, years = stage[stage_name]
        rng = np.random.default_rng(seed + (0 if stage_name == 'ajuste' else 1))
        groups = []
        for year in sorted(np.unique(years)):
            candidates = np.flatnonzero(years == year)
            take = min(cap_per_year, len(candidates))
            chosen = np.sort(rng.choice(candidates, size=take, replace=False))
            groups.append(chosen)
            metadata.append(dict(stage=stage_name, target_year=int(year),
                                 full_samples=len(candidates), selection_samples=take))
        index = np.sort(np.concatenate(groups))
        indexes[stage_name] = index
        selected[stage_name] = (x[index], y[index], positions[index], years[index])
    np.savez_compressed(directory / 'selection_subsample_indices.npz', **indexes)
    common._write_json(directory / 'selection_subsample.json',
                       dict(cap_per_stage_year=cap_per_year, seed=seed, counts=metadata,
                            index_sha256={name: hashlib.sha256(index.tobytes()).hexdigest()
                                          for name, index in indexes.items()},
                            use='iteration and feature selection only; final refit uses full samples'))
    return selected


def tune_iterations(selection: dict, grid: list[int], settings: dict,
                    directory: Path) -> dict[str, int]:
    x0, y0 = selection['ajuste'][:2]
    x1, y1 = selection['selecao_camadas'][:2]
    rows, chosen = [], {}
    for arm in BASE_ARMS:
        columns = common._arm_indices(arm)
        candidates = []
        for count in grid:
            model = fit_hgb(x0, y0, columns, count, settings)
            predicted = common._predict(model, x1, columns)
            row = dict(base_arm=arm, max_iter=count, actual_iterations=model.n_iter_,
                       fit_samples=len(y0), selection_samples=len(y1),
                       selection_rmse=common._rmse(y1, predicted),
                       selection_mae=common._mae(y1, predicted), chosen=False)
            rows.append(row)
            candidates.append(row)
        best = min(candidates, key=lambda row: (row['selection_rmse'],
                                                row['selection_mae'], row['max_iter']))
        best['chosen'] = True
        chosen[arm] = best['max_iter']
        logging.info('HGB %s: %s iterations; selection RMSE %.6f',
                     arm, best['max_iter'], best['selection_rmse'])
    common._write_csv(directory / 'max_iter_search.csv', rows)
    common._write_json(directory / 'max_iter_choices.json', chosen)
    return chosen


def _training_arm(arm: str, full: dict, selection: dict, cutoff: float,
                  directory: Path, width: int, seed: int,
                  max_iter: int, settings: dict) -> dict:
    x0, y0 = full['ajuste'][:2]
    x1, y1 = full['selecao_camadas'][:2]
    xs0, ys0 = selection['ajuste'][:2]
    xs1, ys1 = selection['selecao_camadas'][:2]
    xv, yv, valid_positions, valid_years = full['validacao_modelo']

    def fit(x, y, columns):
        return fit_hgb(x, y, columns, max_iter, settings)

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
    training_metrics = common.metrics(combined_y, common._predict(model, combined_x, columns), cutoff)
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
    joblib.dump(dict(model=model, columns=columns, feature_names=feature_names,
                     phenomena=phenomena, arm=arm, max_iter=max_iter), folder / 'model.joblib')
    record = dict(arm=arm, phenomena=list(phenomena), n_features=len(columns),
                  feature_names=feature_names, max_iter=max_iter,
                  actual_iterations=model.n_iter_,
                  estimator_parameters=model.get_params(),
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
    settings = cfg['raw']['hgb']
    selection = selection_subsample(full,
                                    settings['selection_samples_per_stage_year'],
                                    settings['random_state'] + horizon * 1000, directory)
    iterations = tune_iterations(selection, settings['max_iter_candidates'],
                                 settings, directory)
    records = []
    for arm in common.ARMS:
        logging.info('H%s: fitting HistGradientBoosting %s', horizon, arm)
        records.append(_training_arm(arm, full, selection, cutoff, directory,
                                     profile['width'],
                                     cfg['raw']['sampling']['seed'] + horizon * 100,
                                     iterations[ARM_BASE[arm]], settings))
    winner = common._winner(records)
    flat = [dict(horizon=horizon, arm=row['arm'],
                 phenomena=';'.join(row['phenomena']), n_features=row['n_features'],
                 max_iter=row['max_iter'], actual_iterations=row['actual_iterations'],
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
                    n_features=winner['n_features'], max_iter=winner['max_iter'],
                    validation_metrics=winner['validation_metrics'],
                    high_target_threshold=cutoff, training_target_mean=training_mean,
                    model_path=str(directory / winner['arm'] / 'model.joblib'))
    common._write_json(directory / 'winner.json', selected)
    logging.info('H%s winner: %s; RMSE %.6f; max_iter %s',
                 horizon, winner['arm'], winner['validation_metrics']['rmse'],
                 winner['max_iter'])
    return dict(winner=selected, records=records, references=references)


def _methodology_appendix(run_dir: Path) -> None:
    settings = yaml.safe_load((run_dir / 'config.yaml').read_text(encoding='utf-8'))['hgb']
    cap = settings['selection_samples_per_stage_year']
    lines = [
        '## HistGradientBoosting e referências metodológicas', '',
        'O estimador é `HistGradientBoostingRegressor` do scikit-learn com perda '
        '`squared_error`, coerente com a escolha de RMSE. Cada árvore divide valores '
        'numéricos agrupados em histogramas. Os 24 canais são tratados como numéricos; '
        'não há padronização. `early_stopping=False` impede que o estimador crie uma '
        'divisão aleatória interna, preservando as etapas espaciais/temporais do protocolo. '
        'A saída conserva as unidades do target e não é truncada em zero.', '',
        'Por custo computacional, busca de iterações e filtragem usam uma subamostra '
        f'aleatória fixa, sem reposição, de até {cap:,} pixels por ano em cada etapa interna. '
        'A mesma subamostra é compartilhada por todos os candidatos de um horizonte. '
        'Os índices estão em `selection_subsample_indices.npz`, com contagens e hashes em '
        '`selection_subsample.json`; as amostras completas originais ficam em `samples.npz`. '
        'Os seis modelos finais são reajustados nas amostras completas de ajuste e seleção, '
        'e comparados na validação completa. Assim, RMSE de seleção nesta rodada não tem '
        'a mesma base amostral dos experimentos 01/02; RMSE de validação tem.', '',
        'A grade fixa de `max_iter` é testada nas quatro famílias não filtradas '
        '(originais, GPA, grafos, todas). Cada candidato ajusta somente na subamostra '
        'da etapa inicial e é medido na subamostra da etapa de seleção. Os braços filtrados '
        'herdam o `max_iter` da família correspondente antes da busca de camadas. '
        'Empates usam MAE e depois menos iterações. Não houve busca com o target externo.', '',
        '| Horizonte | Família base | Iterações escolhidas | RMSE na subamostra de seleção |',
        '|---|---|---:|---:|',
    ]
    import csv
    for horizon in (0, 1):
        with (run_dir / f'horizon_{horizon}' / 'max_iter_search.csv').open(encoding='utf-8-sig', newline='') as handle:
            for row in csv.DictReader(handle):
                if row['chosen'] == 'True':
                    lines.append(f"| H{horizon} | {row['base_arm']} | {row['max_iter']} | "
                                 f"{float(row['selection_rmse']):.4f} |")
    lines += ['', f"Parâmetros restantes fixos no YAML: taxa de aprendizado {settings['learning_rate']}, "
              f"{settings['max_leaf_nodes']} folhas máximas, "
              f"{settings['min_samples_leaf']} amostras mínimas por folha, "
              f"penalização L2 de {settings['l2_regularization']}, "
              f"{settings['max_bins']} bins e semente {settings['random_state']}. "
              'São escolhas de desenho experimental; '
              'não representam um ótimo conhecido. Todas as candidatas e os parâmetros '
              'efetivos estão em `max_iter_search.csv` e nos `metrics.json`.', '',
              '### Conceitos e fontes', '',
              '| Conceito empregado | Fonte e aplicação neste experimento |', '|---|---|',
              '| HistGradientBoostingRegressor, perda, hiperparâmetros e early stopping | [Documentação oficial do estimador](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html) e [guia de ensemble](https://scikit-learn.org/stable/modules/ensemble.html#histogram-based-gradient-boosting), scikit-learn. |',
              '| Gradient boosting por árvores | [Friedman (2001), Greedy Function Approximation: A Gradient Boosting Machine](https://doi.org/10.1214/aos/1013203451). A implementação exata é a do scikit-learn. |',
              '| Separação espacial, ajuste e validação | [Valavi et al. (2019), blockCV](https://doi.org/10.1111/2041-210X.13107); [Stock (2025)](https://doi.org/10.3389/frsen.2025.1531097); [guia de validação do scikit-learn](https://scikit-learn.org/stable/modules/cross_validation.html). |',
              '| Filtragem reversa de grupos | [SequentialFeatureSelector, documentação conceitual](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SequentialFeatureSelector.html). A regra efetiva é implementação própria em `src/services/enriched_linear.py` e fica registrada em `selection_candidates.csv`. |',
              '| Métricas RMSE, MAE e R² | [Guia de métricas de regressão](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics), scikit-learn. Viés, captura e P90 seguem as definições do protocolo local. |',
              '| NMAD dos resíduos | [Median absolute deviation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.median_abs_deviation.html), SciPy; fator 1,4826 e aplicação aos resíduos definidos no protocolo. |',
              '| Intervalos exploratórios por blocos | [Lahiri (2003), Resampling Methods for Dependent Data](https://link.springer.com/book/10.1007/978-1-4757-3803-2), cap. 12. Uma região externa não se torna várias regiões independentes. |',
              '| Alinhamento de rasters | [WarpedVRT](https://rasterio.readthedocs.io/en/stable/api/rasterio.vrt.html) e [reamostragem](https://rasterio.readthedocs.io/en/stable/topics/resampling.html), Rasterio. |',
              '| Gradientes usados como GPA neste projeto | [numpy.gradient](https://numpy.org/doc/stable/reference/generated/numpy.gradient.html) e `docs/GRADIENT_INPUTS.md`; não são G1/G2. |',
              '| Grafos e vizinhos espaciais | [SciPy cKDTree](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html) e `docs/GRAPH_INPUTS.md`; os pesos e o raio são definições locais. |',
              '| Escala dos mapas target/previsão | [SymLogNorm](https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.SymLogNorm.html), Matplotlib; mesma escala dentro de cada par. |',
              '| CE90 omitido | [USGS (2024), acurácia posicional](https://pubs.usgs.gov/publication/ofr20241023/full). CE90 não mede erro de intensidade. |', '',
              'Estas fontes fundamentam os conceitos gerais. Os seis braços, grade de '
              'iterações, tamanho da subamostra, atributos derivados e regra de captura '
              'são decisões documentadas deste experimento.', '']
    (run_dir / 'methodology_appendix.md').write_text('\n'.join(lines), encoding='utf-8')


def rebuild_report(run_dir: Path) -> None:
    from plot_enriched_linear_external import render_one
    from report_enriched_linear_all import run_report

    run_dir = run_dir.resolve()
    status_path = run_dir / 'run.json'
    status = json.loads(status_path.read_text(encoding='utf-8'))
    if status.get('kind') != 'enriched_hgb' or not status.get('final_external_test_used'):
        raise ValueError('Expected an external-evaluated HistGradientBoosting run')
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
    status = dict(kind='enriched_hgb', status='running',
                  started=datetime.now(timezone.utc).isoformat(), run=str(run_dir),
                  final_external_test_used=False)
    common._write_json(run_dir / 'run.json', status)
    try:
        audit = common.audit_catalog(cfg, verify_hashes=True)
        common._write_json(run_dir / 'input_audit.json', audit)
        shutil.copy2(cfg['path'], run_dir / 'config.yaml')
        project = Path(__file__).parents[2]
        sources = ('run_enriched_hgb.py', 'src/services/enriched_hgb.py',
                   'src/services/enriched_linear.py', 'src/services/temporal_service.py',
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
                         'docs/ENRICHED_HGB.md', 'docs/GRADIENT_INPUTS.md',
                         'docs/GRAPH_INPUTS.md'):
            original = project / relative
            destination = run_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, destination)
        versions = {name: importlib.metadata.version(name) for name in
                    ('numpy', 'scipy', 'scikit-learn', 'rasterio', 'joblib',
                     'PyYAML', 'threadpoolctl', 'matplotlib')}
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
    except BaseException as exc:
        status.update(status='failed', failed=datetime.now(timezone.utc).isoformat(),
                      error=repr(exc))
        common._write_json(run_dir / 'run.json', status)
        logging.exception('HGB run failed; partial artifacts retained at %s', run_dir)
        raise
    finally:
        logger.removeHandler(file_handler)
        logger.removeHandler(stream_handler)
        file_handler.close()
        stream_handler.close()
