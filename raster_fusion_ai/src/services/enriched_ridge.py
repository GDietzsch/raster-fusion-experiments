"""Ridge experiment on the fixed enriched raster protocol.

Hyperparameters and feature subsets use only the internal selection stage.
The external target is opened only after H0 and H1 winners are frozen.
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
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src.services import enriched_linear as common
from src.splitting.spatial_split import split_grid
from src.geospatial.raster_io import write_raster


BASE_ARMS = ('originais', 'gpa', 'grafos', 'todas')
ARM_BASE = {'originais': 'originais', 'originais_filtradas': 'originais',
            'gpa': 'gpa', 'grafos': 'grafos',
            'todas': 'todas', 'todas_filtradas': 'todas'}


def load_config(path: Path) -> dict:
    cfg = common.load_config(path)
    ridge = cfg['raw'].get('ridge', {})
    grid = ridge.get('lambda_per_sample')
    if not isinstance(grid, list) or len(grid) < 2:
        raise ValueError('Ridge requires at least two lambda_per_sample candidates')
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
           for value in grid) or list(grid) != sorted(set(grid)):
        raise ValueError('Ridge lambda_per_sample must be strictly increasing, finite and positive')
    if ridge.get('solver') != 'cholesky':
        raise ValueError('This Ridge protocol fixes solver=cholesky')
    if cfg['output'].resolve() == (Path(__file__).parents[2] / 'runs' / 'enriched_linear').resolve():
        raise ValueError('Ridge output must not overwrite linear runs')
    return cfg


def check_only(config_path: Path):
    cfg = load_config(config_path)
    audit = common.audit_catalog(cfg, verify_hashes=False)
    print(f"PRECHECK OK: {audit['input_and_development_target_files']} inputs/targets; "
          f"24 channels, H0/H1, {len(cfg['raw']['ridge']['lambda_per_sample'])} positive lambdas. "
          'External targets not opened.')


def fit_ridge(x: np.ndarray, y: np.ndarray, columns: tuple[int, ...],
              lambda_per_sample: float):
    """Keep penalty strength per observation fixed when sample size changes."""
    if lambda_per_sample <= 0 or len(y) < 1 or not columns:
        raise ValueError('Positive regularization, nonempty sample and columns required')
    alpha = float(lambda_per_sample * len(y))
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver='cholesky'))
    model.fit(x[:, columns].astype(np.float64, copy=False), y)
    return model


def tune_lambda(stage: dict, grid: list[float], directory: Path) -> dict[str, float]:
    x0, y0 = stage['ajuste'][:2]
    x1, y1 = stage['selecao_camadas'][:2]
    rows = []
    selected = {}
    for arm in BASE_ARMS:
        columns = common._arm_indices(arm)
        candidates = []
        for value in grid:
            lam = float(value)
            model = fit_ridge(x0, y0, columns, lam)
            predicted = common._predict(model, x1, columns)
            row = dict(base_arm=arm, lambda_per_sample=lam,
                       alpha_initial=float(model.named_steps['ridge'].alpha),
                       initial_fit_samples=len(y0), selection_samples=len(y1),
                       selection_rmse=common._rmse(y1, predicted),
                       selection_mae=common._mae(y1, predicted), chosen=False)
            rows.append(row)
            candidates.append(row)
        best = min(candidates, key=lambda row: (row['selection_rmse'],
                                                row['selection_mae'],
                                                row['lambda_per_sample']))
        best['chosen'] = True
        selected[arm] = best['lambda_per_sample']
        logging.info('Ridge %s: lambda/sample %.8g; selection RMSE %.6f',
                     arm, selected[arm], best['selection_rmse'])
    common._write_csv(directory / 'alpha_search.csv', rows)
    common._write_json(directory / 'alpha_choices.json', selected)
    return selected


def _training_arm(arm: str, stage: dict, high_threshold: float, directory: Path,
                  grid_width: int, seed: int, lambda_per_sample: float) -> dict:
    x0, y0 = stage['ajuste'][:2]
    x1, y1 = stage['selecao_camadas'][:2]
    xv, yv = stage['validacao_modelo'][:2]

    def fit(x, y, columns):
        return fit_ridge(x, y, columns, lambda_per_sample)

    chosen = common.PHENOMENA
    selection_rows = []
    if arm == 'originais_filtradas':
        chosen, selection_rows = common.select_phenomena(arm, x0, y0, x1, y1,
                                                          fit_model=fit)
        columns = common._arm_indices(arm, chosen)
    elif arm == 'todas_filtradas':
        columns, selection_rows = common.select_all_features(x0, y0, x1, y1,
                                                              fit_model=fit)
        chosen = tuple(phenomenon for phenomenon in common.PHENOMENA
                       if any(common.FEATURE_NAMES[i].startswith(f'{phenomenon}__')
                              for i in columns))
    else:
        columns = common._arm_indices(arm, chosen)

    initial = fit(x0, y0, columns)
    initial_metrics = common.metrics(y1, common._predict(initial, x1, columns),
                                     high_threshold)
    combined_x = np.concatenate((x0, x1))
    combined_y = np.concatenate((y0, y1))
    model = fit(combined_x, combined_y, columns)
    training_metrics = common.metrics(combined_y,
                                      common._predict(model, combined_x, columns),
                                      high_threshold)
    validation_prediction = common._predict(model, xv, columns)
    validation_metrics = common.metrics(yv, validation_prediction, high_threshold)
    positions = stage['validacao_modelo'][2]
    validation_uncertainty = common._bootstrap_block_errors(
        yv, validation_prediction, positions[:, 0], positions[:, 1],
        grid_width, seed=seed)
    validation_by_year = []
    for target_year in sorted(set(stage['validacao_modelo'][3])):
        mask = stage['validacao_modelo'][3] == target_year
        validation_by_year.append(dict(
            target_year=int(target_year),
            metrics=common.metrics(yv[mask], validation_prediction[mask], high_threshold)))
    folder = directory / arm
    folder.mkdir()
    names = [common.FEATURE_NAMES[i] for i in columns]
    joblib.dump(dict(model=model, columns=columns, feature_names=names,
                     phenomena=chosen, arm=arm,
                     lambda_per_sample=lambda_per_sample), folder / 'model.joblib')
    record = dict(arm=arm, phenomena=list(chosen), n_features=len(columns),
                  feature_names=names, lambda_per_sample=lambda_per_sample,
                  alpha_initial=float(initial.named_steps['ridge'].alpha),
                  alpha_refit=float(model.named_steps['ridge'].alpha),
                  coefficient_l2_norm=float(np.linalg.norm(model.named_steps['ridge'].coef_)),
                  initial_selection_metrics=initial_metrics,
                  training_metrics=training_metrics,
                  validation_metrics=validation_metrics,
                  validation_uncertainty=validation_uncertainty,
                  validation_by_year=validation_by_year,
                  selection_candidates=selection_rows,
                  training_samples=len(combined_y), validation_samples=len(yv))
    common._write_json(folder / 'metrics.json', record)
    return record


def train_horizon(cfg: dict, profile: dict, zones: np.ndarray,
                  horizon: int, directory: Path) -> dict:
    directory.mkdir()
    collection = common.collect_development(cfg, profile, zones, horizon,
                                             directory / 'samples.npz')
    common._write_json(directory / 'sampling.json', collection['counts'])
    stage = collection['stages']
    positive = np.concatenate((stage['ajuste'][1], stage['selecao_camadas'][1]))
    positive = positive[positive > 0]
    if not len(positive):
        raise ValueError(f'H{horizon} has no positive development targets')
    cutoff = float(np.quantile(positive,
                               cfg['raw']['evaluation']['high_target_positive_quantile']))
    training_mean = float(np.mean(np.concatenate((stage['ajuste'][1],
                                                  stage['selecao_camadas'][1]))))
    validation_target = stage['validacao_modelo'][1]
    references = dict(
        zero=common.metrics(validation_target, np.zeros_like(validation_target), subsets=False),
        training_mean=common.metrics(validation_target,
                                     np.full_like(validation_target, training_mean),
                                     subsets=False))
    common._write_json(directory / 'validation_references.json', references)
    chosen_lambda = tune_lambda(stage, cfg['raw']['ridge']['lambda_per_sample'], directory)
    records = []
    for arm in common.ARMS:
        logging.info('H%s: fitting Ridge %s', horizon, arm)
        records.append(_training_arm(arm, stage, cutoff, directory, profile['width'],
                                     cfg['raw']['sampling']['seed'] + horizon * 100,
                                     chosen_lambda[ARM_BASE[arm]]))
    winner = common._winner(records)
    flat = [dict(horizon=horizon, arm=row['arm'],
                 phenomena=';'.join(row['phenomena']), n_features=row['n_features'],
                 lambda_per_sample=row['lambda_per_sample'],
                 alpha_refit=row['alpha_refit'],
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
                    phenomena=winner['phenomena'],
                    feature_names=winner['feature_names'],
                    n_features=winner['n_features'],
                    lambda_per_sample=winner['lambda_per_sample'],
                    alpha_refit=winner['alpha_refit'],
                    validation_metrics=winner['validation_metrics'],
                    high_target_threshold=cutoff,
                    training_target_mean=training_mean,
                    model_path=str(directory / winner['arm'] / 'model.joblib'))
    common._write_json(directory / 'winner.json', selected)
    logging.info('H%s winner: %s; RMSE %.6f; lambda/sample %.8g',
                 horizon, winner['arm'], winner['validation_metrics']['rmse'],
                 winner['lambda_per_sample'])
    return dict(winner=selected, records=records, references=references)


def _methodology_appendix(run_dir: Path):
    lines = ['## Regularização Ridge e referências metodológicas', '',
             'O Ridge minimiza a soma dos quadrados dos resíduos mais `alpha × ||coeficientes||²`, '
             'após padronizar cada canal com média e desvio ajustados somente nos dados do respectivo ajuste. '
             'O intercepto é mantido. Como a forma do scikit-learn usa soma de erros, '
             '`alpha = lambda_por_amostra × n_amostras` em cada ajuste: a mesma lambda conserva a força '
             'relativa da penalização quando se unem ajuste inicial e seleção de camadas.', '',
             'Para cada horizonte, a grade de lambda foi testada separadamente nas quatro famílias de '
             'entradas **não filtradas** (originais, GPA, grafos e todas), usando somente o ajuste inicial '
             'para treinar e a etapa de seleção de camadas para medir RMSE. '
             '`originais_filtradas` herda a lambda de `originais`; `todas_filtradas` herda a de `todas`. '
             'A filtragem usa essa lambda fixa. O modelo de cada braço é então reajustado com ajuste inicial '
             'mais seleção de camadas, e uma validação separada escolhe o braço pelo menor RMSE. '
             'A grade é uma escolha experimental pré-fixada, não uma garantia de ótimo global.', '',
             '| Horizonte | Família base | Lambda por amostra escolhida | Alpha no ajuste inicial | RMSE na seleção |',
             '|---|---|---:|---:|---:|']
    for horizon in (0, 1):
        with (run_dir / f'horizon_{horizon}' / 'alpha_search.csv').open(
                encoding='utf-8-sig', newline='') as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            if row['chosen'] == 'True':
                lines.append(f"| H{horizon} | {row['base_arm']} | "
                             f"{float(row['lambda_per_sample']):.8g} | "
                             f"{float(row['alpha_initial']):.6g} | "
                             f"{float(row['selection_rmse']):.4f} |")
    lines += ['', '### Penalização efetiva no novo ajuste de cada braço', '',
              '| Horizonte | Braço | Canais | Lambda por amostra | Alpha no novo ajuste |',
              '|---|---|---:|---:|---:|']
    for horizon in (0, 1):
        for arm in common.ARMS:
            metric = json.loads((run_dir / f'horizon_{horizon}' / arm / 'metrics.json').read_text(encoding='utf-8'))
            lines.append(f"| H{horizon} | {arm} | {metric['n_features']} | "
                         f"{metric['lambda_per_sample']:.8g} | {metric['alpha_refit']:.6g} |")
    lines += ['', 'As tabelas `horizon_0/alpha_search.csv` e `horizon_1/alpha_search.csv` '
              'registram todas as lambdas candidatas, RMSE/MAE e escolhas; cada `metrics.json` '
              'guarda a lambda e o alpha efetivo do novo ajuste. A região externa não participou '
              'de nenhuma dessas decisões.', '',
              '### Conceitos e fontes', '',
              '| Conceito empregado | Fonte e aplicação neste experimento |', '|---|---|',
              '| Regressão Ridge e penalização L2 | [Hoerl e Kennard (1970), artigo original](https://doi.org/10.1080/00401706.1970.10488634); [documentação Ridge do scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html). |',
              '| Padronização sem vazamento e pipeline | [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html); [Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html), scikit-learn. |',
              '| Separação de ajuste, seleção e validação | [Guia de validação](https://scikit-learn.org/stable/modules/cross_validation.html), scikit-learn; [Valavi et al. (2019), blockCV](https://doi.org/10.1111/2041-210X.13107) e [Stock (2025), escolha de blocos espaciais](https://doi.org/10.3389/frsen.2025.1531097). |',
              '| Seleção reversa de atributos | [SequentialFeatureSelector](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SequentialFeatureSelector.html) documenta o conceito; a regra efetiva de grupos e RMSE é implementação própria em `src/services/enriched_linear.py` e está integralmente registrada em `selection_candidates.csv`. |',
              '| Incerteza por blocos espaciais | [Lahiri (2003), *Resampling Methods for Dependent Data*](https://link.springer.com/book/10.1007/978-1-4757-3803-2), cap. 12. Os intervalos aqui são exploratórios, com blocos de 96 km e uma única região externa. |',
              '| Erros RMSE, MAE e R² | [Métricas de regressão](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics), scikit-learn. Viés, P90 e captura são definidos no código arquivado desta execução. |',
              '| NMAD e robustez | [Median absolute deviation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.median_abs_deviation.html), SciPy; o fator 1,4826 e a aplicação aos resíduos estão no protocolo. |',
              '| Alinhamento e reamostragem de rasters | [WarpedVRT](https://rasterio.readthedocs.io/en/stable/api/rasterio.vrt.html) e [resampling](https://rasterio.readthedocs.io/en/stable/topics/resampling.html), Rasterio. |',
              '| Canais de gradiente chamados GPA neste projeto | [Diferenças centrais e espaçamento](https://numpy.org/doc/stable/reference/generated/numpy.gradient.html), NumPy. A implementação efetiva, que também invalida a borda, está em `docs/GRADIENT_INPUTS.md` e `generate_gradient_inputs.py`; não são usados G1/G2. |',
              '| Grafos espaciais e vizinhos próximos | [cKDTree](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html), SciPy. O grafo ponderado, raio de 50 km e rasterização são definições locais em `docs/GRAPH_INPUTS.md` e `generate_graph_inputs.py`. |',
              '| Escala cromática simétrica/logarítmica dos mapas | [SymLogNorm](https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.SymLogNorm.html), Matplotlib; mesmos limites dentro de cada par target/previsão. |',
              '| CE90 omitido | [USGS (2024), acurácia posicional](https://pubs.usgs.gov/publication/ofr20241023/full): CE90 descreve erro horizontal, não erro de intensidade. |',
              '| GPA, grafos, captura top-k e target zero | Definições operacionais do projeto em `docs/GRADIENT_INPUTS.md`, `docs/GRAPH_INPUTS.md`, `protocols/EXPERIMENTOS_ENRIQUECIDOS_2026.md` e no código arquivado. Essas definições não implicam causalidade ou ocorrência real de voos. |', '',
              'As fontes acima fundamentam os conceitos gerais. A grade de lambda, os pares temporais, '
              'a regra de seleção, os produtos raster e a captura top-k são decisões deste experimento, '
              'documentadas na configuração, no protocolo e no código congelado.', '']
    (run_dir / 'methodology_appendix.md').write_text('\n'.join(lines), encoding='utf-8')


def rebuild_report(run_dir: Path):
    from plot_enriched_linear_external import render_one
    from report_enriched_linear_all import run_report

    run_dir = run_dir.resolve()
    status_path = run_dir / 'run.json'
    status = json.loads(status_path.read_text(encoding='utf-8'))
    if status.get('kind') != 'enriched_ridge' or not status.get('final_external_test_used'):
        raise ValueError('Expected an external-evaluated enriched Ridge run')
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


def run(config_path: Path):
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
    status = dict(kind='enriched_ridge', status='running',
                  started=datetime.now(timezone.utc).isoformat(), run=str(run_dir),
                  final_external_test_used=False)
    common._write_json(run_dir / 'run.json', status)
    try:
        audit = common.audit_catalog(cfg, verify_hashes=True)
        common._write_json(run_dir / 'input_audit.json', audit)
        shutil.copy2(cfg['path'], run_dir / 'config.yaml')
        project = Path(__file__).parents[2]
        sources = ('run_enriched_ridge.py', 'src/services/enriched_ridge.py',
                   'src/services/enriched_linear.py',
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
                         'docs/ENRICHED_RIDGE.md', 'docs/GRADIENT_INPUTS.md',
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
        logging.exception('Ridge run failed; partial artifacts retained at %s', run_dir)
        raise
    finally:
        logger.removeHandler(file_handler)
        logger.removeHandler(stream_handler)
        file_handler.close()
        stream_handler.close()
