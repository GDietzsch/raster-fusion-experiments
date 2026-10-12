"""Experiment 07: budgeted MLP search on enriched raster features."""
from __future__ import annotations
from src.utils.source_paths import project_root, source_path

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
import time

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import psutil
import yaml
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src.services import enriched_linear as common
from src.services.enriched_hgb import selection_subsample
from src.splitting.spatial_split import split_grid
from src.geospatial.raster_io import write_raster


BASE_ARMS = ('originais', 'gpa', 'grafos', 'todas')
ARM_BASE = {'originais': 'originais', 'originais_filtradas': 'originais',
            'gpa': 'gpa', 'grafos': 'grafos', 'todas': 'todas',
            'todas_filtradas': 'todas'}
PARAMETERS = ('hidden_layer_sizes', 'alpha', 'learning_rate_init')


def load_config(path: Path) -> dict:
    cfg = common.load_config(path)
    settings = cfg['raw'].get('mlp', {})
    fixed = dict(loss='squared_error', activation='relu', solver='adam',
                 early_stopping=False)
    if any(settings.get(key) != value for key, value in fixed.items()):
        raise ValueError('MLP requires squared_error, relu, adam and early_stopping=false')
    for key in ('random_state', 'selection_seed', 'selection_samples_per_stage_year',
                'max_epochs', 'patience', 'min_epochs', 'batch_size'):
        if type(settings.get(key)) is not int or settings[key] < 1:
            raise ValueError(f'mlp.{key} must be a positive integer')
    if settings['min_epochs'] > settings['max_epochs']:
        raise ValueError('min_epochs must not exceed max_epochs')
    grid = settings.get('candidates')
    if not isinstance(grid, list) or not grid:
        raise ValueError('mlp.candidates must be a nonempty list')
    seen = set()
    for candidate in grid:
        if not isinstance(candidate, dict) or set(candidate) != set(PARAMETERS):
            raise ValueError('Each MLP candidate needs hidden_layer_sizes, alpha and learning_rate_init')
        hidden = candidate['hidden_layer_sizes']
        if not isinstance(hidden, list) or not hidden or any(type(v) is not int or v < 1 for v in hidden):
            raise ValueError('hidden_layer_sizes must be positive integers')
        for key in ('alpha', 'learning_rate_init'):
            value = candidate[key]
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or (key == 'learning_rate_init' and value == 0):
                raise ValueError(f'Invalid MLP {key}')
        signature = (tuple(hidden), candidate['alpha'], candidate['learning_rate_init'])
        if signature in seen:
            raise ValueError('Duplicate MLP candidate')
        seen.add(signature)
    previous = ('enriched_linear', 'enriched_ridge', 'enriched_hgb',
                'enriched_xgboost', 'enriched_random_forest', 'enriched_extra_trees')
    if cfg['output'].resolve() in ((project_root() / 'runs' / item).resolve()
                                   for item in previous):
        raise ValueError('MLP output must not overwrite previous experiments')
    return cfg


def candidates(settings: dict) -> list[dict]:
    return settings['candidates']


def check_only(config_path: Path) -> None:
    cfg = load_config(config_path)
    audit = common.audit_catalog(cfg, verify_hashes=False)
    settings = cfg['raw']['mlp']
    print(f"PRECHECK OK: {audit['input_and_development_target_files']} inputs/targets; "
          f"24 channels, H0/H1, {len(candidates(settings))} MLP candidates per base arm, "
          f"up to {settings['max_epochs']} epochs; "
          'external targets not opened.')


class ScaledMLP:
    """Persist the development-only scalers with a regressor returning target units."""

    def __init__(self, scaler: StandardScaler, target_mean: float,
                 target_scale: float, estimator: MLPRegressor):
        self.scaler = scaler
        self.target_mean = target_mean
        self.target_scale = target_scale
        self.estimator = estimator

    def predict(self, x: np.ndarray) -> np.ndarray:
        transformed = self.scaler.transform(x)
        return self.estimator.predict(transformed) * self.target_scale + self.target_mean


def fit_mlp(x: np.ndarray, y: np.ndarray, columns: tuple[int, ...],
            params: dict, epochs: int, settings: dict,
            monitor: tuple[np.ndarray, np.ndarray] | None = None) -> tuple[ScaledMLP, list[dict]]:
    if not columns or not len(y):
        raise ValueError('Nonempty feature set and fitting sample required')
    scaler = StandardScaler().fit(x[:, columns])
    scaled_x = scaler.transform(x[:, columns])
    target_mean = float(np.mean(y))
    target_scale = float(np.std(y)) or 1.0
    scaled_y = (np.asarray(y, dtype=np.float64) - target_mean) / target_scale
    estimator = MLPRegressor(
        loss='squared_error', hidden_layer_sizes=tuple(params['hidden_layer_sizes']),
        activation='relu', solver='adam', alpha=float(params['alpha']),
        batch_size=int(settings['batch_size']),
        learning_rate_init=float(params['learning_rate_init']),
        max_iter=1, shuffle=False, early_stopping=False,
        random_state=int(settings['random_state']),
        n_iter_no_change=int(settings['max_epochs']) + 1)
    model = ScaledMLP(scaler, target_mean, target_scale, estimator)
    score_x = score_y = None
    if monitor is not None:
        score_x, score_y = monitor
        score_x = scaler.transform(score_x[:, columns])
    rng = np.random.default_rng(settings['random_state'])
    history = []
    best_rmse, stale = float('inf'), 0
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(y))
        estimator.partial_fit(scaled_x[order], scaled_y[order])
        if not np.isfinite(estimator.loss_):
            raise ValueError('MLP produced non-finite training loss')
        row = dict(epoch=epoch, train_loss_scaled=float(estimator.loss_),
                   elapsed_seconds=round(time.perf_counter() - started, 3))
        if monitor is not None:
            predicted = estimator.predict(score_x) * target_scale + target_mean
            if not np.isfinite(predicted).all():
                raise ValueError('MLP produced non-finite selection prediction')
            row.update(selection_rmse=common._rmse(score_y, predicted),
                       selection_mae=common._mae(score_y, predicted))
            if epoch >= settings['min_epochs']:
                if row['selection_rmse'] < best_rmse:
                    best_rmse, stale = row['selection_rmse'], 0
                else:
                    stale += 1
        history.append(row)
        if monitor is not None and stale >= settings['patience']:
            break
    return model, history


def _write_checkpoint_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def search_hyperparameters(selection: dict, settings: dict,
                           directory: Path) -> dict[str, dict]:
    """Choose architecture and epoch from internal selection only."""
    trial_path = directory / 'hyperparameter_trials.json'
    grid = candidates(settings)
    common._write_json(directory / 'hyperparameter_candidates.json', grid)
    rows = json.loads(trial_path.read_text(encoding='utf-8')) if trial_path.is_file() else []
    known = {}
    for row in rows:
        key = (row['base_arm'], row['candidate_id'])
        if key in known or row['params'] != grid[row['candidate_id']]:
            raise ValueError('Inconsistent hyperparameter trial checkpoint')
        known[key] = row
    x_fit, y_fit = selection['ajuste'][:2]
    x_score, y_score = selection['selecao_camadas'][:2]

    def evaluate(arm: str, candidate_id: int) -> dict:
        key = (arm, candidate_id)
        if key in known:
            return known[key]
        columns = common._arm_indices(arm)
        params = grid[candidate_id]
        started = time.perf_counter()
        model, history = fit_mlp(x_fit, y_fit, columns, params,
                                 settings['max_epochs'], settings,
                                 monitor=(x_score, y_score))
        rss_after_fit_mb = psutil.Process().memory_info().rss / (1024 ** 2)
        eligible = [item for item in history if item['epoch'] >= settings['min_epochs']]
        best = min(eligible, key=lambda item: (item['selection_rmse'],
                                               item['selection_mae'], item['epoch']))
        n_parameters = sum(layer.size for layer in model.estimator.coefs_) + sum(
            layer.size for layer in model.estimator.intercepts_)
        row = dict(base_arm=arm, candidate_id=candidate_id,
                   params=params, chosen_epochs=best['epoch'],
                   epochs_completed=len(history), n_parameters=n_parameters,
                   fit_samples=len(y_fit), selection_samples=len(y_score),
                   selection_rmse=best['selection_rmse'],
                   selection_mae=best['selection_mae'],
                   fit_and_score_seconds=round(time.perf_counter()-started, 3),
                   rss_after_fit_mb=round(rss_after_fit_mb, 1),
                   history=history)
        rows.append(row)
        known[key] = row
        _write_checkpoint_json(trial_path, rows)
        logging.info('MLP %s candidate %s/%s: epoch %s RMSE %.6f (%.1f s)',
                     arm, candidate_id+1, len(grid), best['epoch'], row['selection_rmse'],
                     row['fit_and_score_seconds'])
        return row

    def rank(row: dict):
        return (row['selection_rmse'], row['selection_mae'],
                row['n_parameters'], row['chosen_epochs'],
                row['candidate_id'])

    chosen = {}
    for arm in BASE_ARMS:
        tested = [evaluate(arm, index) for index in range(len(grid))]
        best = min(tested, key=rank)
        chosen[arm] = dict(params=grid[best['candidate_id']],
                           epochs=best['chosen_epochs'],
                           candidate_id=best['candidate_id'])
        logging.info('MLP %s chosen candidate %s, epoch %s: RMSE %.6f',
                     arm, best['candidate_id']+1, best['chosen_epochs'],
                     best['selection_rmse'])
    common._write_json(directory / 'hyperparameter_choices.json', chosen)
    common._write_csv(directory / 'hyperparameter_trials.csv', [
        dict(base_arm=row['base_arm'], candidate_id=row['candidate_id'],
             hidden_layer_sizes=';'.join(map(str, row['params']['hidden_layer_sizes'])),
             alpha=row['params']['alpha'],
             learning_rate_init=row['params']['learning_rate_init'],
             chosen_epochs=row['chosen_epochs'], epochs_completed=row['epochs_completed'],
             n_parameters=row['n_parameters'], fit_samples=row['fit_samples'],
             selection_samples=row['selection_samples'],
             selection_rmse=row['selection_rmse'], selection_mae=row['selection_mae'],
             fit_and_score_seconds=row['fit_and_score_seconds'],
             rss_after_fit_mb=row['rss_after_fit_mb'])
        for row in rows])
    common._write_csv(directory / 'hyperparameter_epoch_history.csv', [
        dict(base_arm=row['base_arm'], candidate_id=row['candidate_id'], **point)
        for row in rows for point in row['history']])
    return chosen


def _training_arm(arm: str, full: dict, selection: dict, cutoff: float,
                  directory: Path, width: int, seed: int, choice: dict,
                  settings: dict, threads: int) -> dict:
    folder = directory / arm
    record_path = folder / 'metrics.json'
    if record_path.is_file() and (folder / 'model.joblib').is_file():
        record = json.loads(record_path.read_text(encoding='utf-8'))
        if (record['arm'] != arm or record['hyperparameters'] != choice['params'] or
            record['selected_epoch'] != choice['epochs']):
            raise ValueError(f'Existing MLP arm differs from current configuration: {folder}')
        logging.info('MLP %s: reused completed arm checkpoint', arm)
        return record
    folder.mkdir(exist_ok=True)
    x0, y0 = full['ajuste'][:2]
    x1, y1 = full['selecao_camadas'][:2]
    xs0, ys0 = selection['ajuste'][:2]
    xs1, ys1 = selection['selecao_camadas'][:2]
    xv, yv, valid_positions, valid_years = full['validacao_modelo']

    filter_calls = 0

    def fit_filter(x, y, columns):
        nonlocal filter_calls
        filter_calls += 1
        if filter_calls == 1 or filter_calls % 5 == 0:
            logging.info('MLP %s: internal selection fit %s, %s channels, %s epochs',
                         arm, filter_calls, len(columns), choice['epochs'])
        return fit_mlp(x, y, columns, choice['params'],
                       choice['epochs'], settings)[0]

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
    model, history = fit_mlp(combined_x, combined_y, columns,
                             choice['params'], choice['epochs'], settings)
    common._write_csv(folder / 'training_history.csv', history)
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
                     phenomena=phenomena, arm=arm,
                     hyperparameters=choice['params'],
                     selected_epoch=choice['epochs']), folder / 'model.joblib')
    record = dict(arm=arm, phenomena=list(phenomena), n_features=len(columns),
                  feature_names=feature_names, hyperparameters=choice['params'],
                  selected_epoch=choice['epochs'],
                  search_candidate_id=choice['candidate_id'],
                  actual_epochs=len(history),
                  cpu_threads=threads, model_sha256=common._sha256(folder / 'model.joblib'),
                  rss_after_fit_mb=round(rss_after_fit_mb, 1),
                  target_scaler_mean=model.target_mean,
                  target_scaler_scale=model.target_scale,
                  input_scaler_mean=model.scaler.mean_.tolist(),
                  input_scaler_scale=model.scaler.scale_.tolist(),
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
    logging.info('MLP %s: validation RMSE %.6f with %s channels',
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
    settings = cfg['raw']['mlp']
    selection = selection_subsample(full,
                                    settings['selection_samples_per_stage_year'],
                                    settings['selection_seed'] + horizon * 1000, directory)
    chosen = search_hyperparameters(selection, settings, directory)
    records = []
    for arm in common.ARMS:
        logging.info('H%s: fitting MLP %s', horizon, arm)
        records.append(_training_arm(arm, full, selection, cutoff, directory,
                                     profile['width'],
                                     cfg['raw']['sampling']['seed'] + horizon * 100,
                                     chosen[ARM_BASE[arm]], settings,
                                     cfg['raw']['runtime']['cpu_threads']))
    winner = common._winner(records)
    flat = [dict(horizon=horizon, arm=row['arm'],
                 phenomena=';'.join(row['phenomena']),
                 n_features=row['n_features'],
                 selected_epoch=row['selected_epoch'],
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
                    n_features=winner['n_features'], selected_epoch=winner['selected_epoch'],
                    hyperparameters=winner['hyperparameters'],
                    validation_metrics=winner['validation_metrics'],
                    high_target_threshold=cutoff, training_target_mean=training_mean,
                    model_path=str(directory / winner['arm'] / 'model.joblib'))
    common._write_json(directory / 'winner.json', selected)
    logging.info('H%s MLP winner: %s; validation RMSE %.6f',
                 horizon, winner['arm'], winner['validation_metrics']['rmse'])
    return dict(winner=selected, records=records, references=references)


def _training_figures(run_dir: Path) -> None:
    destination = run_dir / 'graficos_treinamento'
    destination.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for horizon, ax in enumerate(axes):
        directory = run_dir / f'horizon_{horizon}'
        trials = json.loads((directory / 'hyperparameter_trials.json').read_text(encoding='utf-8'))
        choices = json.loads((directory / 'hyperparameter_choices.json').read_text(encoding='utf-8'))
        for arm in BASE_ARMS:
            selected = next(row for row in trials if row['base_arm'] == arm and
                            row['candidate_id'] == choices[arm]['candidate_id'])
            ax.plot([point['epoch'] for point in selected['history']],
                    [point['selection_rmse'] for point in selected['history']],
                    label=arm)
            ax.scatter([selected['chosen_epochs']], [selected['selection_rmse']], s=40)
        ax.set(title=f'H{horizon}', xlabel='Época', ylabel='RMSE na seleção interna')
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.suptitle('MLP · histórico real das configurações selecionadas')
    fig.savefig(destination / 'rmse_selecao_por_epoca.png', dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for horizon, ax in enumerate(axes):
        for arm in common.ARMS:
            with (run_dir / f'horizon_{horizon}' / arm /
                  'training_history.csv').open(encoding='utf-8-sig', newline='') as source:
                history = list(csv.DictReader(source))
            ax.plot([int(row['epoch']) for row in history],
                    [float(row['train_loss_scaled']) for row in history], label=arm)
        ax.set(title=f'H{horizon}', xlabel='Época', ylabel='Perda de treino (target padronizado)')
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.suptitle('MLP · perda real por época dos 12 ajustes finais')
    fig.savefig(destination / 'perda_treino_por_epoca.png', dpi=300)
    plt.close(fig)


def _methodology_appendix(run_dir: Path) -> None:
    settings = yaml.safe_load((run_dir / 'config.yaml').read_text(encoding='utf-8'))['mlp']
    lines = [
        '## MLP: arquitetura, épocas e histórico real', '',
        'O experimento 07 usa `MLPRegressor`, ReLU, Adam e perda quadrática. '
        'Entradas e target são padronizados apenas com as respectivas amostras de ajuste; '
        'as previsões retornam às unidades originais. A saída é linear, sem truncamento. '
        f"Semente {settings['random_state']}; batch {settings['batch_size']}; "
        f"até {settings['max_epochs']} épocas; paciência {settings['patience']} após a época "
        f"{settings['min_epochs']}. A busca usa até "
        f"{settings['selection_samples_per_stage_year']:,} pixels por ano/etapa. "
        'A época e a arquitetura são escolhidas pelo RMSE na seleção interna. '
        'Os seis braços finais são reajustados com todas as amostras de desenvolvimento '
        'e avaliados uma vez na validação. Não se usa holdout aleatório interno. '
        'O teste externo não determina épocas, camadas ou parâmetros.', '',
        '| Horizonte | Família base | Camadas ocultas | Alpha | Taxa Adam | Época escolhida |',
        '|---|---|---|---:|---:|---:|',
    ]
    for horizon in (0, 1):
        directory = run_dir / f'horizon_{horizon}'
        chosen = json.loads((directory / 'hyperparameter_choices.json').read_text(encoding='utf-8'))
        for arm in BASE_ARMS:
            choice = chosen[arm]
            params = choice['params']
            lines.append(f"| H{horizon} | {arm} | {','.join(map(str, params['hidden_layer_sizes']))} | "
                         f"{params['alpha']} | {params['learning_rate_init']} | {choice['epochs']} |")
    lines += ['', '### Doze ajustes finais', '',
              'Cada braço abaixo foi reajustado após a seleção. O histórico real por época '
              'e os parâmetros do padronizador estão arquivados junto ao modelo.', '',
              '| Horizonte | Braço | Canais | Ocultas | Alpha | Taxa | Épocas | Amostras de ajuste | Amostras de validação | RMSE treino | RMSE validação |',
              '|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for horizon in (0, 1):
        directory = run_dir / f'horizon_{horizon}'
        for arm in common.ARMS:
            record = json.loads((directory / arm / 'metrics.json').read_text(encoding='utf-8'))
            params = record['hyperparameters']
            lines.append(f"| H{horizon} | {arm} | {record['n_features']} | "
                         f"{','.join(map(str, params['hidden_layer_sizes']))} | "
                         f"{params['alpha']} | {params['learning_rate_init']} | "
                         f"{record['selected_epoch']} | {record['training_samples']:,} | "
                         f"{record['validation_samples']:,} | "
                         f"{record['training_metrics']['rmse']:.4f} | "
                         f"{record['validation_metrics']['rmse']:.4f} |")
    lines += ['', '### Todas as tentativas de arquitetura e época', '',
              'As curvas completas por candidato estão em `horizon_*/hyperparameter_epoch_history.csv`; '
              'os parâmetros, durações, escala, perda e RMSE final de cada braço constam de '
              '`horizon_*/*/metrics.json` e `training_history.csv`. O RSS abaixo é a memória '
              'após o ajuste, não o pico.', '']
    for horizon in (0, 1):
        directory = run_dir / f'horizon_{horizon}'
        trials = json.loads((directory / 'hyperparameter_trials.json').read_text(encoding='utf-8'))
        lines += [f'#### H{horizon} — {len(trials)} tentativas', '',
                  '| Família | Candidato | Ocultas | Alpha | Taxa | Época escolhida | Épocas executadas | RMSE seleção | MAE seleção | Parâmetros | Tempo (s) | RSS (MB) |',
                  '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for row in trials:
            params = row['params']
            lines.append(f"| {row['base_arm']} | {row['candidate_id']+1} | "
                         f"{','.join(map(str, params['hidden_layer_sizes']))} | "
                         f"{params['alpha']} | {params['learning_rate_init']} | "
                         f"{row['chosen_epochs']} | {row['epochs_completed']} | "
                         f"{row['selection_rmse']:.4f} | {row['selection_mae']:.4f} | "
                         f"{row['n_parameters']} | {row['fit_and_score_seconds']:.1f} | "
                         f"{row['rss_after_fit_mb']} |")
        lines.append('')
    lines += [
        '### Curvas observadas', '',
        'O primeiro gráfico mostra RMSE da seleção interna por época das configurações '
        'escolhidas; pontos marcam as épocas selecionadas. O segundo mostra a perda '
        'quadrática real dos reajustes finais, calculada na escala padronizada do '
        'target e distinta do RMSE. Nenhuma curva foi reconstruída por novo treino.', '',
        '![RMSE de seleção por época](graficos_treinamento/rmse_selecao_por_epoca.png)', '',
        '![Perda de treino por época](graficos_treinamento/perda_treino_por_epoca.png)', '',
        '### Conceitos e fontes', '',
        '| Conceito | Fonte |', '|---|---|',
        '| MLP, ReLU, `partial_fit` e perda | [MLPRegressor, scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html). |',
        '| Escala das entradas | [Redes neurais supervisionadas, scikit-learn](https://scikit-learn.org/stable/modules/neural_networks_supervised.html). |',
        '| Padronização ajustada somente no treino | [StandardScaler, scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html). |',
        '| Adam | [Kingma e Ba (2015)](https://arxiv.org/abs/1412.6980). |',
        '| Divisões, GPA, grafos, métricas e figuras | [Protocolo comum](protocols/EXPERIMENTOS_ENRIQUECIDOS_2026.md) e [protocolo MLP](protocols/EXPERIMENTO_07_MLP.md). |', '',
        'A avaliação H1 2020→2021 é complementar e não participa da escolha do vencedor.', '',
    ]
    (run_dir / 'methodology_appendix.md').write_text('\n'.join(lines), encoding='utf-8')


def rebuild_report(run_dir: Path) -> None:
    from plot_enriched_linear_external import render_one
    from report_enriched_linear_all import run_report

    run_dir = run_dir.resolve()
    status_path = run_dir / 'run.json'
    status = json.loads(status_path.read_text(encoding='utf-8'))
    if status.get('kind') != 'enriched_mlp' or not status.get('final_external_test_used'):
        raise ValueError('Expected an externally evaluated MLP run')
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
    _training_figures(run_dir)
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
    print(run_dir / 'RELATORIO.md')


def _snapshot(cfg: dict, run_dir: Path) -> None:
    shutil.copy2(cfg['path'], run_dir / 'config.yaml')
    project = project_root()
    sources = ('run_enriched_mlp.py',
               'src/services/enriched_mlp.py',
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
                     'protocols/EXPERIMENTO_07_MLP.md',
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
        status = dict(kind='enriched_mlp', status='running',
                      started=datetime.now(timezone.utc).isoformat(), run=str(run_dir),
                      final_external_test_used=False)
        common._write_json(run_dir / 'run.json', status)
    else:
        run_dir = resume.resolve()
        if not run_dir.is_dir() or run_dir.parent != cfg['output'].resolve():
            raise ValueError('Resume path must identify an MLP run inside its output root')
        status = json.loads((run_dir / 'run.json').read_text(encoding='utf-8'))
        if status.get('kind') != 'enriched_mlp' or status.get('status') == 'completed':
            raise ValueError('Only an unfinished MLP run can be resumed')
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
            logging.info('MLP input SHA-256 audit: %s files before training', total)

            def progress(count: int, relative: str) -> None:
                if count == 1 or count % 10 == 0 or count == total:
                    logging.info('MLP input audit %s/%s: %s', count, total, relative)

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
    except KeyboardInterrupt:
        status.update(status='interrupted', interrupted=datetime.now(timezone.utc).isoformat())
        common._write_json(run_dir / 'run.json', status)
        logging.warning('MLP run interrupted; resume with --resume %s', run_dir)
        raise SystemExit(130)
    except Exception as exc:
        status.update(status='failed', failed=datetime.now(timezone.utc).isoformat(),
                      error=repr(exc))
        common._write_json(run_dir / 'run.json', status)
        logging.exception('MLP run failed; partial artifacts kept at %s', run_dir)
        raise
    finally:
        logger.removeHandler(file_handler)
        logger.removeHandler(stream_handler)
        file_handler.close()
        stream_handler.close()

