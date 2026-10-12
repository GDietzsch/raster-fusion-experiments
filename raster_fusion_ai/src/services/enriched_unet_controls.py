"""Sequential, paired U-Net controls; no decisions based on external targets."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import csv
import ctypes
import gc
import importlib.metadata
import json
import logging
import os
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from src.services import enriched_unet as u
from src.services import enriched_cnn as cnn
from src.services import enriched_linear as common
from src.splitting.spatial_split import split_grid

KIND = 'enriched_unet_controls'
PROJECT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ('10A', '10B', '10C')
SOURCES = ('run_enriched_unet_controls.py', 'run_enriched_unet_stability.py',
           'run_enriched_unet_context.py', 'run_enriched_unet_gpa.py',
           'src/services/enriched_unet_controls.py', 'src/services/enriched_unet.py',
           'src/services/enriched_cnn.py', 'src/services/enriched_linear.py',
           'src/image_pipeline/models/unet.py', 'src/image_pipeline/models/cnn.py',
           'src/services/temporal_service.py', 'src/splitting/spatial_split.py',
           'src/geospatial/raster_io.py', 'plot_enriched_linear_external.py',
           'protocols/EXPERIMENTOS_10ABC_UNET.md',
           'protocols/EXPERIMENTO_10_UNET.md', 'protocols/EXPERIMENTOS_ENRIQUECIDOS_2026.md',
           'docs/GRADIENT_INPUTS.md', 'docs/GRAPH_INPUTS.md',
           'generate_gradient_inputs.py', 'generate_graph_inputs.py')


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    common._write_json(temporary, value)
    temporary.replace(path)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_config(path: Path) -> dict:
    path = path.resolve()
    raw = yaml.safe_load(path.read_text(encoding='utf-8'))
    base = u.load_config((path.parent / raw['base_config']).resolve())
    source = (path.parent / raw['source_run']).resolve()
    output = (path.parent / raw['output_root']).resolve()
    if not source.is_dir() or read_json(source / 'run.json').get('status') != 'completed':
        raise ValueError('Control source must be a completed U-Net run')
    if read_json(source / 'run.json').get('kind') != u.KIND:
        raise ValueError('Unexpected source kind')
    for protected in (source, base['root'], base['sample_source'], base['cache_source']):
        if output == protected or output.is_relative_to(protected) or protected.is_relative_to(output):
            raise ValueError('Controls output must not overlap source/data/cache')
    if len(raw['seeds']) != 3 or len(set(raw['seeds'])) != 3:
        raise ValueError('Exactly three distinct initialization seeds required')
    if any(type(s) is not int or s < 1 for s in raw['seeds']):
        raise ValueError('Seeds must be positive integers')
    for key in ('epochs', 'training_tiles_per_year', 'monitor_tiles_per_year', 'tile_seed'):
        if type(raw[key]) is not int or raw[key] < 1:
            raise ValueError(f'Invalid {key}')
    if raw['context_depth'] != 3 or raw['epochs'] != 11:
        raise ValueError('Frozen controls require depth 3 and eleven epochs')
    if raw.get('require_cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; refusing an unexpectedly slow CPU overnight run')
    winners = {h: read_json(source / f'horizon_{h}' / 'winner.json') for h in (0, 1)}
    for h, winner in winners.items():
        artifact = torch.load(winner['model_path'], map_location='cpu', weights_only=True)
        if artifact['feature_names'] != winner['feature_names'] or artifact['epoch'] != raw['epochs']:
            raise ValueError('Source winner channels or epoch differ from fixed protocol')
        if artifact['model_config']['depth'] != 5:
            raise ValueError('Expected depth-five source architecture')
        winner['params'] = artifact['params']
        winner['columns'] = artifact['columns']
        metric_path = source / f'horizon_{h}' / winner['arm'] / 'metrics.json'
        if common._sha256(Path(winner['model_path'])) != read_json(metric_path)['model_sha256']:
            raise ValueError('Source winner model hash mismatch')
    if winners[1]['columns'] != list(common._arm_indices('gpa')):
        raise ValueError('10C requires the complete GPA source winner in H1')
    return {'path': path, 'raw': raw, 'base': base, 'source': source,
            'output': output, 'winners': winners}


def job_plan(cfg: dict) -> list[dict]:
    """24 unique fits; B/C reuse A fixed references instead of training duplicates."""
    jobs = []
    for experiment in EXPERIMENTS:
        for h in ((0, 1) if experiment != '10C' else (1,)):
            winner = cfg['winners'][h]
            variants = ('rotating', 'fixed') if experiment == '10A' else (
                ('depth3',) if experiment == '10B' else ('magnitude', 'direction'))
            for variant in variants:
                columns = list(winner['columns'])
                if variant == 'magnitude':
                    columns = [i for i in columns if common.FEATURE_NAMES[i].endswith('__magnitude')]
                elif variant == 'direction':
                    columns = [i for i in columns if '__direcao_' in common.FEATURE_NAMES[i]]
                for seed in cfg['raw']['seeds']:
                    params = dict(winner['params'])
                    params['model_depth'] = cfg['raw']['context_depth'] if experiment == '10B' else 5
                    jobs.append({'id': f'{experiment}_H{h}_{variant}_seed{seed}',
                                 'experiment': experiment, 'horizon': h, 'variant': variant,
                                 'seed': seed, 'columns': columns, 'params': params,
                                 'schedule': 'rotating' if variant == 'rotating' else 'fixed'})
    return jobs


def precheck(path: Path) -> None:
    cfg = load_config(path)
    u.check_only(cfg['base']['path'])
    plan = job_plan(cfg)
    print(f'CONTROLS PRECHECK OK: {len(plan)} fits; 3 seeds; 11 epochs; '
          '10A -> 10B -> 10C; 57 external pairs after all models are frozen.')
    print(f"Output: {cfg['output']}")


@contextmanager
def exclusive_lock(path: Path):
    """OS lock is released after crash; no stale PID-based lock to delete manually."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open('a+b')
    handle.seek(0, 2)
    if handle.tell() == 0:
        handle.write(b'0')
        handle.flush()
    handle.seek(0)
    locked = False
    try:
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locked = True
        yield
    finally:
        if locked:
            handle.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()


@contextmanager
def awake(enabled: bool):
    active = False
    if enabled and os.name == 'nt':
        active = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
        if not active:
            logging.warning('Could not prevent automatic sleep; verify Windows power settings.')
    try:
        yield
    finally:
        if active:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def fingerprint(cfg: dict) -> dict:
    return {'config_sha256': common._sha256(cfg['path']),
            'base_config_sha256': common._sha256(cfg['base']['path']),
            'sources': {name: common._sha256(PROJECT / name) for name in SOURCES},
            'samples': {str(h): common._sha256(cfg['base']['sample_source'] / f'horizon_{h}' / 'samples.npz') for h in (0, 1)},
            'winners': {str(h): common._sha256(Path(cfg['winners'][h]['model_path'])) for h in (0, 1)}}


def choose_run(cfg: dict, experiment: str, resume: Path | None, new_run: bool, signature: dict) -> Path:
    output = cfg['output']
    output.mkdir(parents=True, exist_ok=True)
    if resume is not None:
        selected = resume.resolve()
        if selected.parent != output or not (selected / 'run.json').is_file():
            raise ValueError('Resume must name a controls run in the configured output')
    else:
        previous = sorted(output.glob('*/run.json'), reverse=True)
        selected = None
        if not new_run:
            for state_path in previous:
                state = read_json(state_path)
                if state.get('kind') == KIND and state.get('requested_experiment') == experiment:
                    selected = state_path.parent
                    break
        if selected is None:
            selected = output / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            selected.mkdir()
            write_json(selected / 'fingerprint.json', signature)
            write_json(selected / 'run.json', {'kind': KIND, 'status': 'prepared',
                       'requested_experiment': experiment, 'started': now(),
                       'source_run': str(cfg['source']), 'external_started': False})
            shutil.copy2(cfg['path'], selected / 'config.yaml')
            shutil.copy2(cfg['base']['path'], selected / 'base_config.yaml')
            shutil.copy2(cfg['source'] / 'environment.json', selected / 'source_environment.json')
            write_json(selected / 'environment.json', {'python': sys.version, 'torch': str(torch.__version__),
                       'numpy': np.__version__, 'cuda': torch.version.cuda,
                       'packages': {name: importlib.metadata.version(name) for name in
                                    ('torch', 'numpy', 'rasterio', 'matplotlib', 'scikit-learn', 'PyYAML', 'psutil')},
                       'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
            for name in SOURCES:
                dest = selected / 'source_code' / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(PROJECT / name, dest)
    if read_json(selected / 'fingerprint.json') != signature:
        raise ValueError(f'Sources/configuration/data changed; refusing to mix results in {selected}. Use --new-run for a new protocol.')
    if read_json(selected / 'run.json')['requested_experiment'] != experiment:
        raise ValueError('Resume experiment differs')
    return selected


def tile_plan(stage: tuple, groups: dict, cfg: dict, job: dict, monitor_groups: dict) -> dict:
    raw = cfg['raw']
    epochs = [u._choose_keys(groups, raw['training_tiles_per_year'],
                            epoch if job['schedule'] == 'rotating' else 1, raw['tile_seed'])
              for epoch in range(1, raw['epochs'] + 1)]
    union = sorted(set(key for keys in epochs for key in keys))
    ids = np.concatenate([groups[key] for key in union])
    y = stage[1][ids]
    return {'tile_seed': raw['tile_seed'], 'schedule': job['schedule'],
            'epoch_keys': epochs,
            'monitor_keys': u._choose_keys(monitor_groups, raw['monitor_tiles_per_year'], 1, raw['tile_seed'] + 700),
            'unique_training_tiles': len(union), 'unique_training_centers': len(ids),
            'available_centers': len(stage[1]),
            'sampled_target_summary': {'positive_fraction': float(np.mean(y > 0)),
                                       'mean': float(np.mean(y)), 'maximum': float(np.max(y))},
            'available_target_summary': {'positive_fraction': float(np.mean(stage[1] > 0)),
                                        'mean': float(np.mean(stage[1])), 'maximum': float(np.max(stage[1]))},
            'selection_rule': 'Uniform random tiles within each input year, independent of target values. Fixed plan shared across seeds.'}


def train_job(cfg: dict, job: dict, stages: dict, profile: dict,
              cache: u.InputCache, folder: Path, device: torch.device) -> dict:
    folder.mkdir(parents=True, exist_ok=True)
    signature_path = folder / 'job.json'
    if signature_path.is_file() and read_json(signature_path) != job:
        raise ValueError('Job configuration changed')
    write_json(signature_path, job)
    if (folder / 'metrics.json').is_file():
        record = read_json(folder / 'metrics.json')
        if common._sha256(Path(record['model_path'])) != record['model_sha256']:
            raise ValueError('Completed model changed')
        return record
    combined = u._combined_stages(stages)
    valid = stages['validacao_modelo']
    groups = u._stage_groups(combined, job['horizon'])
    valid_groups = u._stage_groups(valid, job['horizon'])
    plan = tile_plan(combined, groups, cfg, job, valid_groups)
    if (folder / 'tile_plan.json').is_file() and read_json(folder / 'tile_plan.json') != json.loads(json.dumps(plan)):
        raise ValueError('Tile plan changed on resume')
    write_json(folder / 'tile_plan.json', plan)
    columns = tuple(job['columns'])
    scale = cnn._scales(combined[0], combined[1], columns)
    settings = dict(cfg['base']['raw']['unet'])
    settings.update(random_state=job['seed'], tile_seed=cfg['raw']['tile_seed'],
                    tile_schedule=job['schedule'], monitor_only=True,
                    training_tiles_per_year=cfg['raw']['training_tiles_per_year'])
    artifact, history = u._fit(combined, groups, cache, columns, scale, job['params'],
                               cfg['raw']['epochs'], settings, device, folder / 'fit',
                               (valid, valid_groups, cache, [tuple(k) for k in plan['monitor_keys']]))
    # The shared fit helper calls these fields "selection". These controls ONLY
    # observe validation; no epoch is selected and no early stopping is applied.
    diagnostic = []
    for item in history:
        item = dict(item)
        item['monitor_validation_rmse'] = item.pop('selection_rmse')
        item['monitor_validation_mae'] = item.pop('selection_mae')
        diagnostic.append(item)
    common._write_csv(folder / 'training_history.csv', diagnostic)
    model_path = folder / 'fit' / 'model.pt'
    train_ids, train_prediction = u._predict_stage(artifact, combined, groups, cache, device)
    valid_ids, valid_prediction = u._predict_stage(artifact, valid, valid_groups, cache, device)
    threshold = cfg['winners'][job['horizon']]['high_target_threshold']
    record = dict(job)
    record.update(model_path=str(model_path), model_sha256=common._sha256(model_path),
                  n_features=len(columns), feature_names=artifact['feature_names'],
                  model_config=artifact['model_config'], epochs=artifact['epoch'],
                  optimizer_steps=int(history[-1]['optimizer_steps']),
                  parameters=sum(p.numel() for p in u.UNet(len(columns), artifact['model_config']).parameters()),
                  training_target_mean=float(np.mean(combined[1])), high_target_threshold=threshold,
                  training_metrics=common.metrics(combined[1][train_ids], train_prediction, threshold),
                  validation_metrics=common.metrics(valid[1][valid_ids], valid_prediction, threshold),
                  validation_by_year=[{'year': int(y), 'metrics': common.metrics(
                      valid[1][valid_ids][valid[3][valid_ids] == y],
                      valid_prediction[valid[3][valid_ids] == y], threshold)} for y in np.unique(valid[3])],
                  validation_reference_zero=common.metrics(valid[1][valid_ids], np.zeros(len(valid_ids)), subsets=False),
                  validation_reference_training_mean=common.metrics(valid[1][valid_ids],
                      np.full(len(valid_ids), np.mean(combined[1])), subsets=False),
                  validation_uncertainty=common._bootstrap_block_errors(
                      valid[1][valid_ids], valid_prediction, valid[2][valid_ids, 0], valid[2][valid_ids, 1],
                      profile['width'], seed=cfg['raw']['tile_seed']),
                  actual_training_centers=plan['unique_training_centers'],
                  monitor_centers=sum(len(valid_groups[tuple(k)]) for k in plan['monitor_keys']),
                  completed=now())
    write_json(folder / 'metrics.json', record)
    logging.info('%s completed: validation RMSE %.6f', job['id'], record['validation_metrics']['rmse'])
    return record


def external_job(cfg: dict, record: dict, folder: Path, device: torch.device) -> list[dict]:
    from plot_enriched_linear_external import render_one
    dest = folder / 'external'
    dest.mkdir(exist_ok=True)
    h = record['horizon']
    results = u.evaluate_external(cfg['base'], record, h, dest,
                                 [2020, 2021, 2022] if h == 0 else [2022, 2021], device)
    images = []
    for result in results:
        metadata_path = dest / f"image_{result['target_year']}.json"
        if metadata_path.is_file() and Path(read_json(metadata_path)['path']).is_file():
            images.append(read_json(metadata_path))
            continue
        result = dict(result, prediction_label=f"{record['variant']} · seed {record['seed']}")
        metadata = render_one(result, dest)
        write_json(metadata_path, metadata)
        images.append(metadata)
    write_json(folder / 'external_completed.json', {'pairs': len(results), 'images': images, 'finished': now()})
    return results


def belongs(record: dict, experiment: str) -> bool:
    return record['experiment'] == experiment or (
        record['experiment'] == '10A' and record['variant'] == 'fixed' and
        (experiment == '10B' or experiment == '10C' and record['horizon'] == 1))


def label(record: dict, experiment: str) -> str:
    if record['experiment'] == '10A' and record['variant'] == 'fixed':
        return {'10A': 'fixed', '10B': 'depth5_reference', '10C': 'magnitude_direction_reference'}[experiment]
    return record['variant']


def aggregate(rows: list[dict], keys: tuple[str, ...], metrics: tuple[str, ...]) -> list[dict]:
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    result = []
    for values, items in sorted(groups.items()):
        entry = dict(zip(keys, values))
        entry['n_seeds'] = len(items)
        for metric in metrics:
            values_array = np.asarray([i[metric] for i in items if i.get(metric) is not None], dtype=float)
            entry[metric + '_mean'] = float(values_array.mean()) if len(values_array) else None
            entry[metric + '_sd'] = float(values_array.std(ddof=1)) if len(values_array) > 1 else None
        result.append(entry)
    return result


def fmt(value) -> str:
    return '—' if value is None else f'{value:.4f}'


def curves(folder: Path, record: dict) -> None:
    path = folder / 'training_history.csv'
    if not path.is_file():
        return
    with path.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for ax, key, title in zip(axes, ('train_loss_scaled', 'monitor_validation_rmse'),
                              ('MSE de treino padronizado', 'RMSE no monitor fixo de validação')):
        ax.plot([int(r['epoch']) for r in rows], [float(r[key]) for r in rows], marker='.')
        ax.set(xlabel='Época', ylabel=title, title=title)
        ax.grid(alpha=.2)
    fig.suptitle(record['id'] + ' — monitor descritivo, sem escolha de época')
    fig.savefig(folder / 'curvas.png', dpi=300)
    plt.close(fig)


def rebuild_reports(run_dir: Path) -> None:
    run_dir = run_dir.resolve()
    state = read_json(run_dir / 'run.json')
    records = [read_json(p) for p in sorted((run_dir / 'jobs').glob('*/metrics.json'))]
    plan = read_json(run_dir / 'plan.json')
    for r in records:
        folder = run_dir / 'jobs' / r['id']
        if not (folder / 'curvas.png').is_file():
            curves(folder, r)
    index = ['# Controles U-Net 10A, 10B e 10C', '',
             f"Estado: **{state['status']}**. Modelos concluídos: {len(records)}/{len(plan)}.", '',
             'Os modelos são congelados antes de abrir targets externos. Nenhuma semente é escolhida pelo teste externo.', '',
             '[Protocolo e justificativas](source_code/protocols/EXPERIMENTOS_10ABC_UNET.md).', '',
             'Os mapas são gerados automaticamente; a inspeção visual humana fica pendente para a análise após a execução.', '']
    for experiment in EXPERIMENTS:
        selected = [r for r in records if belongs(r, experiment)]
        if not selected:
            continue
        out = run_dir / experiment
        out.mkdir(exist_ok=True)
        rows, externals = [], []
        for r in selected:
            rows.append({'horizon': r['horizon'], 'variant': label(r, experiment), 'seed': r['seed'],
                         'channels': r['n_features'], 'epochs': r['epochs'], 'parameters': r['parameters'],
                         'training_rmse': r['training_metrics']['rmse'],
                         **{k: r['validation_metrics'].get(k) for k in ('rmse', 'mae', 'r2', 'bias', 'nmad', 'p90_absolute_error', 'capture_5pct', 'capture_10pct', 'capture_20pct')},
                         'job': r['id']})
            for path in sorted((run_dir / 'jobs' / r['id'] / 'external').glob('external_target_*_metrics.json')):
                e = read_json(path)
                persist = e.get('reference_persistence')
                # Shared external helper returns either None or a record containing metrics.
                persist_metrics = (persist.get('metrics', persist) if isinstance(persist, dict) else {})
                externals.append({'horizon': r['horizon'], 'variant': label(r, experiment), 'seed': r['seed'],
                                  'input_year': e['input_year'], 'target_year': e['target_year'],
                                  'channels': r['n_features'], 'pixels': e['common_pixels'],
                                  **{k: e['metrics'].get(k) for k in ('rmse', 'mae', 'r2', 'bias', 'nmad', 'p90_absolute_error', 'capture_5pct', 'capture_10pct', 'capture_20pct')},
                                  'zero_rmse': e['reference_zero']['rmse'],
                                  'training_mean_rmse': e['reference_training_mean']['rmse'],
                                  'persistence_rmse': persist_metrics.get('rmse'), 'job': r['id']})
        common._write_csv(out / 'validation_all.csv', rows)
        summary = aggregate(rows, ('horizon', 'variant'), ('rmse', 'mae', 'r2', 'bias', 'nmad', 'p90_absolute_error', 'capture_5pct', 'capture_10pct', 'capture_20pct'))
        common._write_csv(out / 'validation_summary.csv', summary)
        if externals:
            common._write_csv(out / 'external_all.csv', externals)
            common._write_csv(out / 'external_summary.csv', aggregate(externals,
                              ('horizon', 'variant', 'input_year', 'target_year'), ('rmse', 'mae', 'r2', 'bias', 'nmad', 'p90_absolute_error', 'capture_5pct', 'capture_10pct', 'capture_20pct')))
        # Paired changes per initialization seed; negative error delta favors intervention.
        reference = {'10A': 'rotating', '10B': 'depth5_reference', '10C': 'magnitude_direction_reference'}[experiment]
        pairs = []
        for row in rows:
            baseline = next((x for x in rows if x['variant'] == reference and
                             x['seed'] == row['seed'] and x['horizon'] == row['horizon']), None)
            if row['variant'] != reference and baseline:
                pairs.append({'horizon': row['horizon'], 'seed': row['seed'], 'variant': row['variant'],
                              'reference': reference, 'delta_rmse': row['rmse'] - baseline['rmse'],
                              'delta_mae': row['mae'] - baseline['mae']})
        if pairs:
            common._write_csv(out / 'paired_validation_deltas.csv', pairs)
        doc = [f'# Experimento {experiment} — controle U-Net', '',
               '[Protocolo congelado e referências](../source_code/protocols/EXPERIMENTOS_10ABC_UNET.md).', '',
               'H0 = inputs do mesmo ano; H1 = inputs do ano anterior. Esses rótulos não são testes formais de hipótese nula/alternativa.', '',
               'Todas as condições usam 11 épocas, AdamW e os hiperparâmetros herdados do vencedor original. '
               'As curvas de validação são diagnósticas: não acionam parada nem escolhem época. '
               'Média e desvio padrão amostral refletem três inicializações; não são intervalos de confiança populacionais.', '',
               '10A compara rotação e plano fixo pareados por semente. 10B reutiliza o plano fixo de 10A e reduz a profundidade de 5 para 3: '
               'campo receptivo e capacidade mudam juntos. 10C usa somente H1, vencedor GPA, e compara magnitude, seno/cosseno da direção e ambos. '
               'Referências reutilizadas são os mesmos modelos de 10A, não novas repetições.', '',
               'O plano fixo é sorteado por ano sem consultar o target; não há garantia de representar todos os hotspots. '
               'tile_plan.json registra a cobertura e distribuições. Os contextos podem atravessar as zonas internas; '
               'o buffer de três pixels permanece uma limitação. A área externa já foi explorada anteriormente.', '',
               '## Validação completa: média ± desvio padrão', '',
               '| Horizonte | Condição | Sementes | RMSE | MAE | R² |',
               '|---|---|---:|---:|---:|---:|']
        for s in summary:
            doc.append(f"| H{s['horizon']} | {s['variant']} | {s['n_seeds']} | {fmt(s['rmse_mean'])} ± {fmt(s['rmse_sd'])} | {fmt(s['mae_mean'])} ± {fmt(s['mae_sd'])} | {fmt(s['r2_mean'])} |")
        doc += ['', '## Todas as execuções', '', '| Horizonte | Condição | Seed | Canais | Épocas | RMSE treino | RMSE validação | MAE | R² |', '|---|---|---:|---:|---:|---:|---:|---:|---:|']
        for r in rows:
            doc.append(f"| H{r['horizon']} | {r['variant']} | {r['seed']} | {r['channels']} | {r['epochs']} | {fmt(r['training_rmse'])} | {fmt(r['rmse'])} | {fmt(r['mae'])} | {fmt(r['r2'])} |")
        doc += ['', '[Tabela completa](validation_all.csv) · [Resumo](validation_summary.csv) · [Diferenças pareadas](paired_validation_deltas.csv)', '',
                '## Testes externos de todas as sementes', '',
                'Métricas nas unidades originais, sem normalização pelo target externo ou truncamento. '
                'NMAD descreve dispersão robusta dos resíduos; não substitui RMSE nem mede viés. '
                'P90 absoluto é erro de intensidade; CE90 posicional não se aplica. '
                'Persistência usa informação histórica adicional. Target zero significa ausência de registro nesta fotografia.', '',
                '| H | Entrada→target | Condição | Seed | Canais | Pixels | RMSE | MAE | R² | Viés | NMAD | P90 abs. | Captura 10% (fração) | RMSE zero | RMSE média treino | RMSE persistência |',
                '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for e in externals:
            doc.append(f"| H{e['horizon']} | {e['input_year']}→{e['target_year']} | {e['variant']} | {e['seed']} | {e['channels']} | {e['pixels']} | " + ' | '.join(fmt(e[k]) for k in ('rmse', 'mae', 'r2', 'bias', 'nmad', 'p90_absolute_error', 'capture_10pct', 'zero_rmse', 'training_mean_rmse', 'persistence_rmse')) + ' |')
        doc += ['', 'Os JSONs por modelo preservam todas as métricas, subconjuntos zero/positivo/alto, captura 5/10/20%, referências e incerteza espacial exploratória.', '',
                '## Canais, parâmetros, curvas e mapas por execução', '']
        for r in selected:
            relative = f"../jobs/{r['id']}"
            doc += [f"### {label(r, experiment)} · H{r['horizon']} · seed {r['seed']}", '',
                    f"Canais mantidos: {', '.join(r['feature_names'])}.", '',
                    'Canais removidos em relação ao vencedor original: ' + (', '.join(
                        name for name in read_json(run_dir / 'source_winners.json')[str(r['horizon'])]['feature_names']
                        if name not in r['feature_names']) or 'nenhum') + '.', '',
                    f"Parâmetros: `{json.dumps(r['params'])}`; {r['parameters']:,} parâmetros treináveis; "
                    f"{r['epochs']} épocas; {r['optimizer_steps']} passos; {r['actual_training_centers']:,} centros distintos supervisionados.", '',
                    f"[Métricas completas]({relative}/metrics.json) · [Plano espacial]({relative}/tile_plan.json) · [Histórico]({relative}/training_history.csv)", '',
                    f"![Curvas {r['id']}]({relative}/curvas.png)", '']
            for image_path in sorted((run_dir / 'jobs' / r['id'] / 'external').glob('image_*.json')):
                info = read_json(image_path)
                doc += [f"**H{r['horizon']} {info['input_year']}→{info['target_year']} — target e previsão na mesma escala, PNG 600 dpi.**", '',
                        f"![Comparação externa]({relative}/external/{Path(info['path']).name})", '',
                        f"[GeoTIFF]({relative}/external/{Path(info['prediction']).name})", '']
        doc += ['## Referências', '', references(), '']
        (out / 'RELATORIO.md').write_text('\n'.join(doc), encoding='utf-8')
        index += [f'- [{experiment}: relatório, métricas, curvas e imagens]({experiment}/RELATORIO.md)']
    (run_dir / 'RELATORIO.md').write_text('\n'.join(index) + '\n', encoding='utf-8')


def references() -> str:
    return '\n'.join([
        '- [Ronneberger et al. (2015), U-Net](https://arxiv.org/abs/1505.04597): arquitetura de contração/expansão e conexões entre escalas.',
        '- [Araujo, Norris e Sim (2019), Computing Receptive Fields](https://distill.pub/2019/computing-receptive-fields/): campo receptivo e dependência da arquitetura.',
        '- [Ioffe e Szegedy (2015), BatchNorm](https://arxiv.org/abs/1502.03167): normalização; em treino agrega estatísticas espaciais, portanto o campo receptivo convolucional local não descreve toda a dependência durante otimização.',
        '- [Loshchilov e Hutter (2019), AdamW](https://arxiv.org/abs/1711.05101): regularização e otimizador.',
        '- [Valavi et al. (2019), blockCV](https://doi.org/10.1111/2041-210X.13107): avaliação espacial e dependência entre amostras.',
        '- [PyTorch, reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html): sementes e limites de reprodutibilidade entre plataformas.',
        '- [scikit-learn, métricas de regressão](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics): RMSE, MAE e R².',
        '- [SciPy, median_abs_deviation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.median_abs_deviation.html): base do NMAD.',
        '- [NumPy, gradient](https://numpy.org/doc/stable/reference/generated/numpy.gradient.html) e [SciPy, cKDTree](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html): produtos de gradiente e grafos já congelados no experimento fonte.'])


def run(path: Path, experiment: str = 'all', resume: Path | None = None, new_run: bool = False) -> Path:
    cfg = load_config(path)
    cfg['output'].mkdir(parents=True, exist_ok=True)
    with exclusive_lock(cfg['output'] / '.execution.lock'):
        signature = fingerprint(cfg)
        run_dir = choose_run(cfg, experiment, resume, new_run, signature)
        status = read_json(run_dir / 'run.json')
        if status['status'] == 'completed':
            print(f"Already completed: {run_dir / 'RELATORIO.md'}")
            return run_dir
        jobs = job_plan(cfg)
        if experiment != 'all':
            jobs = [j for j in jobs if belongs(j, experiment)]
        write_json(run_dir / 'plan.json', jobs)
        write_json(run_dir / 'source_winners.json', cfg['winners'])
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        handlers = [logging.FileHandler(run_dir / 'training.log', mode='a', encoding='utf-8'), logging.StreamHandler()]
        for handler in handlers:
            handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
            logger.addHandler(handler)
        status.update(status='running', resumed=now())
        write_json(run_dir / 'run.json', status)
        print(f'RUN DIRECTORY: {run_dir}', flush=True)
        device = cnn._device(cfg['base']['raw']['unet'])
        try:
            with awake(cfg['raw'].get('prevent_sleep', False)):
                current_audit = common.audit_catalog(cfg['base'], verify_hashes=False)
                source_audit = read_json(cfg['source'] / 'input_audit.json')
                if [(r['path'], r['sha256']) for r in current_audit['recorded_files']] != [
                        (r['path'], r['sha256']) for r in source_audit['recorded_files']]:
                    raise ValueError('Catalog differs from experiment 10')
                write_json(run_dir / 'input_audit.json', current_audit)
                torch.set_num_threads(cfg['base']['raw']['runtime']['cpu_threads'])
                if device.type == 'cuda':
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = False
                profile = common._profile(cfg['base']['reference'])
                zones = split_grid((profile['height'], profile['width']), cfg['base']['raw']['split'])
                active_horizon, stages, cache = None, None, None
                for number, job in enumerate(jobs, 1):
                    status.update(current_job=job['id'], phase='training', completed_jobs=number - 1)
                    write_json(run_dir / 'run.json', status)
                    logging.info('CONTROL %s/%s: %s', number, len(jobs), job['id'])
                    h = job['horizon']
                    if active_horizon != h:
                        stages, cache = None, None
                        gc.collect()
                        source = cfg['base']['sample_source'] / f'horizon_{h}' / 'samples.npz'
                        cnn._verify_reused_patch_bank(cfg['base'], h, source)
                        stages = cnn._load_stages(source)
                        cnn._check_stages(stages, zones, h)
                        cache = u.InputCache(cfg['base'], 'treinamento', profile)
                        active_horizon = h
                    train_job(cfg, job, stages, profile, cache, run_dir / 'jobs' / job['id'], device)
                    rebuild_reports(run_dir)
                    gc.collect()
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                stages, cache, zones = None, None, None
                gc.collect()
                # All configurations/seeds are fixed BEFORE any external evaluation.
                status.update(phase='external', completed_jobs=len(jobs), models_frozen=now(), external_started=True)
                write_json(run_dir / 'run.json', status)
                for job in jobs:
                    status['current_job'] = job['id']
                    write_json(run_dir / 'run.json', status)
                    folder = run_dir / 'jobs' / job['id']
                    external_job(cfg, read_json(folder / 'metrics.json'), folder, device)
                    rebuild_reports(run_dir)
                    gc.collect()
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                status.update(status='completed', phase='completed', finished=now(), current_job=None)
                write_json(run_dir / 'run.json', status)
                rebuild_reports(run_dir)
                print(f"COMPLETED: {run_dir / 'RELATORIO.md'}", flush=True)
        except BaseException as exc:
            status.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                          error=repr(exc), stopped=now())
            write_json(run_dir / 'run.json', status)
            logging.exception('Partial artifacts retained. Repeat the same command to resume.')
            raise
        finally:
            for handler in handlers:
                logger.removeHandler(handler)
                handler.close()
        return run_dir
