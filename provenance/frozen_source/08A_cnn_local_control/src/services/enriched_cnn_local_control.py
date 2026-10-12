"""Paired, fixed-recipe ablation of the 41 km branch in experiment 08."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import importlib.metadata
import json
import logging
import platform
import shutil
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from src.services import enriched_cnn as cnn
from src.services import enriched_linear as common


KIND = 'enriched_cnn_local_control'
PAIRS = ((0, 2020, 2020, False), (0, 2021, 2021, False),
         (0, 2022, 2022, False), (1, 2021, 2022, False),
         (1, 2020, 2021, True))


class LocalPatchView(cnn.PatchView):
    """Read only scale zero from the frozen two-scale patch banks."""

    def batch(self, ids: np.ndarray) -> np.ndarray:
        out = np.empty((len(ids), 1, 24, 5, 5), dtype=np.float32)
        start = 0
        for (bank, indices), end in zip(self.segments, self.offsets):
            mask = (ids >= start) & (ids < end)
            if np.any(mask):
                out[mask, 0] = bank[indices[ids[mask] - start], 0]
            start = int(end)
        return out


def load_config(path: Path) -> dict:
    cfg = cnn.load_config(path)
    raw = cfg['raw']
    control = raw.get('control')
    if not isinstance(control, dict) or set(control) != {'reference_run'}:
        raise ValueError('control.reference_run must identify the completed two-scale run')
    reference = (cfg['path'].parent / control['reference_run']).resolve()
    if cfg['output'].is_relative_to(reference) or reference.is_relative_to(cfg['output']):
        raise ValueError('Control and reference output directories must be separate')
    status = json.loads((reference / 'run.json').read_text(encoding='utf-8'))
    if status.get('kind') != 'enriched_cnn' or status.get('status') != 'completed' or not status.get('final_external_test_used'):
        raise ValueError('Reference must be the completed, externally evaluated two-scale CNN')
    if Path(status['sample_source']).resolve() != cfg['sample_source']:
        raise ValueError('Frozen sample source differs from reference')
    reference_raw = yaml.safe_load((reference / 'config.yaml').read_text(encoding='utf-8'))
    comparable = dict(raw)
    comparable.pop('control')
    comparable['output_root'] = reference_raw['output_root']
    if comparable != reference_raw:
        raise ValueError('Control changes a baseline setting other than output_root')
    cfg['reference_run'] = reference
    return cfg


def _reference(cfg: dict, horizon: int) -> tuple[dict, dict, dict]:
    source = cfg['reference_run'] / f'horizon_{horizon}'
    winner = json.loads((source / 'winner.json').read_text(encoding='utf-8'))
    record = json.loads((source / winner['arm'] / 'metrics.json').read_text(encoding='utf-8'))
    artifact = torch.load(source / winner['arm'] / 'model.pt', map_location='cpu', weights_only=True)
    if artifact.get('wide_context', True) is not True:
        raise ValueError('Reference artifact is not a two-scale CNN')
    if (artifact['feature_names'] != winner['feature_names'] or
            record['feature_names'] != winner['feature_names'] or
            artifact['epoch'] != record['selected_epoch'] or
            artifact['params'] != record['hyperparameters'] or
            common._sha256(source / winner['arm'] / 'model.pt') != record['model_sha256']):
        raise ValueError(f'H{horizon} reference winner, artifact or hash differs')
    return winner, record, artifact


def _verify_banks(cfg: dict, horizon: int) -> None:
    folder = cfg['reference_run'] / f'horizon_{horizon}'
    marker = json.loads((folder / 'patch_bank.json').read_text(encoding='utf-8'))
    expected_hash = common._sha256(cfg['sample_source'] / f'horizon_{horizon}' / 'samples.npz')
    if (marker['horizon'] != horizon or marker['patch_size'] != 5 or
            marker['wide_step_pixels'] != cnn.WIDE_STEP or
            marker['sample_source_sha256'] != expected_hash or
            common._sha256(folder / 'samples.npz') != expected_hash):
        raise ValueError(f'H{horizon} frozen patches do not match frozen samples')
    for stage in cnn.STAGES:
        bank = np.load(folder / f'{stage}_patches.npy', mmap_mode='r')
        if list(bank.shape) != marker['shape'][stage] or bank.shape[1:] != (2, 24, 5, 5):
            raise ValueError(f'H{horizon} {stage} patch bank has unexpected shape')


def check_only(path: Path) -> None:
    cfg = load_config(path)
    cnn.check_only(path)
    for horizon in (0, 1):
        _verify_banks(cfg, horizon)
        _reference(cfg, horizon)
    for region, years in (('treinamento', (2020, 2021, 2022)),
                          ('teste_final', (2020, 2021, 2022))):
        for year in years:
            folder = cfg['reference_run'] / 'aligned_inputs'
            if not (folder / f'{region}_{year}.npy').is_file() or not (folder / f'{region}_{year}.json').is_file():
                raise FileNotFoundError(f'Frozen aligned input cache missing: {region} {year}')
    print('CONTROLE LOCAL OK: H0/H1, mesmos bancos de patches, vencedores, '
          'configuração e caches da CNN original; targets externos não abertos.')


def _snapshot(cfg: dict, run_dir: Path) -> None:
    shutil.copy2(cfg['path'], run_dir / 'config.yaml')
    project = Path(__file__).parents[2]
    sources = ('run_enriched_cnn_local_control.py',
               'src/services/enriched_cnn_local_control.py',
               'src/services/enriched_cnn.py',
               'src/services/enriched_linear.py',
               'src/services/enriched_hgb.py',
               'src/services/temporal_service.py',
               'src/splitting/spatial_split.py',
               'src/geospatial/raster_io.py',
               'plot_enriched_linear_external.py')
    records = []
    for name in sources:
        path = project / name
        destination = run_dir / 'source_code' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        records.append({'path': name, 'sha256': common._sha256(path)})
    common._write_json(run_dir / 'source_files.json', records)
    for name in ('protocols/EXPERIMENTO_08A_CNN_CONTROLE_LOCAL.md',
                 'protocols/EXPERIMENTO_08_CNN.md',
                 'protocols/EXPERIMENTOS_ENRIQUECIDOS_2026.md'):
        destination = run_dir / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(project / name, destination)
    packages = {name: importlib.metadata.version(name) for name in
                ('numpy', 'torch', 'rasterio', 'matplotlib', 'scikit-learn',
                 'PyYAML', 'psutil')}
    common._write_json(run_dir / 'environment.json',
                       {'python': sys.version, 'platform': platform.platform(),
                        'packages': packages, 'cuda_available': torch.cuda.is_available(),
                        'cuda_device': torch.cuda.get_device_name(0)
                        if torch.cuda.is_available() else None})


def _train_one(cfg: dict, horizon: int, run_dir: Path, device: torch.device) -> dict:
    folder = run_dir / f'horizon_{horizon}'
    folder.mkdir(exist_ok=True)
    winner, original_record, original_artifact = _reference(cfg, horizon)
    model_path, metrics_path = folder / 'model.pt', folder / 'metrics.json'
    if metrics_path.is_file() and model_path.is_file():
        record = json.loads(metrics_path.read_text(encoding='utf-8'))
        if (record['reference_model_sha256'] != original_record['model_sha256'] or
                record['model_sha256'] != common._sha256(model_path)):
            raise ValueError(f'H{horizon} control checkpoint differs from reference')
        return record
    source = cfg['reference_run'] / f'horizon_{horizon}'
    stages = cnn._load_stages(source / 'samples.npz')
    banks = {stage: np.load(source / f'{stage}_patches.npy', mmap_mode='r')
             for stage in cnn.STAGES}
    fit_view = LocalPatchView(
        [(banks[stage], np.arange(len(stages[stage][1])))
         for stage in ('ajuste', 'selecao_camadas')],
        [stages[stage][1] for stage in ('ajuste', 'selecao_camadas')])
    validation_view = LocalPatchView(
        [(banks['validacao_modelo'], np.arange(len(stages['validacao_modelo'][1])))],
        [stages['validacao_modelo'][1]])
    columns = tuple(original_artifact['columns'])
    scale = {'input_mean': np.asarray(original_artifact['input_mean'], dtype=np.float32),
             'input_scale': np.asarray(original_artifact['input_scale'], dtype=np.float32),
             'target_mean': original_artifact['target_mean'],
             'target_scale': original_artifact['target_scale']}
    params = original_artifact['params']
    epochs = int(original_artifact['epoch'])
    logging.info('Local control H%s: %s channels, %s epochs, %s',
                 horizon, len(columns), epochs, params)
    artifact, history, _ = cnn.fit_cnn(
        fit_view, columns, scale, params, epochs, cfg['raw']['cnn'], device,
        wide_context=False)
    if artifact['wide_context'] is not False or artifact['epoch'] != epochs:
        raise ValueError('Local-only model was not trained as planned')
    torch.save(artifact, model_path)
    common._write_csv(folder / 'training_history.csv', history)
    fit_prediction = cnn.predict_view(artifact, fit_view,
                                      cfg['raw']['cnn']['batch_size'], device)
    validation_prediction = cnn.predict_view(
        artifact, validation_view, cfg['raw']['cnn']['batch_size'], device)
    threshold = winner['high_target_threshold']
    record = {
        'horizon': horizon, 'reference_arm': winner['arm'],
        'reference_model_sha256': original_record['model_sha256'],
        'reference_validation_metrics': original_record['validation_metrics'],
        'reference_n_parameters': sum(v.numel() for v in original_artifact['state_dict'].values()),
        'n_features': len(columns), 'feature_names': original_artifact['feature_names'],
        'hyperparameters': params, 'epochs': epochs,
        'optimizer_steps': history[-1]['optimizer_steps'],
        'n_parameters': sum(v.numel() for v in artifact['state_dict'].values()),
        'model_path': str(model_path), 'model_sha256': common._sha256(model_path),
        'training_samples': len(fit_view), 'validation_samples': len(validation_view),
        'training_metrics': common.metrics(fit_view.y, fit_prediction, threshold),
        'validation_metrics': common.metrics(validation_view.y, validation_prediction,
                                             threshold),
        'validation_by_year': [
            {'target_year': int(year),
             'metrics': common.metrics(
                 validation_view.y[stages['validacao_modelo'][3] == year],
                 validation_prediction[stages['validacao_modelo'][3] == year], threshold)}
            for year in np.unique(stages['validacao_modelo'][3])],
    }
    common._write_json(metrics_path, record)
    common._write_json(folder / 'winner.json',
                       {'horizon': horizon, 'arm': winner['arm'],
                        'feature_names': winner['feature_names'],
                        'high_target_threshold': threshold,
                        'training_target_mean': winner['training_target_mean'],
                        'model_path': str(model_path)})
    logging.info('Local control H%s validation RMSE %.6f vs original %.6f',
                 horizon, record['validation_metrics']['rmse'],
                 original_record['validation_metrics']['rmse'])
    return record


def _external_record(root: Path, horizon: int, year: int, supplemental: bool) -> dict:
    folder = root / ('supplemental_H1_2021' if supplemental else f'horizon_{horizon}')
    return json.loads((folder / f'external_target_{year}_metrics.json').read_text(encoding='utf-8'))


def rebuild_report(run_dir: Path) -> None:
    from plot_enriched_linear_external import render_one

    run_dir = run_dir.resolve()
    status = json.loads((run_dir / 'run.json').read_text(encoding='utf-8'))
    if status.get('kind') != KIND or not status.get('final_external_test_used'):
        raise ValueError('Expected an externally evaluated local CNN control')
    reference = Path(status['reference_run'])
    records = [json.loads((run_dir / f'horizon_{h}' / 'metrics.json').read_text(encoding='utf-8'))
               for h in (0, 1)]
    lines = [
        '# Relatório final — experimento 08A, controle local da CNN', '',
        f'Execução: `{run_dir}`. Referência de duas escalas: `{reference}`.', '',
        '## Por que fizemos este controle', '',
        'A CNN original combina uma janela local 5×5 com outra grade esparsa 5×5 '
        'que alcança ±41 km. Embora tenha obtido RMSE de validação baixo, ela '
        'não superou a previsão zero em nenhum dos cinco pares externos. Como não '
        'havia uma CNN equivalente só com o ramo local, não era possível atribuir '
        'o ganho interno à segunda escala. Este controle retira o ramo amplo e '
        'mantém fixos os vencedores H0/H1, canais, amostras, escaladores, seed, '
        'hiperparâmetros, épocas e regra de avaliação. Não houve nova busca nem '
        'seleção de camadas.', '',
        'O controle ainda recebe canais KDE cujos pixels centrais incorporam '
        'suavização de cerca de 41 km. Portanto, o contraste mede a contribuição '
        'da **amostragem espacial adicional** do ramo, não a contribuição total '
        'dos KDEs. A retirada do ramo também reduz parâmetros. O buffer de três '
        'pixels não separava os contextos amplos da CNN original, o que pode '
        'favorecer sua validação. Os targets externos já haviam sido examinados '
        'antes deste controle; suas comparações são exploratórias e não foram '
        'usadas para ajustar o modelo.', '',
        '## Treinamento e validação pareada', '',
        '| Horizonte | Braço/canais congelados | Filtros | Épocas | Passos | '
        'Parâmetros local/original | RMSE local | RMSE original | Δ RMSE local−original | MAE local | MAE original |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for record in records:
        h = record['horizon']
        local, original = record['validation_metrics'], record['reference_validation_metrics']
        lines.append(
            f"| H{h} | {record['reference_arm']} / {record['n_features']} | "
            f"{record['hyperparameters']['filters']} | {record['epochs']} | "
            f"{record['optimizer_steps']} | {record['n_parameters']:,} / "
            f"{record['reference_n_parameters']:,} | {local['rmse']:.4f} | "
            f"{original['rmse']:.4f} | {local['rmse']-original['rmse']:+.4f} | "
            f"{local['mae']:.4f} | {original['mae']:.4f} |")
    lines += ['', 'Δ RMSE positivo significa que a CNN sem ramo amplo teve erro '
              'maior; negativo significa erro menor. A validação compara os '
              'mesmos centros e targets, mas os contextos vistos pelas duas '
              'arquiteturas têm extensão diferente.', '',
              '### Canais e configuração', '']
    for record in records:
        lines += [f"**H{record['horizon']} — {record['reference_arm']}:** "
                  f"{', '.join(record['feature_names'])}.", '',
                  f"AdamW: `{json.dumps(record['hyperparameters'])}`; "
                  f"{record['training_samples']:,} amostras de treino, "
                  f"{record['validation_samples']:,} de validação. "
                  f"RMSE treino {record['training_metrics']['rmse']:.4f}; "
                  f"R² validação {record['validation_metrics']['r2']:.4f}; "
                  f"NMAD validação {record['validation_metrics']['nmad']:.4f}; "
                  f"captura top 10% {100*record['validation_metrics']['capture_10pct']:.2f}%.", '']
    lines += ['Os JSONs H0/H1 guardam todas as métricas, inclusive subconjuntos '
              'do target e validação por ano. `training_history.csv` registra '
              'perda por época; os pesos estão em `model.pt`.', '',
              '## Área externa — mesmos cinco pares', '',
              '| Par | Pixels | RMSE local | RMSE original | Δ RMSE | RMSE zero | '
              'MAE local | R² local | Captura top 10% local |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    external = []
    for h, input_year, target_year, supplemental in PAIRS:
        local = _external_record(run_dir, h, target_year, supplemental)
        original = _external_record(reference, h, target_year, supplemental)
        if (local['target_sha256'] != original['target_sha256'] or
                local['common_pixels'] != original['common_pixels'] or
                local['input_year'] != input_year):
            raise ValueError(f'External pair H{h} {input_year}→{target_year} differs')
        external.append((local, original, supplemental))
        a, b = local['metrics'], original['metrics']
        lines.append(f"| H{h} {input_year}→{target_year}"
                     f"{' complementar' if supplemental else ''} | "
                     f"{local['common_pixels']:,} | {a['rmse']:.4f} | {b['rmse']:.4f} | "
                     f"{a['rmse']-b['rmse']:+.4f} | "
                     f"{local['reference_zero']['rmse']:.4f} | {a['mae']:.4f} | "
                     f"{a['r2']:.4f} | {100*a['capture_10pct']:.2f}% |")
    lines += ['', 'Resultados externos não selecionam uma nova arquitetura: '
              'esta região já foi examinada em experimentos anteriores. RMSE '
              'menor é melhor; a referência zero prevê zero em todos os pixels '
              'válidos. Zero no target significa ausência de registro naquela '
              'fotografia, não impossibilidade de ocorrência.', '',
              '## Curvas reais por época', '']
    fig, ax = plt.subplots(figsize=(10, 5))
    for record in records:
        import csv
        with (run_dir / f"horizon_{record['horizon']}" / 'training_history.csv').open(
                encoding='utf-8-sig', newline='') as handle:
            history = list(csv.DictReader(handle))
        ax.plot([int(row['epoch']) for row in history],
                [float(row['train_loss_scaled']) for row in history],
                label=f"H{record['horizon']}")
    ax.set(xlabel='Época', ylabel='MSE de treino padronizado',
           title='Controle local — perda observada por época')
    ax.grid(alpha=.2)
    ax.legend()
    fig.tight_layout()
    graph = run_dir / 'curva_treinamento.png'
    fig.savefig(graph, dpi=300)
    plt.close(fig)
    lines += ['![Perda de treino do controle local](curva_treinamento.png)', '',
              '## Imagens externas: target original e previsão local', '',
              'Cada figura mostra target e previsão na mesma máscara e escala. '
              'PNG 600 dpi; GeoTIFFs conservam a grade original.', '']
    figures = []
    for local, _, supplemental in external:
        h, iy, ty = local['horizon'], local['input_year'], local['target_year']
        destination = run_dir / 'supplemental_H1_2021' if supplemental else run_dir
        figure = render_one(local, destination)
        figures.append(figure)
        name = f'teste_externo_H{h}_{iy}_{ty}.png'
        figure_path = f'supplemental_H1_2021/{name}' if supplemental else name
        lines += [f"**H{h} · {iy}→{ty}**", '',
                  f"![Target original e previsão CNN local H{h} {iy}→{ty}]({figure_path})", '',
                  f"[GeoTIFF da previsão local]({Path(local['prediction']).relative_to(run_dir).as_posix()})", '']
    common._write_json(run_dir / 'figuras_teste_externo.json', figures)
    lines += ['## Reprodutibilidade e referências', '',
              f'Configuração, código e protocolo congelados neste run; '
              f'referência original em `{reference}`. Foram reutilizados '
              'os bancos de patches e caches de inputs da referência, somente '
              'para leitura. Os alvos externos não entraram no treino. '
              'CE90 posicional não se aplica a erro de intensidade.', '',
              '- [Protocolo deste controle](protocols/EXPERIMENTO_08A_CNN_CONTROLE_LOCAL.md).',
              '- [Protocolo CNN original](protocols/EXPERIMENTO_08_CNN.md).',
              '- [PyTorch Conv2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html).',
              '- [PyTorch AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html).',
              '- [Valavi et al. (2019)](https://doi.org/10.1111/2041-210X.13107).', '']
    (run_dir / 'RELATORIO.md').write_text('\n'.join(lines), encoding='utf-8')
    status['status'] = 'completed'
    common._write_json(run_dir / 'run.json', status)
    print(run_dir / 'RELATORIO.md')


def run(path: Path, resume: Path | None = None) -> None:
    cfg = load_config(path)
    cfg['output'].mkdir(parents=True, exist_ok=True)
    if resume is None:
        run_dir = cfg['output'] / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        run_dir.mkdir(exist_ok=False)
        status = {'kind': KIND, 'status': 'running',
                  'started': datetime.now(timezone.utc).isoformat(),
                  'run': str(run_dir), 'reference_run': str(cfg['reference_run']),
                  'final_external_test_used': False}
        common._write_json(run_dir / 'run.json', status)
    else:
        run_dir = resume.resolve()
        if not run_dir.is_dir() or run_dir.parent != cfg['output'].resolve():
            raise ValueError('Resume path must be an unfinished local control run')
        status = json.loads((run_dir / 'run.json').read_text(encoding='utf-8'))
        if status.get('kind') != KIND or status.get('status') == 'completed':
            raise ValueError('Only an unfinished local control can be resumed')
        if (run_dir / 'config.yaml').read_bytes() != cfg['path'].read_bytes():
            raise ValueError('Control config changed since run began')
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
        if not (run_dir / 'source_files.json').is_file():
            _snapshot(cfg, run_dir)
        for h in (0, 1):
            _verify_banks(cfg, h)
        cfg['_aligned_cache'] = cfg['reference_run'] / 'aligned_inputs'
        device = cnn._device(cfg['raw']['cnn'])
        torch.set_num_threads(cfg['raw']['runtime']['cpu_threads'])
        if device.type == 'cuda':
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        logging.info('CNN local-only control; reference %s; device %s',
                     cfg['reference_run'], device)
        records = [_train_one(cfg, h, run_dir, device) for h in (0, 1)]
        status['training_finished'] = datetime.now(timezone.utc).isoformat()
        status['reference_arms'] = {str(r['horizon']): r['reference_arm'] for r in records}
        common._write_json(run_dir / 'run.json', status)
        for h in (0, 1):
            winner = json.loads((run_dir / f'horizon_{h}' / 'winner.json').read_text(encoding='utf-8'))
            cnn.evaluate_external(cfg, winner, h, run_dir / f'horizon_{h}',
                                  cfg['raw']['evaluation'][f'final_h{h}_target_years'], device)
        supplement = run_dir / 'supplemental_H1_2021'
        supplement.mkdir(exist_ok=True)
        winner = json.loads((run_dir / 'horizon_1' / 'winner.json').read_text(encoding='utf-8'))
        cnn.evaluate_external(cfg, winner, 1, supplement, [2021], device)
        status['final_external_test_used'] = True
        status['external_finished'] = datetime.now(timezone.utc).isoformat()
        common._write_json(run_dir / 'run.json', status)
        rebuild_report(run_dir)
        status = json.loads((run_dir / 'run.json').read_text(encoding='utf-8'))
        status['finished'] = datetime.now(timezone.utc).isoformat()
        common._write_json(run_dir / 'run.json', status)
    except KeyboardInterrupt:
        status['status'] = 'interrupted'
        status['interrupted'] = datetime.now(timezone.utc).isoformat()
        common._write_json(run_dir / 'run.json', status)
        logging.warning('CNN local control interrupted; resume with --resume %s', run_dir)
        raise SystemExit(130)
    except Exception as exc:
        status['status'] = 'failed'
        status['failed'] = datetime.now(timezone.utc).isoformat()
        status['error'] = repr(exc)
        common._write_json(run_dir / 'run.json', status)
        logging.exception('CNN local control failed; partial artifacts kept at %s', run_dir)
        raise
    finally:
        logger.removeHandler(file_handler)
        logger.removeHandler(stream_handler)
        file_handler.close()
        stream_handler.close()
