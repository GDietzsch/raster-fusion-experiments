"""Document effective training settings and reconstruct diagnostic curves from saved runs.

This script never fits a model, opens an external target, or changes a winner.
"""
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
import csv
import hashlib
import json
import logging
import shutil

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parent
ARMS = ('originais', 'originais_filtradas', 'gpa', 'grafos', 'todas',
        'todas_filtradas')
LABELS = {
    'enriched_linear': 'Regressão linear',
    'enriched_ridge': 'Ridge',
    'enriched_hgb': 'HistGradientBoosting',
    'enriched_xgboost': 'XGBoost',
    'enriched_random_forest': 'Random Forest',
    'enriched_extra_trees': 'Extra Trees',
}
START = '<!-- BEGIN DETALHES_TREINAMENTO_V1 -->'
END = '<!-- END DETALHES_TREINAMENTO_V1 -->'
COLORS = {'originais': '#4263a6', 'gpa': '#d18b31',
          'grafos': '#23957a', 'todas': '#a34e7c'}


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f'No rows for {path}')
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(p, dtype=np.float64)-y),
                                 dtype=np.float64)))


def effective_parameters(kind: str, metric: dict, config: dict) -> dict:
    if kind == 'enriched_linear':
        return dict(pipeline=['StandardScaler', 'LinearRegression'],
                    scaler_with_mean=True, scaler_with_std=True,
                    fit_intercept=True, positive=False,
                    solver='least_squares', iterations='not_applicable')
    if kind == 'enriched_ridge':
        return dict(pipeline=['StandardScaler', 'Ridge'], solver='cholesky',
                    fit_intercept=True, lambda_per_sample=metric['lambda_per_sample'],
                    alpha_initial=metric['alpha_initial'],
                    alpha_refit=metric['alpha_refit'],
                    iterations='not_applicable')
    if kind == 'enriched_hgb':
        return dict(metric['estimator_parameters'],
                    actual_iterations=metric['actual_iterations'])
    if kind == 'enriched_xgboost':
        return dict(metric['estimator_parameters'],
                    n_estimators=metric['n_estimators'],
                    actual_rounds=metric['actual_rounds'],
                    n_jobs=metric['cpu_threads'])
    settings = config['extra_trees'] if kind == 'enriched_extra_trees' else config['random_forest']
    return dict(criterion=settings['criterion'], bootstrap=settings['bootstrap'],
                oob_score=False, n_jobs=metric['cpu_threads'],
                random_state=settings['random_state'],
                n_estimators=metric['n_estimators'],
                filtering_trees=metric['filtering_trees'],
                **metric['hyperparameters'])


def budget_label(kind: str, metric: dict) -> str:
    if kind == 'enriched_linear':
        return 'solução direta; sem épocas'
    if kind == 'enriched_ridge':
        return f"solução direta; λ={metric['lambda_per_sample']:.6g}; α={metric['alpha_refit']:.6g}"
    if kind == 'enriched_hgb':
        return f"{metric['actual_iterations']} iterações (limite {metric['max_iter']})"
    if kind == 'enriched_xgboost':
        return f"{metric['actual_rounds']} rodadas (limite {metric['n_estimators']})"
    return f"{metric['n_estimators']} árvores"


def collect_rows(run_dir: Path, kind: str, config: dict) -> list[dict]:
    rows = []
    for horizon in (0, 1):
        sampling = read_json(run_dir / f'horizon_{horizon}' / 'sampling.json')
        stage_counts = {stage: sum(int(item['sampled']) for item in sampling
                                   if item['stage'] == stage)
                        for stage in ('ajuste', 'selecao_camadas', 'validacao_modelo')}
        for arm in ARMS:
            folder = run_dir / f'horizon_{horizon}' / arm
            metric = read_json(folder / 'metrics.json')
            if metric['training_samples'] != stage_counts['ajuste'] + stage_counts['selecao_camadas']:
                raise ValueError(f'Fit sample count differs from sampling manifest: {folder}')
            if metric['validation_samples'] != stage_counts['validacao_modelo']:
                raise ValueError(f'Validation count differs from sampling manifest: {folder}')
            params = effective_parameters(kind, metric, config)
            rows.append(dict(horizon=f'H{horizon}', arm=arm,
                             n_features=metric['n_features'],
                             feature_names=';'.join(metric['feature_names']),
                             initial_fit_samples=stage_counts['ajuste'],
                             selection_samples=stage_counts['selecao_camadas'],
                             tuning_fit_samples=metric.get('selection_fit_samples',
                                                            stage_counts['ajuste']),
                             tuning_score_samples=metric.get('selection_score_samples',
                                                              stage_counts['selecao_camadas']),
                             final_fit_samples=metric['training_samples'],
                             validation_samples=metric['validation_samples'],
                             budget=budget_label(kind, metric),
                             effective_parameters=json.dumps(params, ensure_ascii=False,
                                                             sort_keys=True),
                             selection_rmse=metric['initial_selection_metrics']['rmse'],
                             refit_training_rmse=metric['training_metrics']['rmse'],
                             validation_rmse=metric['validation_metrics']['rmse'],
                             model_sha256=sha256(folder / 'model.joblib')))
    return rows


def plot_stages(rows: list[dict], destination: Path, label: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), constrained_layout=True)
    for horizon, ax in zip(('H0', 'H1'), axes):
        subset = [row for row in rows if row['horizon'] == horizon]
        x = np.arange(len(subset))
        for offset, key, name, color in ((-0.27, 'selection_rmse', 'Seleção (ajuste inicial)', '#9aa7b9'),
                                          (0, 'refit_training_rmse', 'Treino após reajuste', '#5785bd'),
                                          (0.27, 'validation_rmse', 'Validação', '#d37454')):
            ax.bar(x+offset, [row[key] for row in subset], width=0.25,
                   label=name, color=color)
        ax.set_xticks(x, [row['arm'].replace('_', '\n') for row in subset])
        ax.set_ylabel('RMSE (unidades do target)')
        ax.set_title(horizon)
        ax.grid(axis='y', alpha=0.25)
        ax.legend(ncol=3, fontsize=9)
    fig.suptitle(f'{label} · métricas por conjunto e etapa', fontsize=15)
    fig.supxlabel('Seleção usa o ajuste inicial; treino e validação usam o modelo reajustado. As bases diferem.',
                 fontsize=9)
    fig.savefig(destination, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_search(run_dir: Path, kind: str, destination: Path) -> bool:
    if kind == 'enriched_linear':
        return False
    spec = {
        'enriched_ridge': ('alpha_search.csv', 'lambda_per_sample', 'Lambda por amostra'),
        'enriched_hgb': ('max_iter_search.csv', 'max_iter', 'Limite de iterações'),
        'enriched_xgboost': ('n_estimators_search.csv', 'n_estimators', 'Limite de rodadas'),
        'enriched_random_forest': ('hyperparameter_trials.csv', 'candidate_id', 'ID da configuração'),
        'enriched_extra_trees': ('hyperparameter_trials.csv', 'candidate_id', 'ID da configuração'),
    }[kind]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for horizon, ax in zip((0, 1), axes):
        with (run_dir / f'horizon_{horizon}' / spec[0]).open(encoding='utf-8-sig', newline='') as stream:
            trials = list(csv.DictReader(stream))
        for arm in COLORS:
            subset = [row for row in trials if row['base_arm'] == arm]
            if kind in ('enriched_random_forest', 'enriched_extra_trees'):
                screening = [row for row in subset if row['stage'] == 'screening']
                refinement = [row for row in subset if row['stage'] == 'refinement']
                ax.scatter([int(row['candidate_id'])+1 for row in screening],
                           [float(row['selection_rmse']) for row in screening],
                           s=28, color=COLORS[arm], label=arm)
                ax.scatter([int(row['candidate_id'])+1 for row in refinement],
                           [float(row['selection_rmse']) for row in refinement],
                           s=65, marker='x', linewidths=2, color=COLORS[arm])
            else:
                ordered = sorted(subset, key=lambda row: float(row[spec[1]]))
                ax.plot([float(row[spec[1]]) for row in ordered],
                        [float(row['selection_rmse']) for row in ordered],
                        'o-', color=COLORS[arm], label=arm)
                chosen = [row for row in ordered if row['chosen'].lower() == 'true']
                ax.scatter([float(row[spec[1]]) for row in chosen],
                           [float(row['selection_rmse']) for row in chosen],
                           s=90, marker='*', color=COLORS[arm], edgecolor='black', zorder=5)
        if kind == 'enriched_ridge':
            ax.set_xscale('log')
        ax.set_title(f'H{horizon}')
        ax.set_xlabel(spec[2])
        ax.set_ylabel('RMSE na seleção interna')
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    if kind in ('enriched_random_forest', 'enriched_extra_trees'):
        key = 'extra_trees' if kind == 'enriched_extra_trees' else 'random_forest'
        trees = yaml.safe_load((run_dir / 'config.yaml').read_text(encoding='utf-8'))[key]['final_trees']
        meaning = f'× = finalistas reavaliados com {trees} árvores'
    else:
        meaning = '★ = configuração escolhida por família base'
    fig.suptitle(f'{LABELS[kind]} · busca interna; {meaning}', fontsize=13)
    fig.savefig(destination, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return True


def sampled_stage(archive, stage: str, seed: int, cap: int):
    years = archive[f'{stage}_target_year']
    rng = np.random.default_rng(seed)
    indices = np.sort(np.concatenate([
        rng.choice(np.flatnonzero(years == year),
                   size=min(cap, int(np.count_nonzero(years == year))), replace=False)
        for year in np.unique(years)]))
    return (archive[f'{stage}_x'][indices],
            archive[f'{stage}_y'][indices].astype(np.float64),
            dict(stage=stage, count=len(indices),
                 years=[int(year) for year in np.unique(years)],
                 indices_sha256=hashlib.sha256(indices.tobytes()).hexdigest()))


def prefix_counts(kind: str, n: int, config: dict) -> list[int]:
    result = {1, n}
    result.update(max(1, min(n, round(n*fraction)))
                  for fraction in np.linspace(0.05, 1, 20))
    if kind == 'enriched_hgb':
        result.update(i for i in config['hgb']['max_iter_candidates'] if i <= n)
    elif kind == 'enriched_xgboost':
        result.update(i for i in config['xgboost']['n_estimators_candidates'] if i <= n)
    else:
        key = 'extra_trees' if kind == 'enriched_extra_trees' else 'random_forest'
        result.update(i for i in (config[key]['screening_trees'],
                                  config[key]['filtering_trees']) if i <= n)
    return sorted(result)


def reconstruct_curves(run_dir: Path, kind: str, config: dict, directory: Path) -> None:
    all_rows = []
    provenance = dict(kind=kind, purpose='post-hoc diagnostic only; no refit or reselection',
                      generated_utc=datetime.now(timezone.utc).isoformat(),
                      script_sha256=sha256(Path(__file__)),
                      max_samples_per_stage_year=5000, horizons=[])
    for horizon in (0, 1):
        folder = run_dir / f'horizon_{horizon}'
        winner = read_json(folder / 'winner.json')
        artifact_path = folder / winner['arm'] / 'model.joblib'
        artifact = joblib.load(artifact_path)
        if artifact['arm'] != winner['arm'] or artifact['feature_names'] != winner['feature_names']:
            raise ValueError(f'Winner and artifact disagree: {folder}')
        model = artifact['model']
        columns = tuple(artifact['columns'])
        samples_path = folder / 'samples.npz'
        with np.load(samples_path, allow_pickle=False) as archive:
            sample = [sampled_stage(archive, stage, 8200+horizon*100+offset, 5000)
                      for offset, stage in enumerate(('ajuste', 'selecao_camadas',
                                                      'validacao_modelo'))]
        x_train = np.concatenate((sample[0][0][:, columns], sample[1][0][:, columns]))
        y_train = np.concatenate((sample[0][1], sample[1][1]))
        x_valid, y_valid = sample[2][0][:, columns], sample[2][1]
        if kind == 'enriched_hgb':
            n = int(model.n_iter_)
        elif kind == 'enriched_xgboost':
            n = int(model.get_booster().num_boosted_rounds())
        else:
            n = len(model.estimators_)
        checkpoints = prefix_counts(kind, n, config)
        horizon_rows = []
        if kind == 'enriched_hgb':
            iterator = ((step, train_pred, valid_pred)
                        for step, (train_pred, valid_pred) in enumerate(
                            zip(model.staged_predict(x_train),
                                model.staged_predict(x_valid)), start=1)
                        if step in checkpoints)
        elif kind == 'enriched_xgboost':
            iterator = ((step,
                         model.predict(x_train, iteration_range=(0, step)),
                         model.predict(x_valid, iteration_range=(0, step)))
                        for step in checkpoints)
        else:
            def forest_prefixes():
                train_sum = np.zeros(len(y_train), dtype=np.float64)
                valid_sum = np.zeros(len(y_valid), dtype=np.float64)
                for step, tree in enumerate(model.estimators_, start=1):
                    train_sum += tree.predict(x_train)
                    valid_sum += tree.predict(x_valid)
                    if step in checkpoints:
                        yield step, train_sum/step, valid_sum/step
            iterator = forest_prefixes()
        for step, train_pred, valid_pred in iterator:
            horizon_rows.append(dict(horizon=f'H{horizon}', winner=winner['arm'],
                                     n_estimators_or_iterations=step,
                                     sampled_fit_n=len(y_train),
                                     sampled_fit_rmse=rmse(y_train, train_pred),
                                     sampled_validation_n=len(y_valid),
                                     sampled_validation_rmse=rmse(y_valid, valid_pred)))
        if [row['n_estimators_or_iterations'] for row in horizon_rows] != checkpoints:
            raise ValueError(f'Incomplete prefix curve: {folder}')
        all_rows.extend(horizon_rows)
        provenance['horizons'].append(dict(horizon=horizon, winner=winner['arm'],
                                           model_sha256=sha256(artifact_path),
                                           samples_sha256=sha256(samples_path),
                                           source_counts=[part[2] for part in sample],
                                           checkpoints=checkpoints))
    write_csv(directory / 'curvas_prefixo_vencedores.csv', all_rows)
    (directory / 'curvas_prefixo_procedencia.json').write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding='utf-8')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for horizon, ax in zip(('H0', 'H1'), axes):
        data = [row for row in all_rows if row['horizon'] == horizon]
        x = [row['n_estimators_or_iterations'] for row in data]
        ax.plot(x, [row['sampled_fit_rmse'] for row in data], 'o-', ms=3,
                label='Ajuste final (amostra)', color='#5785bd')
        ax.plot(x, [row['sampled_validation_rmse'] for row in data], 'o-', ms=3,
                label='Validação (amostra)', color='#d37454')
        ax.axvline(x[-1], color='#777777', lw=1, ls='--')
        ax.set_title(f"{horizon} · {data[0]['winner']}")
        ax.set_xlabel('Árvores acumuladas' if kind in ('enriched_random_forest',
                                                      'enriched_extra_trees') else 'Iterações')
        ax.set_ylabel('RMSE (unidades do target)')
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(f'{LABELS[kind]} · desempenho por prefixo do vencedor salvo', fontsize=14)
    fig.supxlabel('Reconstrução posterior em amostras fixas; não usada para escolher parâmetros, canais ou vencedor.',
                 fontsize=9)
    fig.savefig(directory / 'curvas_prefixo_vencedores.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)


def report_section(run_dir: Path, kind: str, config: dict, rows: list[dict],
                   status: dict) -> str:
    elapsed = (datetime.fromisoformat(status['finished'])-
               datetime.fromisoformat(status['started'])).total_seconds()/60
    selection = (datetime.fromisoformat(status['selection_finished'])-
                 datetime.fromisoformat(status['started'])).total_seconds()/60
    lines = [START, '## Auditoria complementar dos treinamentos', '',
             'Esta seção foi acrescentada depois da execução a partir da configuração, '
             'dos `metrics.json`, modelos e amostras arquivados. Nenhum modelo foi '
             'reajustado, nenhum vencedor foi alterado e nenhum target externo foi '
             'aberto para gerar os gráficos. As durações abaixo são da execução '
             'inteira e da fase até o congelamento dos vencedores; não foram '
             'registrados tempos individuais de todos os braços.', '',
             'O [manifesto dos diagnósticos](graficos_treinamento/diagnosticos_manifest.json) '
             'guarda hashes dos gráficos, tabelas e do '
             '[código usado](graficos_treinamento/audit_enriched_training_reports.py).', '',
             f'Início UTC: `{status["started"]}`; fim UTC: `{status["finished"]}`; '
             f'duração total {elapsed:.1f} min; seleção interna até os vencedores '
             f'{selection:.1f} min. Limite configurado: '
             f'{config["runtime"]["cpu_threads"]} threads; amostragem '
             f'original até {config["sampling"]["per_stage_year"]:,} pixels por '
             f'ano/etapa; semente espacial {config["split"]["seed"]}; semente '
             f'de amostragem {config["sampling"]["seed"]}.', '',
             '“Época” não é uma unidade comum a estes estimadores. Linear e '
             'Ridge resolvem o ajuste diretamente; HGB/XGBoost usam iterações de '
             'boosting; Random Forest e Extra Trees agregam árvores. A tabela mostra o orçamento '
             'efetivo de cada modelo final, com os parâmetros completos em '
             '[CSV auditável](graficos_treinamento/parametros_efetivos.csv).', '',
             '| Horizonte | Braço | Canais | Ajuste inicial | Seleção | Ajuste final | Validação | Orçamento/parâmetro variável | RMSE treino | RMSE validação |',
             '|---|---|---:|---:|---:|---:|---:|---|---:|---:|']
    for row in rows:
        lines.append(f"| {row['horizon']} | {row['arm']} | {row['n_features']} | "
                     f"{row['initial_fit_samples']:,} | {row['selection_samples']:,} | "
                     f"{row['final_fit_samples']:,} | {row['validation_samples']:,} | "
                     f"{row['budget']} | {row['refit_training_rmse']:.4f} | "
                     f"{row['validation_rmse']:.4f} |")
    if kind in ('enriched_hgb', 'enriched_xgboost', 'enriched_random_forest',
                'enriched_extra_trees'):
        cap_key = {'enriched_hgb': 'hgb', 'enriched_xgboost': 'xgboost',
                   'enriched_random_forest': 'random_forest',
                   'enriched_extra_trees': 'extra_trees'}[kind]
        lines += ['', f'Na busca/filtragem, o teto foi de '
                  f'{config[cap_key]["selection_samples_per_stage_year"]:,} '
                  'pixels por ano/etapa. As contagens reais por braço constam do CSV. '
                  'O ajuste final voltou a usar todas as amostras disponíveis nas '
                  'etapas `ajuste` e `selecao_camadas`.', '']
    lines += ['', '### Parâmetros configurados e efetivos', '',
              'A configuração congelada desta execução é [config.yaml](config.yaml). '
              'Os parâmetros específicos de cada braço, número real de iterações '
              'ou árvores, canais, tamanhos de amostra e SHA-256 do modelo constam '
              'em [parametros_efetivos.csv](graficos_treinamento/parametros_efetivos.csv). '
              'Abaixo está o bloco de configuração do estimador; valores de grade '
              'são candidatos, enquanto a tabela e o CSV mostram os escolhidos.', '',
              '```yaml']
    key = {'enriched_linear': None, 'enriched_ridge': 'ridge',
           'enriched_hgb': 'hgb', 'enriched_xgboost': 'xgboost',
           'enriched_random_forest': 'random_forest',
           'enriched_extra_trees': 'extra_trees'}[kind]
    if key is None:
        lines.append('pipeline: StandardScaler -> LinearRegression')
        lines.append('fit_intercept: true')
        lines.append('positive: false')
    else:
        lines.extend(yaml.safe_dump({key: config[key]}, allow_unicode=True,
                                    sort_keys=False).rstrip().splitlines())
    lines += ['```', '',
              '### Gráficos de treinamento e seleção', '',
              'As barras mostram RMSE ao fim do ajuste inicial/seleção, no treino '
              'do modelo reajustado e na validação. São métricas de conjuntos e '
              'ajustes diferentes: **não são uma curva por época**. A barra de '
              'treino mede as próprias amostras usadas no reajuste; a de '
              'validação mede amostras separadas.', '',
              '![RMSE por braço e etapa](graficos_treinamento/rmse_por_etapa.png)', '']
    if kind != 'enriched_linear':
        lines += ['A busca interna abaixo usa somente a etapa `selecao_camadas`. '
                  'Cada ponto representa o ajuste completo de um candidato, não '
                  'uma época do mesmo modelo.', '',
                  '![Busca interna](graficos_treinamento/busca_interna.png)', '']
    if kind in ('enriched_hgb', 'enriched_xgboost', 'enriched_random_forest',
                'enriched_extra_trees'):
        lines += ['As curvas seguintes foram **reconstruídas posteriormente** '
                  'dos modelos finais salvos: prefixos de iterações de boosting '
                  'ou árvores de floresta, sem novo treino. Mostram RMSE em '
                  'subamostras fixas das amostras de ajuste final e validação '
                  '(até 5.000 pixels por ano/etapa). Elas não são o histórico '
                  'registrado durante o treinamento, não usam o target externo '
                  'e não alteram a seleção já concluída. O CSV traz os pontos '
                  'exatos e o JSON registra índices, hashes e modelos usados.', '',
                  '![Curvas por prefixo dos vencedores](graficos_treinamento/curvas_prefixo_vencedores.png)', '',
                  '[Pontos das curvas](graficos_treinamento/curvas_prefixo_vencedores.csv) · '
                  '[Procedência da amostra](graficos_treinamento/curvas_prefixo_procedencia.json)', '']
    lines += [END, '']
    return '\n'.join(lines)


def enrich_run(run_dir: Path) -> Path:
    run_dir = run_dir.resolve()
    status = read_json(run_dir / 'run.json')
    kind = status.get('kind')
    if kind not in LABELS or status.get('status') != 'completed':
        raise ValueError(f'Expected one completed enriched run: {run_dir}')
    report_path = run_dir / 'RELATORIO.md'
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    config = yaml.safe_load((run_dir / 'config.yaml').read_text(encoding='utf-8'))
    rows = collect_rows(run_dir, kind, config)
    directory = run_dir / 'graficos_treinamento'
    directory.mkdir(exist_ok=True)
    write_csv(directory / 'parametros_efetivos.csv', rows)
    plot_stages(rows, directory / 'rmse_por_etapa.png', LABELS[kind])
    if kind != 'enriched_linear':
        plot_search(run_dir, kind, directory / 'busca_interna.png')
    if kind in ('enriched_hgb', 'enriched_xgboost', 'enriched_random_forest',
                'enriched_extra_trees'):
        reconstruct_curves(run_dir, kind, config, directory)
    section = report_section(run_dir, kind, config, rows, status)
    (run_dir / 'DETALHES_TREINAMENTO.md').write_text(section, encoding='utf-8')
    shutil.copy2(Path(__file__), directory / 'audit_enriched_training_reports.py')
    files = sorted(path for path in directory.iterdir() if path.is_file()
                   and path.name != 'diagnosticos_manifest.json')
    (directory / 'diagnosticos_manifest.json').write_text(
        json.dumps(dict(kind=kind, run=str(run_dir),
                        generated_utc=datetime.now(timezone.utc).isoformat(),
                        files={path.name: sha256(path) for path in files}),
                   ensure_ascii=False, indent=2), encoding='utf-8')
    original = report_path.read_text(encoding='utf-8')
    if START in original:
        original = original.split(START, 1)[0].rstrip()
    report_path.write_text(original.rstrip() + '\n\n' + section, encoding='utf-8')
    return report_path


def try_enrich_run(run_dir: Path) -> bool:
    """Preserve a completed model run even if a supplementary plot fails."""
    try:
        enrich_run(run_dir)
        return True
    except Exception:
        logging.exception('Training diagnostics failed for %s; model run remains complete', run_dir)
        return False


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, help='One completed run directory')
    args = parser.parse_args()
    if args.run:
        runs = [args.run]
    else:
        runs = sorted(path for family in (ROOT / 'runs').glob('enriched_*')
                      for path in family.iterdir()
                      if path.is_dir() and (path / 'run.json').is_file()
                      and read_json(path / 'run.json').get('status') == 'completed'
                      and (path / 'RELATORIO.md').is_file())
    for run_dir in runs:
        print(enrich_run(run_dir), flush=True)


if __name__ == '__main__':
    main()
