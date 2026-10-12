"""Post-protocol H1 2020->2021 evaluation of four frozen enriched models.

The H0 2021 records are read from completed runs; no model is fitted or selected.
"""
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
import csv
import json
import logging

from threadpoolctl import threadpool_limits

from plot_enriched_linear_external import render_one
from src.services import enriched_linear as common


ROOT = Path(__file__).resolve().parent
SOURCES = (
    ('Regressão linear', 'enriched_linear', '20260925T141835755712Z'),
    ('Ridge', 'enriched_ridge', '20260925T191737660876Z'),
    ('HistGradientBoosting', 'enriched_hgb', '20260925T213151313641Z'),
    ('XGBoost', 'enriched_xgboost', '20260926T150535837633Z'),
)


def source_specs() -> list[dict]:
    specs = []
    for label, kind, run_id in SOURCES:
        run_dir = ROOT / 'runs' / kind / run_id
        status = json.loads((run_dir / 'run.json').read_text(encoding='utf-8'))
        if status.get('status') != 'completed':
            raise ValueError(f'Source run is not complete: {run_dir}')
        h0 = json.loads((run_dir / 'horizon_0' / 'external_target_2021_metrics.json').read_text(encoding='utf-8'))
        winner = json.loads((run_dir / 'horizon_1' / 'winner.json').read_text(encoding='utf-8'))
        model = Path(winner['model_path']).resolve()
        if not model.is_relative_to(run_dir.resolve()) or not model.is_file():
            raise ValueError(f'Frozen model missing or outside its run: {model}')
        if h0['horizon'] != 0 or h0['input_year'] != 2021 or h0['target_year'] != 2021:
            raise ValueError(f'Unexpected H0 comparison record: {run_dir}')
        if winner['horizon'] != 1:
            raise ValueError(f'Unexpected H1 winner: {run_dir}')
        cfg = common.load_config(ROOT / 'configs' / f'{kind}.yaml')
        specs.append(dict(label=label, kind=kind, run_id=run_id,
                          run_dir=run_dir, model=model, winner=winner,
                          h0=h0, cfg=cfg))
    if len({(s['h0']['target_sha256'], s['h0']['common_pixels']) for s in specs}) != 1:
        raise ValueError('Original H0 target or common support differs across models')
    return specs


def _row(label: str, horizon: str, arm: str, n_features: int, record: dict) -> dict:
    m = record['metrics']
    return dict(model=label, horizon=horizon, input_year=record['input_year'],
                target_year=record['target_year'], arm=arm, n_features=n_features,
                common_pixels=record['common_pixels'], positive_fraction=1-m['target_zero']/m['n'],
                rmse=m['rmse'], mae=m['mae'], r2=m['r2'], bias=m['bias'],
                capture_10pct=m['capture_10pct'],
                zero_rmse=record['reference_zero']['rmse'],
                zero_mae=record['reference_zero']['mae'],
                persistence_rmse=(record['reference_persistence'] or {}).get('metrics', {}).get('rmse'),
                persistence_mae=(record['reference_persistence'] or {}).get('metrics', {}).get('mae'))


def _report(output: Path, records: list[dict], rows: list[dict]) -> None:
    h0 = [r for r in rows if r['horizon'] == 'H0']
    h1 = [r for r in rows if r['horizon'] == 'H1']
    lines = [
        '# Foco no target externo de 2021 — H0 e H1', '',
        'Análise **exploratória posterior ao protocolo** dos experimentos 01–04. '
        'Nenhum modelo foi treinado, filtrado ou escolhido novamente. H0 2021→2021 vem '
        'dos testes finais já concluídos; H1 2020→2021 foi aplicado agora aos vencedores H1 '
        'congelados. Os testes finais originais, inclusive H1 2021→2022, permanecem válidos e inalterados.', '',
        f"Target externo 2021: SHA-256 `{records[0]['h1']['target_sha256']}`. "
        'A leitura focal de 2021 foi escolhida depois de conhecer a distribuição do target; '
        'os números abaixo não são uma nova validação independente para selecionar modelos.', '',
    ]
    for horizon, group in (('H0 · entrada 2021 → target 2021', h0),
                           ('H1 · entrada 2020 → target 2021', h1)):
        lines += [f'## {horizon}', '',
                  '| Modelo congelado | Braço | Canais | Pixels | RMSE | MAE | R² | Captura top 10% |',
                  '|---|---|---:|---:|---:|---:|---:|---:|']
        for row in group:
            lines.append(f"| {row['model']} | {row['arm']} | {row['n_features']} | "
                         f"{row['common_pixels']:,} | {row['rmse']:.4f} | {row['mae']:.4f} | "
                         f"{row['r2']:.4f} | {100*row['capture_10pct']:.2f}% |")
        lines += ['', '| Referência | RMSE | MAE |', '|---|---:|---:|',
                  f"| Previsão zero | {group[0]['zero_rmse']:.4f} | {group[0]['zero_mae']:.4f} |",
                  f"| Persistência do target 2020 | {group[0]['persistence_rmse']:.4f} | {group[0]['persistence_mae']:.4f} |", '']
    lines += [
        '## Leitura dos resultados', '',
        'O target de 2021 tem 50,08% de pixels positivos na máscara comum, mas as intensidades '
        'continuam concentradas: a mediana entre positivos é 0,6472 e os 10% de pixels com '
        'maior target contêm 65,51% da massa. Equilíbrio de presença não implica equilíbrio '
        'de intensidade.', '',
        'Ridge tem o menor RMSE entre os modelos de camadas em H0 (3,8529) e H1 (4,0604). '
        'XGBoost tem o menor MAE e a maior captura top 10% em H0. No novo H1, '
        'HistGradientBoosting tem o menor MAE (1,8424) e a maior captura top 10% (23,57%), '
        'mas RMSE de 5,2538. Nenhum dos quatro modelos supera a previsão zero ou a '
        'persistência em RMSE e MAE. Todos têm R² negativo.', '',
        'A captura top 10% em H1 foi maior que em H0 para os quatro modelos, mas H0 e H1 '
        'usam entradas de anos diferentes e modelos selecionados separadamente; essa '
        'diferença observada não estabelece vantagem causal do horizonte H1.', '',
        'Persistência usa o target de 2020 como informação adicional e não possui as mesmas '
        'entradas dos modelos de camadas. A captura é a fração da massa do target nos 10% '
        'de pixels com maior previsão, não uma probabilidade ou contagem de voos. O target zero '
        'denota ausência de registro na fotografia considerada.', '',
        '## Mapas adicionais H1 — target original e previsão', '',
        'Cada PNG tem 9.000 × 3.600 pixels (600 dpi), com mesma máscara e escala de cores dentro do par. '
        'Os GeoTIFFs preservam a resolução original.', '',
    ]
    for item in records:
        name = item['label']
        image = Path(item['figure']['path']).relative_to(output).as_posix()
        tif = Path(item['h1']['prediction']).relative_to(output).as_posix()
        lines += [f'**{name}** · [GeoTIFF de previsão]({tif})', '',
                  f'![Target original e previsão H1 2020→2021, {name}]({image})', '']
    lines += [
        '## Procedência e limites', '',
        'O arquivo `comparacao_2021.csv` contém as métricas completas de H0 e H1. '
        'Cada subpasta contém o JSON de métricas H1, o GeoTIFF e a figura. '
        '`procedencia.json` identifica os runs, modelos congelados e hashes dos modelos. '
        'Não inferir contribuição causal das camadas nem tratar esta avaliação adicional como '
        'uma segunda região independente.', '',
    ]
    (output / 'RELATORIO.md').write_text('\n'.join(lines), encoding='utf-8')


def run() -> Path:
    specs = source_specs()
    output_root = ROOT / 'reports' / 'target_2021_frozen_models'
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(exist_ok=False)
    status = dict(status='running', started=datetime.now(timezone.utc).isoformat(),
                  kind='post_protocol_target_2021_h0_h1', no_training=True,
                  source_runs=[s['run_id'] for s in specs])
    common._write_json(output / 'run.json', status)
    handler = logging.FileHandler(output / 'evaluation.log', encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    try:
        records, rows, provenance = [], [], []
        for spec in specs:
            logging.info('H1 2020→2021: evaluating frozen %s winner from %s', spec['label'], spec['run_id'])
            folder = output / spec['kind']
            folder.mkdir()
            cfg = spec['cfg']
            cfg['raw']['evaluation']['final_h1_target_years'] = [2021]
            with threadpool_limits(limits=cfg['raw']['runtime']['cpu_threads']):
                h1 = common.evaluate_external(cfg, spec['winner'], 1, folder)[0]
            if h1['input_year'] != 2020 or h1['target_year'] != 2021:
                raise ValueError('Unexpected H1 temporal pair')
            if h1['target_sha256'] != spec['h0']['target_sha256']:
                raise ValueError('H0/H1 target raster differs')
            figure = render_one(h1, folder)
            records.append(dict(label=spec['label'], kind=spec['kind'],
                                h0=spec['h0'], h1=h1, figure=figure))
            h0_winner = json.loads((spec['run_dir'] / 'horizon_0' / 'winner.json').read_text(encoding='utf-8'))
            rows.extend((_row(spec['label'], 'H0', h0_winner['arm'], h0_winner['n_features'], spec['h0']),
                         _row(spec['label'], 'H1', spec['winner']['arm'], spec['winner']['n_features'], h1)))
            provenance.append(dict(model=spec['label'], source_run=str(spec['run_dir']),
                                   h1_model=str(spec['model']), h1_model_sha256=common._sha256(spec['model']),
                                   h0_record=str(spec['run_dir'] / 'horizon_0' / 'external_target_2021_metrics.json'),
                                   config=str(cfg['path'])))
            common._write_json(output / 'procedencia.json', provenance)
        if len({(r['common_pixels'], round(r['positive_fraction'], 12)) for r in rows}) != 1:
            logging.warning('H0 and H1 use different common pixel masks; compare sample sizes explicitly')
        with (output / 'comparacao_2021.csv').open('w', newline='', encoding='utf-8-sig') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        _report(output, records, rows)
        status.update(status='completed', finished=datetime.now(timezone.utc).isoformat())
        common._write_json(output / 'run.json', status)
        return output
    except KeyboardInterrupt:
        status.update(status='interrupted', interrupted=datetime.now(timezone.utc).isoformat())
        common._write_json(output / 'run.json', status)
        raise
    except Exception as exc:
        status.update(status='failed', failed=datetime.now(timezone.utc).isoformat(), error=repr(exc))
        common._write_json(output / 'run.json', status)
        logging.exception('Complementary 2021 evaluation failed; partial files kept in %s', output)
        raise
    finally:
        logger.removeHandler(handler)
        handler.close()


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true', help='Verify the four completed source runs without inference')
    args = parser.parse_args()
    if args.check_only:
        specs = source_specs()
        print(f'PRECHECK OK: {len(specs)} frozen H1 models and matching H0 2021 records.')
    else:
        output = run()
        print(output / 'RELATORIO.md')


if __name__ == '__main__':
    main()
