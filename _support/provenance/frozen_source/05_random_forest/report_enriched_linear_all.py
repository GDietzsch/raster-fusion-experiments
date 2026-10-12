"""Export every trained arm and evaluated feature-selection candidate from one run."""
from argparse import ArgumentParser
import csv
import json
from pathlib import Path


ARMS = ('originais', 'originais_filtradas', 'gpa', 'grafos', 'todas', 'todas_filtradas')
PHENOMENA = ('Garimpos', 'Ocorrencias_Minerais', 'Pistas_Regulares', 'Pistas_Irregulares')
PRODUCTS = ('original', 'magnitude', 'direcao_seno', 'direcao_cosseno', 'conectividade', 'suporte')
SHORT = {'Garimpos': 'G', 'Ocorrencias_Minerais': 'M',
         'Pistas_Regulares': 'R', 'Pistas_Irregulares': 'I'}


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f'No rows for {path}')
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value, digits=4):
    return '—' if value is None else f'{float(value):.{digits}f}'


def label(features):
    return ', '.join(f'{SHORT[name]}:{product}' for name, product in
                     (feature.split('__', 1) for feature in features))


def flat_metrics(row, metrics, stage):
    for key in ('n', 'rmse', 'mae', 'r2', 'bias', 'median_error',
                'p90_absolute_error', 'nmad', 'negative_predictions',
                'target_zero', 'capture_5pct', 'capture_10pct', 'capture_20pct'):
        row[f'{stage}_{key}'] = metrics.get(key)
    for subset in ('zero', 'positive', 'high'):
        sub = metrics.get('subsets', {}).get(subset) or {}
        for key in ('n', 'rmse', 'mae', 'r2', 'bias', 'nmad'):
            row[f'{stage}_{subset}_{key}'] = sub.get(key)


def run_report(run, output):
    run = run.resolve()
    output = output.resolve()
    if not output.is_dir():
        raise FileNotFoundError(output)
    status = json.loads((run / 'run.json').read_text(encoding='utf-8'))
    if status.get('status') != 'completed':
        raise ValueError('Run not completed')

    arm_rows, year_rows, original_rows, removal_rows, inventory_rows = [], [], [], [], []
    tables = {}
    for horizon in (0, 1):
        folder = run / f'horizon_{horizon}'
        comparison = read_csv(folder / 'comparison.csv')
        if len(comparison) != len(ARMS) or tuple(row['arm'] for row in comparison) != ARMS:
            raise ValueError(f'Unexpected arms in H{horizon} comparison')
        metrics_by_arm = {arm: json.loads((folder / arm / 'metrics.json').read_text(encoding='utf-8'))
                          for arm in ARMS}
        all_features = metrics_by_arm['todas']['feature_names']
        if len(all_features) != 24:
            raise ValueError(f'Expected 24 features in H{horizon}')
        table = []
        for comparison_row in comparison:
            arm = comparison_row['arm']
            metric = metrics_by_arm[arm]
            if int(comparison_row['n_features']) != len(metric['feature_names']):
                raise ValueError(f'Feature count differs for H{horizon} {arm}')
            validation = metric['validation_metrics']
            if abs(float(comparison_row['validation_rmse']) - validation['rmse']) > 1e-9:
                raise ValueError(f'Validation RMSE differs for H{horizon} {arm}')
            row = {'horizon': f'H{horizon}', 'arm': arm,
                   'n_features': metric['n_features'],
                   'feature_names': ';'.join(metric['feature_names']),
                   'selected_winner': arm == status['winners'][str(horizon)]}
            for stage, key in (('selection', 'initial_selection_metrics'),
                               ('refit_training', 'training_metrics'),
                               ('validation', 'validation_metrics')):
                flat_metrics(row, metric[key], stage)
            arm_rows.append(row)
            table.append(row)
            for year in metric['validation_by_year']:
                year_row = {'horizon': f'H{horizon}', 'arm': arm,
                            'target_year': year['target_year']}
                flat_metrics(year_row, year['metrics'], 'validation')
                year_rows.append(year_row)
        tables[horizon] = table

        for feature in all_features:
            name, product = feature.split('__', 1)
            if name not in PHENOMENA or product not in PRODUCTS:
                raise ValueError(f'Unexpected feature {feature}')
            inventory_rows.append({
                'horizon': f'H{horizon}', 'feature': feature,
                'originais_filtradas': feature in metrics_by_arm['originais_filtradas']['feature_names'],
                'todas_filtradas': feature in metrics_by_arm['todas_filtradas']['feature_names'],
                'gpa': feature in metrics_by_arm['gpa']['feature_names'],
                'grafos': feature in metrics_by_arm['grafos']['feature_names']})

        candidates = read_csv(folder / 'selection_candidates.csv')
        originals = [row for row in candidates if row['arm'] == 'originais_filtradas']
        removals = [row for row in candidates if row['arm'] == 'todas_filtradas']
        if len(originals) != 15 or not removals:
            raise ValueError(f'Incomplete selection log H{horizon}')
        best_original = min(originals, key=lambda row: (float(row['selection_rmse']),
                                                         float(row['selection_mae']),
                                                         int(row['n_features'])))
        selected_original = metrics_by_arm['originais_filtradas']['feature_names']
        if best_original['features'].split(';') != selected_original:
            raise ValueError(f'Original selection differs H{horizon}')
        for row in originals:
            original_rows.append({'horizon': f'H{horizon}',
                                  'combination': label(row['features'].split(';')),
                                  'features': row['features'],
                                  'n_features': int(row['n_features']),
                                  'selection_rmse': float(row['selection_rmse']),
                                  'selection_mae': float(row['selection_mae']),
                                  'selected_for_refit': row is best_original})

        baseline = removals[0]
        if baseline['step'] != '0' or baseline['features'].split(';') != all_features:
            raise ValueError(f'Unexpected elimination baseline H{horizon}')
        base = all_features
        for step in range(0, max(int(row['step']) for row in removals) + 1):
            this_step = [row for row in removals if int(row['step']) == step]
            if step and len(this_step) != 17-step:
                raise ValueError(f'Incomplete elimination step {step} H{horizon}')
            accepted = [row for row in this_step if row['accepted'] == 'True']
            if len(accepted) > 1 or (step == 0 and len(accepted) != 1):
                raise ValueError(f'Unexpected accepted candidates step {step} H{horizon}')
            for row in this_step:
                remaining = row['features'].split(';')
                removed = [] if step == 0 else [feature for feature in base if feature not in remaining]
                if step and len(removed) not in (1, 2):
                    raise ValueError(f'Invalid removal H{horizon} step {step}')
                removal_rows.append({'horizon': f'H{horizon}', 'step': step,
                                     'removed_group': 'baseline' if step == 0 else label(removed),
                                     'removed_features': ';'.join(removed),
                                     'remaining_features': ';'.join(remaining),
                                     'n_features': int(row['n_features']),
                                     'selection_rmse': float(row['selection_rmse']),
                                     'selection_mae': float(row['selection_mae']),
                                     'removal_accepted': row['accepted'] == 'True'})
            if accepted:
                base = accepted[0]['features'].split(';')
        if base != metrics_by_arm['todas_filtradas']['feature_names']:
            raise ValueError(f'Final selected features differ H{horizon}')

    external_rows, external_reference_rows, external_subset_rows, external_feature_rows = [], [], [], []
    external_sources = []
    for horizon in (0, 1):
        folder = run / f'horizon_{horizon}'
        winner = json.loads((folder / 'winner.json').read_text(encoding='utf-8'))
        if winner['arm'] != status['winners'][str(horizon)] or winner['n_features'] != len(winner['feature_names']):
            raise ValueError(f'Winner information differs for H{horizon}')
        for feature in winner['feature_names']:
            external_feature_rows.append({'horizon': f'H{horizon}', 'winner': winner['arm'],
                                          'feature': feature, 'phenomenon': feature.split('__', 1)[0],
                                          'product': feature.split('__', 1)[1]})
        for path in sorted(folder.glob('external_target_*_metrics.json')):
            record = json.loads(path.read_text(encoding='utf-8'))
            if record['horizon'] != horizon or record['metrics']['n'] != record['common_pixels']:
                raise ValueError(f'External support differs in {path}')
            metrics = record['metrics']
            row = {'horizon': f'H{horizon}', 'winner': winner['arm'],
                   'n_features': winner['n_features'], 'input_year': record['input_year'],
                   'target_year': record['target_year'], 'common_pixels': record['common_pixels'],
                   'evaluated_fraction': record['evaluated_fraction'],
                   'prediction': record['prediction'], 'target_source': record['target_source'],
                   'high_target_threshold': winner['high_target_threshold'],
                   'rmse_ci95_low': record['uncertainty']['rmse_ci95'][0],
                   'rmse_ci95_high': record['uncertainty']['rmse_ci95'][1],
                   'mae_ci95_low': record['uncertainty']['mae_ci95'][0],
                   'mae_ci95_high': record['uncertainty']['mae_ci95'][1]}
            flat_metrics(row, metrics, 'model')
            external_rows.append(row)
            external_sources.append(record)
            for subset_name in ('zero', 'positive', 'high'):
                subset = metrics['subsets'][subset_name]
                subset_row = {'horizon': f'H{horizon}', 'input_year': record['input_year'],
                              'target_year': record['target_year'], 'subset': subset_name,
                              'n': None if subset is None else subset['n'],
                              'rmse': None if subset is None else subset['rmse'],
                              'mae': None if subset is None else subset['mae'],
                              'r2': None if subset is None else subset['r2'],
                              'bias': None if subset is None else subset['bias'],
                              'nmad': None if subset is None else subset['nmad']}
                external_subset_rows.append(subset_row)
            for reference_name, source in (('zero', record['reference_zero']),
                                           ('training_mean', record['reference_training_mean']),
                                           ('persistence', record.get('reference_persistence'))):
                if source is None:
                    continue
                reference = source['metrics'] if reference_name == 'persistence' else source
                external_reference_rows.append({'horizon': f'H{horizon}',
                                                'input_year': record['input_year'],
                                                'target_year': record['target_year'],
                                                'reference': reference_name,
                                                'n': reference['n'], 'rmse': reference['rmse'],
                                                'mae': reference['mae'], 'r2': reference['r2'],
                                                'bias': reference['bias'],
                                                'capture_10pct': reference['capture_10pct']})
    expected_features = sum(int(json.loads((run / f'horizon_{horizon}' / 'winner.json').read_text(encoding='utf-8'))['n_features'])
                            for horizon in (0, 1))
    if len(external_rows) != 4 or len(external_feature_rows) != expected_features:
        raise ValueError('External evaluations or winner features are incomplete')

    write_csv(output / 'resultados_12_treinamentos.csv', arm_rows)
    write_csv(output / 'validacao_por_ano.csv', year_rows)
    write_csv(output / '30_combinacoes_originais.csv', original_rows)
    removal_file = f'{len(removal_rows)}_avaliacoes_filtragem.csv'
    write_csv(output / removal_file, removal_rows)
    write_csv(output / 'inventario_48_canais.csv', inventory_rows)
    write_csv(output / 'teste_externo_metricas.csv', external_rows)
    write_csv(output / 'teste_externo_referencias.csv', external_reference_rows)
    write_csv(output / 'teste_externo_subconjuntos.csv', external_subset_rows)
    write_csv(output / 'teste_externo_canais.csv', external_feature_rows)

    kind = status.get('kind')
    title = {'enriched_linear': 'experimento 01, regressão linear',
             'enriched_ridge': 'experimento 02, regressão Ridge',
             'enriched_hgb': 'experimento 03, HistGradientBoosting',
             'enriched_xgboost': 'experimento 04, XGBoost',
             'enriched_random_forest': 'experimento 05, Random Forest'}.get(kind)
    if title is None:
        raise ValueError(f'Unsupported experiment kind: {kind}')
    lines = [
        f'# Todos os resultados — {title}', '',
        f'Fonte: `{run}`. Este documento organiza os artefatos já produzidos pela execução concluída; '
        'não realiza treino nem aplica candidatos à área de teste final.', '',
        '## Como ler', '',
        'H0 usa entradas e target do mesmo ano; H1 usa entradas do ano anterior para prever o target seguinte. '
        'A seleção de camadas usa RMSE da etapa de seleção. Os seis conjuntos passam por novo ajuste e são comparados '
        'na validação; só o vencedor de cada horizonte recebeu teste externo. '
        'As linhas de candidatos abaixo representam modelos provisórios avaliados na seleção; **não** receberam '
        'novo ajuste, validação final ou teste externo individual.', '',
        'G = Garimpos; M = Ocorrências minerais; R = Pistas regulares; I = Pistas irregulares. '
        'Em GPA, cada fenômeno tem magnitude e direção representada por seno e cosseno. '
        'Em grafos, cada fenômeno tem conectividade e suporte.', '',
        '## Todos os conjuntos treinados e validados', '',
    ]
    for horizon in (0, 1):
        lines += [f'### H{horizon}', '',
                  '| Conjunto | Canais | RMSE seleção | RMSE treino após novo ajuste | RMSE validação | MAE validação | R² validação | NMAD validação | Captura 10% validação |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for row in tables[horizon]:
            lines.append(f"| {row['arm']}{' **(vencedor)**' if row['selected_winner'] else ''} | "
                         f"{row['n_features']} | {fmt(row['selection_rmse'])} | "
                         f"{fmt(row['refit_training_rmse'])} | {fmt(row['validation_rmse'])} | "
                         f"{fmt(row['validation_mae'])} | {fmt(row['validation_r2'])} | "
                         f"{fmt(row['validation_nmad'])} | {100*row['validation_capture_10pct']:.2f}% |")
        lines += ['', 'O conjunto **gpa** contém os 12 canais GPA (magnitude, seno e cosseno da direção para cada '
                  'um dos quatro fenômenos). Não houve filtro dentro do braço GPA.', '']

    lines += ['## Teste na área externa — configuração e resultados', '',
              'Esta seção **não** é a validação usada para escolher modelos. A área externa foi avaliada '
              'depois da seleção. H0 usa inputs e target do mesmo ano (2020, 2021 e 2022); H1 usa inputs '
              'de 2021 para o target de 2022. São cenários operacionais diferentes. Os modelos não receberam '
              'o target externo como entrada nem foram reajustados após esse teste.', '',
              '| Horizonte | Conjunto aplicado | Nº canais | Fenômeno | Canais efetivamente usados |',
              '|---|---|---:|---|---|']
    for horizon in (0, 1):
        subset = [item for item in external_feature_rows if item['horizon'] == f'H{horizon}']
        for phenomenon in PHENOMENA:
            products = [item['product'] for item in subset if item['phenomenon'] == phenomenon]
            lines.append(f"| H{horizon} | {status['winners'][str(horizon)]} | {len(subset)} | "
                         f"{phenomenon.replace('_', ' ')} | {', '.join(products)} |")
    lines += ['', 'A tabela identifica os canais efetivamente usados por cada vencedor. '
              '`teste_externo_canais.csv` contém cada nome de canal completo, com o total por horizonte.', '',
              '### Métricas de intensidade', '',
              'Erro e target estão nas unidades originais do raster. Viés = previsão − target; valores positivos '
              'indicam superestimação média. P90 é o percentil 90 do erro absoluto. Nenhuma previsão negativa '
              f"foi truncada. As quatro avaliações usam {external_rows[0]['common_pixels']:,} "
              f"pixels comuns ({100*external_rows[0]['evaluated_fraction']:.2f}% da grade) "
              'nesta execução.', '',
              '| Modelo | Input → target | RMSE | MAE | R² | Viés | NMAD | P90 erro abs. | Previsões negativas |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in external_rows:
        lines.append(f"| {row['horizon']} {row['winner']} | {row['input_year']} → {row['target_year']} | "
                     f"{fmt(row['model_rmse'])} | {fmt(row['model_mae'])} | {fmt(row['model_r2'])} | "
                     f"{fmt(row['model_bias'])} | {fmt(row['model_nmad'])} | "
                     f"{fmt(row['model_p90_absolute_error'])} | "
                     f"{100*row['model_negative_predictions']/row['common_pixels']:.2f}% |")
    lines += ['', '### Priorização espacial', '',
              'Captura é a fração da **massa do target** que está nos pixels com maior previsão; não é '
              'probabilidade, acurácia de classificação nem contagem de voos.', '',
              '| Modelo | Input → target | Captura top 5% | Captura top 10% | Captura top 20% |',
              '|---|---|---:|---:|---:|']
    for row in external_rows:
        lines.append(f"| {row['horizon']} {row['winner']} | {row['input_year']} → {row['target_year']} | "
                     f"{100*row['model_capture_5pct']:.2f}% | "
                     f"{100*row['model_capture_10pct']:.2f}% | "
                     f"{100*row['model_capture_20pct']:.2f}% |")
    thresholds = {h: json.loads((run / f'horizon_{h}' / 'winner.json').read_text(encoding='utf-8'))['high_target_threshold']
                  for h in (0, 1)}
    high_empty = all(row['n'] is None for row in external_subset_rows if row['subset'] == 'high')
    high_note = ('O grupo de alta intensidade ficou vazio na área externa para os limiares '
                 if high_empty else 'O grupo de alta intensidade usa os limiares ')
    lines += ['', '### Erro por tipo de pixel', '',
              'Target zero quer dizer sem registro na fotografia considerada; não significa impossibilidade '
              f'de ocorrência. {high_note}definidos no treino '
              f"(H0: {thresholds[0]:.4f}; H1: {thresholds[1]:.4f}); os limiares não foram alterados após o teste.", '',
              '| Modelo | Target | Pixels | RMSE | MAE | R² |', '|---|---|---:|---:|---:|---:|']
    for row in external_subset_rows:
        if row['n'] is None:
            continue
        label_subset = {'zero': 'zero', 'positive': 'positivo', 'high': 'alta intensidade'}[row['subset']]
        lines.append(f"| {row['horizon']} {row['input_year']} → {row['target_year']} | "
                     f"{label_subset} | {row['n']:,} | {fmt(row['rmse'])} | "
                     f"{fmt(row['mae'])} | {fmt(row['r2'])} |")
    lines += ['', 'R² é indefinido no subconjunto de target zero, pois o target é constante ali.', '',
              '### Referências na mesma área', '',
              'Zero prevê intensidade zero em todos os pixels; média usa somente a média do target de treino '
              'do respectivo horizonte. Persistência usa o target do ano anterior e, portanto, dispõe de '
              'informação adicional. Não há persistência para H0 2020 nesta execução.', '',
              '| Horizonte | Input → target | Método | RMSE | MAE | R² |',
              '|---|---|---|---:|---:|---:|']
    for row in external_rows:
        lines.append(f"| {row['horizon']} | {row['input_year']} → {row['target_year']} | "
                     f"modelo | {fmt(row['model_rmse'])} | {fmt(row['model_mae'])} | {fmt(row['model_r2'])} |")
        for ref in (item for item in external_reference_rows
                    if item['horizon'] == row['horizon'] and item['target_year'] == row['target_year']):
            lines.append(f"| {ref['horizon']} | {ref['input_year']} → {ref['target_year']} | "
                         f"{ref['reference']} | {fmt(ref['rmse'])} | {fmt(ref['mae'])} | {fmt(ref['r2'])} |")
    uncertainty = external_sources[0]['uncertainty']
    lines += ['', '### Incerteza e arquivos de previsão', '',
              f"Os intervalos abaixo foram calculados pelo experimento com {uncertainty['replicates']:,} "
              f"reamostragens de {uncertainty['occupied_blocks']} blocos ocupados de aproximadamente "
              f"{uncertainty['approximate_block_km']:.0f} km. São exploratórios e dependem da escala dos blocos; "
              'não equivalem a replicar o teste em novas regiões.', '',
              '| Modelo | Input → target | IC95% exploratório RMSE | IC95% exploratório MAE |',
              '|---|---|---|---|']
    for row in external_rows:
        lines.append(f"| {row['horizon']} {row['winner']} | {row['input_year']} → {row['target_year']} | "
                     f"[{fmt(row['rmse_ci95_low'])}; {fmt(row['rmse_ci95_high'])}] | "
                     f"[{fmt(row['mae_ci95_low'])}; {fmt(row['mae_ci95_high'])}] |")
    lines += ['', 'GeoTIFFs de previsão, no grid da área externa:', '']
    for row in external_rows:
        lines.append(f"- {row['horizon']} {row['input_year']} → {row['target_year']}: `{row['prediction']}`")
    lines += ['', '### Imagens do teste externo: target original e previsão', '',
              'Cada par usa a mesma máscara e a mesma escala cromática simétrica/logarítmica. '
              'A escala é recalculada entre anos; cores de imagens diferentes não são comparação numérica direta. '
              'Azul identifica valores previstos negativos, preservados sem truncamento. '
              'Os PNGs têm 9.000 × 3.600 pixels, 600 dpi; apenas a exibição foi reduzida 2×, '
              'e os GeoTIFFs acima mantêm a resolução original.', '']
    for row in external_rows:
        name = f"teste_externo_{row['horizon']}_{row['input_year']}_{row['target_year']}.png"
        if not (output / name).is_file():
            raise FileNotFoundError(f'External comparison figure is missing: {output / name}')
        lines += [f"**{row['horizon']} · entrada {row['input_year']} → target {row['target_year']}**", '',
                  f"![Target original e previsão, {row['horizon']} {row['target_year']}]({name})", '']
    lines += ['', 'Os CSVs `teste_externo_metricas.csv`, `teste_externo_referencias.csv`, '
              '`teste_externo_subconjuntos.csv` e `teste_externo_canais.csv` preservam os valores completos '
              'e os nomes exatos dos atributos. Por horizonte, somente o respectivo vencedor foi testado '
              'externamente; os demais braços não têm métricas nessa área. Aplicá-los agora à mesma área '
              'seria uma comparação exploratória '
              'pós-seleção, distinta do teste final protocolado.', '']

    lines += ['## Canais que permaneceram após a filtragem', '',
              'A filtragem é diferente em H0 e H1. Um canal removido continua disponível nos outros conjuntos; '
              '“removido” refere-se somente ao respectivo braço filtrado.', '',
              '| Horizonte e braço | Permaneceram | Foram removidos |', '|---|---|---|']
    for horizon in (0, 1):
        folder = run / f'horizon_{horizon}'
        all_features = json.loads((folder / 'todas' / 'metrics.json').read_text(encoding='utf-8'))['feature_names']
        for arm, universe in (('originais_filtradas', [x for x in all_features if x.endswith('__original')]),
                              ('todas_filtradas', all_features)):
            kept = json.loads((folder / arm / 'metrics.json').read_text(encoding='utf-8'))['feature_names']
            removed = [x for x in universe if x not in kept]
            lines.append(f'| H{horizon} {arm} | {label(kept)} | {label(removed)} |')
    lines += ['', 'A remoção de um grupo de direção exclui seno e cosseno juntos; a de um grupo de grafo '
              'exclui conectividade e suporte juntos. O inventário completo de presença por canal está em '
              '`inventario_48_canais.csv`.', '',
              '## As 15 combinações de originais avaliadas em cada horizonte', '',
              'Cada linha abaixo foi avaliada apenas na etapa de seleção. O vencedor dessa etapa foi reajustado '
              'como `originais_filtradas` e seu resultado de validação aparece na tabela principal.', '']
    for horizon in (0, 1):
        lines += [f'### H{horizon}', '', '| Combinação | Nº canais | RMSE seleção | MAE seleção | Escolhida para novo ajuste |',
                  '|---|---:|---:|---:|---|']
        for row in [item for item in original_rows if item['horizon'] == f'H{horizon}']:
            lines.append(f"| {row['combination']} | {row['n_features']} | {fmt(row['selection_rmse'])} | "
                         f"{fmt(row['selection_mae'])} | {'sim' if row['selected_for_refit'] else 'não'} |")
        lines.append('')

    lines += ['## Todas as tentativas da eliminação no conjunto completo', '',
              'Em cada passo, cada grupo ainda presente foi retirado temporariamente e o RMSE de seleção foi '
              'medido. `sim` marca a retirada aceita. A linha de passo zero é a referência com os 24 canais. '
              'A busca para quando nenhuma retirada restante melhora o RMSE; não é uma busca exaustiva de todos '
              'os subconjuntos dos 24 canais.', '']
    for horizon in (0, 1):
        subset = [item for item in removal_rows if item['horizon'] == f'H{horizon}']
        lines += [f'### H{horizon} — {len(subset)} avaliações', '',
                  '| Passo | Grupo retirado para avaliação | Canais restantes | RMSE seleção | MAE seleção | Retirada aceita |',
                  '|---:|---|---:|---:|---:|---|']
        for row in subset:
            lines.append(f"| {row['step']} | {row['removed_group']} | {row['n_features']} | "
                         f"{fmt(row['selection_rmse'])} | {fmt(row['selection_mae'])} | "
                         f"{'sim' if row['removal_accepted'] else 'não'} |")
        lines.append('')
    lines += ['## Arquivos completos', '',
              'Os cinco CSVs das etapas internas preservam os números sem arredondamento: '
              '`resultados_12_treinamentos.csv`, `validacao_por_ano.csv`, `30_combinacoes_originais.csv`, '
              f'`{removal_file}` e `inventario_48_canais.csv`. '
              'Cada `horizon_*/<conjunto>/metrics.json` da execução contém ainda as métricas por target zero, '
              'positivo e alta intensidade, captura 5/10/20%, viés e percentil 90 do erro absoluto.', '']
    appendix = run / 'methodology_appendix.md'
    if appendix.is_file():
        lines += [appendix.read_text(encoding='utf-8'), '']
    (output / 'TODOS_RESULTADOS.md').write_text('\n'.join(lines), encoding='utf-8')
    print(output / 'TODOS_RESULTADOS.md')


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run_report(args.run, args.output)


if __name__ == '__main__':
    main()
