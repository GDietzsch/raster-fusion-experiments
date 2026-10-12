"""Verify saved fits, compute paired summaries, and write manuscript-ready text."""
from pathlib import Path
import json
import sys
import hashlib
import csv
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

OUT = Path(__file__).resolve().parent
ROOT = (Path(__file__).resolve().parents[2] / 'raster_fusion_ai')
sys.path.insert(0, str(ROOT))
from src.services import enriched_linear as common


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def digest(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def score(y, p):
    e = p.astype(np.float64)-y
    ss = float(np.sum((y-y.mean())**2))
    return dict(n=len(y), rmse=float(np.sqrt(np.mean(e**2))), mae=float(np.mean(np.abs(e))),
                r2=float(1-np.sum(e**2)/ss) if ss else None)


def table(rows, keys, labels):
    text = '| ' + ' | '.join(labels) + ' |\n| ' + ' | '.join(['---']*len(keys)) + ' |\n'
    for row in rows:
        vals = [f'{row[k]:.4f}' if isinstance(row[k], (float, np.floating)) else str(row[k]) for k in keys]
        text += '| ' + ' | '.join(vals) + ' |\n'
    return text


def main():
    protocol = read(OUT/'protocol.json')
    assert digest(OUT/'protocol.json') == read(OUT/'prepared.json')['protocol_sha256']
    assert read(OUT/'status.json')['status'] in ('fits_completed', 'completed')
    rows, yearly, contrasts, distributions, checks, separations = [], [], [], [], [], []
    pooled, figures = {}, []
    for spec in protocol['models']:
        h = spec['horizon']
        assert digest(Path(spec['source_model'])) == spec['source_model_sha256']
        source = ROOT/f'runs/enriched_cnn/20260926T223539611207Z/horizon_{h}'
        with np.load(source/'samples.npz') as z:
            samples = {k: z[k] for k in z.files}
        vy = samples['validacao_modelo_y'].astype(np.float64)
        vp = samples['validacao_modelo_positions']
        years = samples['validacao_modelo_target_year']
        combined = {key: np.full(len(vy), np.nan) for key in ('original', '750', '41000', 'mean_750', 'mean_41000')}
        assessed = []
        for fold in protocol['folds']:
            folder = OUT/f'H{h}_{fold}'
            assert digest(folder/'indices.npz') == read(folder/'assessment.json')['indices_sha256']
            with np.load(folder/'indices.npz') as z:
                ids = {k: z[k] for k in z.files}
            vi = ids['validation_ids']
            assessed.extend(vi.tolist())
            tree = cKDTree(vp[vi].astype(np.float64)*np.array([abs(protocol['transform'][4]), abs(protocol['transform'][0])]))
            predictions = {'original': np.load(folder/'original_predictions.npy')}
            counts = {}
            for gap in (750, 41000):
                dest = folder/f'gap_{gap}m'
                metrics = read(dest/'metrics.json')
                assert digest(dest/'model.pt') == metrics['model_sha256']
                assert digest(dest/'predictions.npy') == metrics['prediction_sha256']
                assert metrics['epochs'] == spec['epochs']
                artifact = torch.load(dest/'model.pt', map_location='cpu', weights_only=True)
                assert artifact['columns'] == spec['columns']
                assert artifact['params'] == spec['params']
                assert artifact['epoch'] == spec['epochs']
                assert artifact['architecture'] == spec['architecture']
                assert artifact['wide_context'] is True
                predictions[str(gap)] = np.load(dest/'predictions.npy')
                train_y = []
                train_x = []
                train_positions = []
                strata = []
                for stage in ('ajuste', 'selecao_camadas'):
                    ti = ids[f'{stage}_{gap}_ids']
                    assert len(np.unique(ti)) == len(ti)
                    pos = samples[f'{stage}_positions'][ti]
                    train_positions.append(pos)
                    dist = tree.query(pos.astype(np.float64)*np.array([abs(protocol['transform'][4]), abs(protocol['transform'][0])]), workers=4)[0]
                    assert np.all(dist >= gap), (h, fold, gap, dist.min())
                    assert np.allclose(dist, ids[f'{stage}_{gap}_distance_m'], atol=1e-8, rtol=0)
                    train_y.append(samples[f'{stage}_y'][ti])
                    train_x.append(samples[f'{stage}_x'][ti])
                    yy = samples[f'{stage}_target_year'][ti]
                    strata.extend([(stage, int(yr), int((yy==yr).sum())) for yr in np.unique(yy)])
                counts[gap] = strata
                ty = np.concatenate(train_y).astype(np.float64)
                tx = np.concatenate(train_x)[:, spec['columns']]
                expected_mean = np.mean(tx, axis=0, dtype=np.float64).astype(np.float32)
                expected_scale = np.std(tx, axis=0, dtype=np.float64)
                expected_scale[expected_scale == 0] = 1
                assert np.array_equal(expected_mean, np.asarray(artifact['input_mean'], dtype=np.float32))
                assert np.array_equal(expected_scale.astype(np.float32), np.asarray(artifact['input_scale'], dtype=np.float32))
                assert artifact['target_mean'] == float(ty.mean())
                assert artifact['target_scale'] == (float(ty.std()) or 1.0)
                physical_scale = np.array([abs(protocol['transform'][4]), abs(protocol['transform'][0])])
                training_tree = cKDTree(np.concatenate(train_positions).astype(np.float64)*physical_scale)
                eval_distances = training_tree.query(vp[vi].astype(np.float64)*physical_scale, workers=4)[0]
                assert np.all(eval_distances >= gap)
                separations.append(dict(horizon=h, fold=fold, gap_m=gap,
                    assessment_to_training_min_km=float(eval_distances.min()/1000),
                    assessment_to_training_median_km=float(np.median(eval_distances)/1000),
                    assessment_to_training_p90_km=float(np.quantile(eval_distances,.9)/1000),
                    assessment_to_training_max_km=float(eval_distances.max()/1000)))
                combined[f'mean_{gap}'][vi] = ty.mean()
                assert metrics['n_train'] == len(ty)
                distributions.append(dict(horizon=h, fold=fold, gap_m=gap, n_train=len(ty),
                     train_target_mean=float(ty.mean()), train_positive_fraction=float((ty>0).mean()),
                     eval_target_mean=float(vy[vi].mean()), eval_positive_fraction=float((vy[vi]>0).mean())))
                checks.append(dict(horizon=h, fold=fold, gap_m=gap, n_train=len(ty), n_eval=len(vi),
                     minimum_distance_m=float(min(ids[f'{s}_{gap}_distance_m'].min() for s in ('ajuste', 'selecao_camadas'))),
                     model_and_prediction_hashes_verified=True, metrics_recomputed=True))
                s = score(vy[vi], predictions[str(gap)])
                assert abs(s['rmse']-metrics['rmse']) < 1e-9
            assert counts[750] == counts[41000]
            for condition, pred in predictions.items():
                assert len(pred) == len(vi) and np.isfinite(pred).all()
                combined[condition][vi] = pred
                rows.append(dict(horizon=h, model=spec['model'], fold=fold, condition=condition, **score(vy[vi], pred)))
                for yr in np.unique(years[vi]):
                    m = years[vi] == yr
                    yearly.append(dict(horizon=h, model=spec['model'], fold=fold, condition=condition,
                                       target_year=int(yr), **score(vy[vi][m], pred[m])))
            a, b = score(vy[vi], predictions['750']), score(vy[vi], predictions['41000'])
            contrasts.append(dict(horizon=h, model=spec['model'], fold=fold, n=len(vi), rmse_750=a['rmse'],
                rmse_41000=b['rmse'], delta_rmse=b['rmse']-a['rmse'], percent_rmse=100*(b['rmse']/a['rmse']-1),
                mae_750=a['mae'], mae_41000=b['mae'], r2_750=a['r2'], r2_41000=b['r2']))
        assert len(assessed) == len(vy) and np.array_equal(np.sort(assessed), np.arange(len(vy)))
        assert all(np.isfinite(p).all() for p in combined.values())
        combined['zero'] = np.zeros(len(vy))
        for condition, pred in combined.items():
            rows.append(dict(horizon=h, model=spec['model'], fold='pooled', condition=condition, **score(vy, pred)))
            for yr in np.unique(years):
                mask = years == yr
                yearly.append(dict(horizon=h, model=spec['model'], fold='pooled', condition=condition,
                                  target_year=int(yr), **score(vy[mask], pred[mask])))
        original_score = score(vy, combined['original'])
        assert abs(original_score['rmse']-spec['original_rmse']) < 1e-4
        a, b = score(vy, combined['750']), score(vy, combined['41000'])
        pooled[h] = {'model':spec['model'], 'original':original_score, 'a':a, 'b':b,
                     'zero':score(vy, combined['zero']), 'mean_750':score(vy, combined['mean_750']),
                     'mean_41000':score(vy, combined['mean_41000'])}
        contrasts.append(dict(horizon=h, model=spec['model'], fold='pooled', n=len(vy), rmse_750=a['rmse'],
            rmse_41000=b['rmse'], delta_rmse=b['rmse']-a['rmse'], percent_rmse=100*(b['rmse']/a['rmse']-1),
            mae_750=a['mae'], mae_41000=b['mae'], r2_750=a['r2'], r2_41000=b['r2']))
        np.savez_compressed(OUT/f'H{h}_pooled_predictions.npz', target=vy, positions=vp, target_year=years, **combined)
    for name, records in [('metrics', rows), ('metrics_by_year', yearly), ('paired_comparisons', contrasts),
                           ('target_distributions', distributions), ('verification_checks', checks),
                           ('actual_separations', separations)]:
        common._write_csv(OUT/f'{name}.csv', records)
    common._write_json(OUT/'summary.json', pooled)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), layout='constrained')
    for ax, h in zip(axes, (0, 1)):
        subset = [r for r in contrasts if r['horizon']==h]
        x = np.arange(3)
        ax.bar(x-.19, [r['rmse_750'] for r in subset], .38, color='#356fa3', label='Minimum gap 0.75 km')
        ax.bar(x+.19, [r['rmse_41000'] for r in subset], .38, color='#c45432', label='Minimum gap 41 km')
        ax.set_xticks(x, ['West', 'East', 'Pooled'])
        ax.set_ylabel('RMSE (original target units)')
        ax.set_title(f'{pooled[h]["model"]} / H{h}')
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        for c in ax.containers:
            ax.bar_label(c, fmt='%.2f', padding=3, fontsize=9)
        ax.margins(y=.2)
    axes[0].legend(frameon=False, fontsize=9, loc='upper left')
    fig.suptitle('Spatial exclusion sensitivity — matched training counts', fontsize=13)
    fig.savefig(OUT/'paired_rmse.png', dpi=220)
    fig.savefig(OUT/'paired_rmse.pdf')
    plt.close(fig)
    figures.append('paired_rmse.png')
    h0, h1 = pooled[0], pooled[1]
    medians = {h: [r['assessment_to_training_median_km'] for r in separations if r['horizon']==h and r['gap_m']==41000] for h in (0, 1)}
    text = '# Sensibilidade à exclusão espacial de 41 km\n\n'
    text += 'Rodada concluída em 10/10/2026. Oito modelos treinados do zero, apenas com dados de desenvolvimento. '
    text += 'Vencedores previamente definidos: CNN H0 (24 canais, 25 épocas) e ResNet compacta H1 (12 canais GPA/gradientes, 32 épocas).\n\n'
    text += (f'**Resultado principal:** o RMSE agregado aumentou {100*(h0["b"]["rmse"]/h0["a"]["rmse"]-1):.1f}% em H0 e '
             f'{100*(h1["b"]["rmse"]/h1["a"]["rmse"]-1):.1f}% em H1; a piora apareceu nas duas metades em ambos os casos. '
             'Isso evidencia sensibilidade à exclusão espacial, mas não identifica causalmente vazamento ou a causa da queda externa. '
             f'**Ressalva central para H1:** as distâncias medianas efetivas ao treino ficaram entre {min(medians[1]):.0f} e {max(medians[1]):.0f} km '
             f'(H0: {min(medians[0]):.0f}–{max(medians[0]):.0f} km). Portanto, H1 é um teste amplo de transferência espacial e temporal, '
             'e sua variação não deve ser atribuída apenas a um afastamento de 41 km.\n\n')
    text += '## Resultados pareados\n\n'
    text += table(contrasts, ['model','horizon','fold','n','rmse_750','rmse_41000','percent_rmse','r2_750','r2_41000'],
                  ['Modelo','H','Avaliação','n','RMSE ≥0,75 km','RMSE ≥41 km','Δ RMSE (%)','R² ≥0,75 km','R² ≥41 km'])
    text += '\nPooled concatena as previsões das duas metades: cada ponto de validação original é avaliado exatamente uma vez por condição. '
    text += 'O RMSE agregado foi calculado dos erros de todos os pontos; não é a média dos RMSEs das metades.\n\n'
    text += '![Comparação pareada](paired_rmse.png)\n\n'
    text += '## O que foi mantido e o que mudou\n\n'
    text += ('A grade foi dividida na coluna central, 5053,5, sem usar valores do target. Em cada repetição, foram avaliados os pontos de validação originais de uma metade. '
             'O conjunto candidato de treino foi o ajuste final original (ajuste + seleção de camadas). Retiraram-se os centros a menos de 750 m ou de 41.000 m '
             'de qualquer centro de avaliação dessa metade, considerando todos os anos simultaneamente. A distância é euclidiana na projeção original, em metros. '
             'As duas condições receberam o mesmo número de amostras em cada etapa e ano, com prioridades aleatórias comuns (seed 4201); a seed de treino foi 4210. '
             'Canais, hiperparâmetros, épocas e batch size foram congelados. As normalizações foram recalculadas exclusivamente no treino de cada condição. '
             'Não houve early stopping, filtragem de canais nem seleção nova na avaliação. O limite de 100.000 centros por etapa/ano foi preservado.\n\n')
    text += ('H0 usa targets 2020/2021/2022 nas duas partes. H1 usa targets 2020/2021 no treino e 2022 na avaliação, preservando a relação input do ano anterior → target. '
             'Os modelos finais originais de H0 e H1 haviam incorporado a etapa de seleção no reajuste final; isso também foi preservado.\n\n')
    text += ('![Desenho espacial](spatial_design.png)\n\nO mapa mostra uma subamostra apenas para visualização. A exclusão foi calculada contra todos os centros de avaliação da metade, '
             'não contra os pontos exibidos no mapa. Pequenas ilhas de treino podem permanecer dentro da metade avaliada, desde que atendam à distância mínima.\n\n')
    text += '### Distâncias efetivas dos pontos avaliados ao treino mais próximo\n\n'
    text += table(separations, ['horizon','fold','gap_m','assessment_to_training_min_km','assessment_to_training_median_km','assessment_to_training_p90_km'],
                  ['H','Metade','Exclusão (m)','Mínimo (km)','Mediana (km)','P90 (km)'])
    text += ('\n41 km é um limite mínimo, não uma distância fixa. A retirada de muitos pontos vizinhos pode deixar os pontos avaliados muito mais distantes do treino, '
             'aproximando a tarefa de uma transferência geográfica. Esse efeito faz parte deste desenho e limita a atribuição da diferença exclusivamente à largura nominal do buffer.\n\n')
    text += '## Por que não bastava mudar buffer_pixels\n\n'
    text += ('O código original erode cada lado de uma fronteira, por distância de Chebyshev. Com blocos aleatórios de 96 pixels (24 km), '
             'buffer_pixels=164 (41 km por lado) deixa 92.936 pixels na zona 1 e **zero** nas zonas 2 e 3, antes mesmo da máscara comum de elegibilidade. '
             'Mesmo 82 pixels por lado (~41 km de separação total) deixam apenas 3.136/1.568 pixels nas zonas 2/3 na grade inteira. '
             'Por isso foi necessário um experimento de exclusão espacial, com pontos de avaliação fixos.\n\n'
             'H0 originalmente usa zonas 1/2/3 para ajuste/seleção/validação. H1 usa anos diferentes dentro da zona 1; aumentar apenas a erosão dessa zona não separaria '
             'espacialmente treino e validação de H1. A exclusão nova considera proximidade entre anos.\n\n'
             'O controle ≥750 m desta rodada é uma distância explícita entre centros. Não reproduz exatamente a erosão original de três pixels por lado: '
             'em H0 a separação herdada já é maior. Os resultados originais servem apenas como referência descritiva.\n\n')
    refs = [r for r in rows if r['fold']=='pooled']
    text += '## Referências na mesma avaliação\n\n'
    text += table(refs, ['model','horizon','condition','rmse','mae','r2'], ['Modelo','H','Condição','RMSE','MAE','R²'])
    text += '\n`original` usa o checkpoint publicado; `750` e `41000` usam os novos ajustes pareados; `zero` prevê zero; '
    text += '`mean_750` e `mean_41000` preveem a média do respectivo treino, separadamente em cada metade.\n\n'
    text += '## Limites de interpretação\n\n'
    text += ('Este é um teste de sensibilidade pós-hoc, com uma seed, duas metades geográficas e configurações selecionadas anteriormente no desenvolvimento. '
             'Não é uma validação independente de toda a seleção de modelos, nem um estudo completo de validação cruzada espacial. '
             'As metades têm distribuições de target distintas; o arquivo target_distributions.csv documenta também como o afastamento muda a distribuição do treino. '
             'Igualar o número de amostras controla o tamanho do treino, mas não elimina diferenças geográficas, de covariáveis e de intensidade do target. '
             'Uma piora mostra sensibilidade à exclusão espacial; não quantifica causalmente vazamento nem prova que esse seja o motivo da queda na região externa.\n\n'
             'Compartilhar entradas exógenas não equivale, por si só, a usar rótulos de avaliação no treino. A preocupação aqui é a dependência espacial e a diferença '
             'entre prever em locais próximos aos dados de treino e transferir o modelo para outra região.\n\n'
             'Em H1, a avaliação original mede previsão temporal dentro da mesma zona; reaproveitar locais em anos distintos pode ser compatível com esse objetivo. '
             'A nova exclusão acrescenta uma exigência de transferência espacial à separação temporal. Assim, a piora neste teste não invalida automaticamente '
             'a avaliação original de previsão temporal em locais próximos ao treino.\n\n'
             'Uma separação de 41 km entre centros não torna os contextos independentes. CNN e ResNet amostram entradas até ±41 km em cada eixo; '
             'suportes diretos e os suportes adicionais das covariáveis derivadas podem se sobrepor. Os rasters de entrada foram mantidos congelados, '
             'sem reconstrução dos KDEs/grafos por partição. Halos e alcances de outros modelos não foram testados nesta rodada. '
             'Não foram calculados p-valores nem intervalos que tratassem pixels como observações independentes.\n\n')
    text += '## Texto para o artigo (English)\n\n'
    english = ('As a post-hoc sensitivity analysis, we retrained the previously selected CNN for H0 and compact ResNet for H1 after excluding training centers '
        'within 41 km of the assessment centers. The original assessment samples were partitioned into western and eastern halves using the raster midpoint, '
        'and each half was assessed separately. A paired control used a minimum center separation of 0.75 km, with identical assessment points and matched '
        'training counts within each original stage and target year. Channels, hyperparameters, epoch counts, and the training seed were fixed, and normalization '
        'was fitted using training samples only. Pooled assessment RMSE changed from '
        f'{h0["a"]["rmse"]:.3f} to {h0["b"]["rmse"]:.3f} for CNN H0 and from {h1["a"]["rmse"]:.3f} to {h1["b"]["rmse"]:.3f} for ResNet H1. '
        f'After exclusion, median nearest-training distances across the two assessment halves ranged from {min(medians[0]):.0f} to {max(medians[0]):.0f} km for H0 '
        f'and from {min(medians[1]):.0f} to {max(medians[1]):.0f} km for H1. The H1 analysis therefore constitutes a broad spatial-temporal transfer stress test '
        'beyond its original within-zone temporal assessment. '
        'These comparisons measure sensitivity to spatial exclusion while controlling training sample size. They do not isolate a causal effect of information '
        'overlap, because exclusion also changes the geographic and covariate distributions of training data. Moreover, a 41-km center separation does not '
        'guarantee non-overlapping model or derived-covariate supports. The analysis uses previously selected configurations and should therefore be interpreted '
        'as exploratory internal robustness evidence, rather than independent validation or an explanation of external transfer performance.')
    text += english+'\n\n'
    text += ('Suggested limitation for the conclusion: Internal performance remains conditional on the spatial evaluation design. The original three-pixel-per-side '
             'buffer does not establish independence at the scales of the model contexts and derived covariates; wider spatial exclusion and independent geographic '
             'replication are required before interpreting internal scores as evidence of regional transferability.\n\n')
    text += '## Arquivos e reprodução\n\n'
    text += ('`protocol.json` foi congelado antes dos ajustes. `training_coverage.csv` e `verification_checks.csv` registram as distâncias e contagens. '
             '`metrics.csv`, `metrics_by_year.csv` e `paired_comparisons.csv` contêm os resultados numéricos. '
             'Cada pasta de condição contém checkpoint, previsões e histórico de treino. `sources.json` contém SHA-256 dos dados/cache de desenvolvimento. '
             'Os checkpoints originais foram conferidos por hash e os seus RMSEs foram reproduzidos nas previsões agregadas.\n\n'
             'Reproduzir/retomar: `C:\\Gepeto_SandBox\\raster_fusion_ai\\.venv\\Scripts\\python.exe C:\\Gepeto_SandBox\\work\\buffer_sensitivity_20261010\\run_experiment.py`. '
             'Gerar análise: executar `analyze_results.py` com o mesmo Python.\n\n'
             'Referência metodológica: [Roberts et al. (2017), Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure](https://doi.org/10.1111/ecog.02881). '
             'A referência fundamenta a discussão da estrutura espacial; os resultados quantitativos acima são desta execução local.\n')
    (OUT/'RELATORIO.md').write_text(text, encoding='utf-8')
    (OUT/'MANUSCRIPT_TEXT.txt').write_text(english+'\n', encoding='utf-8')
    # Check that all source caches remain byte-identical after fitting.
    for source in read(OUT/'sources.json'):
        path = Path(source['path'])
        assert digest(path) == source['sha256'], path
    for path in [OUT/'run_experiment.py', ROOT/'src/services/enriched_cnn.py',
                 ROOT/'src/services/enriched_linear.py', ROOT/'src/splitting/spatial_split.py']:
        assert digest(path) == digest(OUT/'source_snapshot'/path.name), path
    common._write_json(OUT/'VERIFICACAO.json', dict(status='passed', fitted_models=len(checks),
        original_rmse_reproduced=True, source_hashes_unchanged=True, projected_distances_recomputed=True,
        matched_stage_year_counts=True, assessment_points_same_between_conditions=True,
        each_original_assessment_point_scored_once_per_condition=True, frozen_protocol_unchanged=True,
        metrics_recomputed=True, training_epochs_verified=True, training_only_normalization_verified=True,
        channels_hyperparameters_architecture_verified=True, external_data_accessed=False,
        training_source_matches_frozen_snapshot=True,
        figures=figures, completed_utc=common._utc()))
    status = read(OUT/'status.json')
    status.update(status='completed', analysis_completed_utc=common._utc())
    common._write_json(OUT/'status.json', status)
    print(json.dumps(pooled, indent=2), flush=True)


if __name__ == '__main__':
    main()
