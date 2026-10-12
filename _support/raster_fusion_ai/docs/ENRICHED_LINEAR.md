# Experimento 01 — regressão linear com camadas enriquecidas

O experimento está implementado em `run_enriched_linear.py`, com parâmetros congelados em `configs/enriched_linear.yaml` e funções auditáveis em `src/services/enriched_linear.py`. O comando `--check-only` valida arquivos, manifesto, grades e NoData sem treinamento e sem abrir os valores dos targets externos. Sem essa opção, uma execução nova gera `runs/enriched_linear/<data_hora>/` e não sobrescreve rodadas anteriores.

## Por que regressão linear primeiro

É uma referência simples e interpretável. O estimador de mínimos quadrados escolhe coeficientes para minimizar a soma dos quadrados dos resíduos; sua simplicidade ajuda a medir se as transformações de gradiente e grafo acrescentam sinal antes de modelos mais flexíveis. Coeficientes ou subconjuntos escolhidos não demonstram causalidade, especialmente com entradas correlacionadas. [Documentação do scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html); [exemplo de importância enganosa com variáveis correlacionadas](https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html).

## Separação temporal e espacial

- H0: inputs e target do mesmo ano, 2020–2022; zona espacial 1 para ajuste inicial, zona 2 para selecionar camadas e refazer o ajuste, zona 3 para escolher o conjunto vencedor. Teste externo nos anos 2020–2022.
- H1: inputs de 2019, 2020 e 2021 para targets de 2020, 2021 e 2022. Target 2020 serve ao ajuste inicial, 2021 à seleção de camadas e novo ajuste, 2022 à escolha do vencedor, sempre na zona espacial 1. Teste externo: inputs de 2021 para target de 2022.

Os blocos espaciais são de 96 pixels (24 km), com exclusão de três pixels nas interfaces. Esse tamanho preserva comparabilidade com o piloto anterior; não é estimativa comprovada do alcance da dependência. A literatura recomenda separação espacial quando pixels próximos podem tornar a validação otimista: [Valavi et al. (2019)](https://doi.org/10.1111/2041-210X.13107) e [estudo de escolha dos blocos (2025)](https://doi.org/10.3389/frsen.2025.1531097).

## Seis conjuntos de entradas

Cada um dos quatro fenômenos — garimpos, ocorrências minerais, pistas regulares e pistas irregulares — dispõe de original, magnitude, seno/cosseno da direção, conectividade e suporte do grafo. Os braços são: originais (4), originais filtradas (1–4), GPA (12), grafos (8), todas (24) e todas filtradas (subconjunto de 24). Direção em graus não é usada por sua descontinuidade circular.

Em `originais_filtradas`, são comparadas as 15 combinações não vazias dos quatro fenômenos. Em `todas_filtradas`, uma busca reversa remove grupos de produto por fenômeno: original; magnitude; direção seno/cosseno; conectividade/suporte. Só aceita retirada que não piore o RMSE na etapa de seleção. Isso é uma busca gulosa, com resultados posteriormente examinados numa validação separada.

Todos os seis braços usam a mesma máscara de pixels onde os 24 canais e o target são válidos. A amostragem é uniforme sem reposição, limitada a 100.000 pixels por ano e etapa. Normalização dos inputs é ajustada no treino; target e métricas ficam na escala original. O modelo de cada braço é ajustado novamente após a etapa de seleção, e o menor RMSE na validação escolhe o vencedor; empates: MAE, depois menor número de canais. A região externa não participa dessas decisões.

## Métricas e artefatos

O relatório da execução apresenta RMSE, MAE, R², viés, NMAD, percentil 90 do erro absoluto e captura da massa do target nas áreas de maior escore (5%, 10%, 20%). Mostra target zero, positivo e intensidades altas separadamente, além das referências zero, média de treino e persistência histórica quando aplicável. NMAD é `1,4826022185 × mediana(|resíduo − mediana(resíduo)|)` e complementa, sem substituir, as métricas sensíveis aos extremos ([SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.median_abs_deviation.html)). CE90 não é calculado: é medida de erro posicional horizontal ([USGS, 2024](https://pubs.usgs.gov/publication/ofr20241023/full)).

A execução grava `RELATORIO.md`, CSV e JSON de seleção/métricas, amostras e posições, GeoTIFF das previsões externas, modelos, configuração, ambiente, hashes das fontes, máscara espacial e código utilizado. Intervalos exploratórios de RMSE e MAE são obtidos por bootstrap de blocos de ~96 km. Essa escala é análise de sensibilidade, não garantia de independência.

Os novos modelos não são ajustados nos targets externos. A região foi examinada em investigações anteriores do projeto; essa limitação aparecerá no relatório da execução.

## Complemento visual e relatório consolidado da primeira execução

A execução concluída em `runs/enriched_linear/20260925T141835755712Z` foi auditada e ganhou um relatório consolidado em `C:\Gepeto_SandBox\experiment_analysis\20260925T150131703700Z\TODOS_RESULTADOS.md`. Esse relatório inclui as seis combinações finais por horizonte, candidatos da filtragem, canais dos vencedores e todas as métricas externas. Quatro figuras colocam o target original à esquerda e a previsão à direita para H0 2020, 2021 e 2022, e H1 2021→2022. Foram exportadas em PNG de 600 dpi e verificadas visualmente.

Os scripts `plot_enriched_linear_external.py` e `report_enriched_linear_all.py` reproduzem, nessa ordem, as figuras e o relatório a partir dos artefatos já concluídos, sem retreinar ou alterar as previsões. O padrão provisório para os próximos modelos está registrado em `protocols/EXPERIMENTOS_ENRIQUECIDOS_2026.md`.
