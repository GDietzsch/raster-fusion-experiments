# Experimento 02 — Ridge com camadas enriquecidas

O script `run_enriched_ridge.py` usa o mesmo manifesto, os mesmos 24 canais, seis conjuntos, zonas espaciais, anos, amostras e métricas do experimento 01. É uma execução nova, com diretório próprio em `runs/enriched_ridge/<data_hora>`. Não reutiliza pesos nem seleciona parâmetros com a região externa.

## Hipótese e estimador

A hipótese operacional é que a penalização L2 pode estabilizar a combinação linear dos canais originais, GPA e grafo quando há correlação entre eles. O modelo é `Pipeline(StandardScaler(), Ridge(...))` do scikit-learn. A escala de cada canal é ajustada exclusivamente nas amostras usadas em cada ajuste. O target conserva suas unidades. Não há truncamento de previsões negativas.

O parâmetro `alpha` da implementação penaliza a soma dos quadrados dos coeficientes. Como o ajuste inicial e o novo ajuste têm quantidades diferentes de pixels, definimos `alpha = lambda_por_amostra × n_amostras` a cada ajuste. Assim a intensidade relativa da penalização fica aproximadamente comparável quando a amostra muda. O solver é `cholesky`, com `alpha` sempre positivo. Essa parametrização e a grade são escolhas do experimento; não há garantia de que a grade contenha o ótimo.

## Busca interna de regularização

Em cada horizonte, testamos `[0,000001; 0,00001; 0,0001; 0,001; 0,01; 0,1; 1]` para `lambda_por_amostra`. A busca é separada para quatro famílias sem filtragem: originais, GPA, grafos e todas. Cada candidato é treinado somente na etapa de ajuste inicial e pontuado por RMSE na etapa de seleção de camadas. O menor RMSE escolhe a lambda; empates usam MAE, depois a menor lambda.

`originais_filtradas` herda a lambda escolhida para originais, e `todas_filtradas` herda a de todas. A busca dos subconjuntos usa essa lambda fixa, com as mesmas 15 combinações originais e a mesma eliminação reversa de 16 grupos fenômeno×produto do experimento 01. Depois disso, cada um dos seis braços é ajustado novamente com ajuste inicial mais seleção de camadas. A validação, não usada para escolher lambda nem canais, escolhe o braço vencedor por RMSE, depois MAE e número de atributos. Os vencedores H0 e H1 são congelados antes de abrir qualquer target externo.

H0 faz ajuste inicial na zona 1, seleção na zona 2 e validação na zona 3, com targets 2020–2022. H1 faz ajuste inicial com target 2020, seleção com 2021 e validação com 2022, na zona 1. O teste externo H0 usa entradas/targets 2020, 2021 e 2022; H1 usa entradas 2021 para target 2022. H0 e H1 são tarefas distintas.

## Relatório final

O script grava `alpha_search.csv`, `alpha_choices.json`, `comparison.csv`, `selection_candidates.csv`, `winner.json`, métricas de todos os braços, modelos e previsões georreferenciadas. Depois do teste externo, cria `RELATORIO.md` e `TODOS_RESULTADOS.md`, com quatro imagens de 600 dpi: target original à esquerda e previsão à direita em cada par externo. O relatório mostra canais usados/removidos, métricas, referências zero/média/persistência, resultados por tipo de target, intervalos exploratórios e todas as combinações realmente avaliadas. A área externa não é usada para refazer lambda, atributos ou escolha do vencedor.

`--check-only` confere os dados e a configuração sem treinamento e sem abrir os valores dos targets externos. `--report-only <pasta_da_execução>` refaz imagens e relatório de uma execução externa concluída ou parcialmente finalizada, sem retreinar nem alterar os GeoTIFFs previstos.

## Referências

- Hoerl, A. E., & Kennard, R. W. (1970). [Ridge Regression: Biased Estimation for Nonorthogonal Problems](https://doi.org/10.1080/00401706.1970.10488634). *Technometrics*, 12(1), 55–67.
- [Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html), [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html), [Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) e [validação](https://scikit-learn.org/stable/modules/cross_validation.html), documentação oficial do scikit-learn.
- Valavi et al. (2019). [blockCV: An R package for generating spatially or environmentally separated folds](https://doi.org/10.1111/2041-210X.13107). *Methods in Ecology and Evolution*.
- Stock, A. (2025). [Choosing blocks for spatial cross-validation: lessons from a marine remote sensing case study](https://doi.org/10.3389/frsen.2025.1531097). *Frontiers in Remote Sensing*.
- Definições específicas deste projeto: `docs/GRADIENT_INPUTS.md`, `docs/GRAPH_INPUTS.md` e `protocols/EXPERIMENTOS_ENRIQUECIDOS_2026.md`. O relatório final adiciona as referências de métricas, reamostragem, NMAD, CE90 e visualização junto às respectivas definições.
