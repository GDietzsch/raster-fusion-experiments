# Experimento 03 — HistGradientBoosting com camadas enriquecidas

`run_enriched_hgb.py` executa H0 e H1 com os mesmos dados, 24 canais possíveis, seis conjuntos, zonas espaciais, anos, amostras finais, métricas e regra do vencedor dos experimentos 01 e 02. Os modelos e o relatório ficam em `runs/enriched_hgb/<data_hora>`. Nada é reutilizado como peso treinado; os targets da área externa só são abertos depois que os vencedores de H0 e H1 são gravados.

## Estimador e hipótese

O modelo é `sklearn.ensemble.HistGradientBoostingRegressor`, com perda quadrática porque RMSE é o critério primário. Árvores podem representar relações não lineares e interações entre as camadas originais, GPA e de grafo sem criar canais adicionais. Esta é uma hipótese preditiva, não uma afirmação causal sobre voos. Os rasters são tratados como variáveis numéricas, sem padronização; o target permanece na unidade original. `early_stopping=False` impede a criação de uma validação aleatória interna que misturaria blocos ou anos. Previsões negativas são preservadas e medidas, não truncadas.

O YAML fixa taxa de aprendizado 0,08, no máximo 15 folhas por árvore, mínimo de 80 amostras por folha, penalização L2 das folhas de 1 e 128 bins. A grade interna de árvores é 60, 120 e 180 iterações. Essas escolhas limitam a busca e o custo; não garantem o melhor modelo possível. A semente é fixa e a versão do scikit-learn será registrada na execução.

## Seleção e novo ajuste

Como treinar centenas de candidatos de árvores nos 100.000 pixels por ano de cada etapa seria oneroso, a busca de iterações e a filtragem usam uma subamostra **fixa e compartilhada** de até 20.000 pixels por ano, separadamente nas etapas de ajuste e seleção de camadas. Os índices e hashes são salvos. Nenhum pixel da validação ou da área externa entra nessa busca. H0 compara zonas espaciais distintas; H1 compara anos distintos na região de desenvolvimento, como no protocolo comum.

A grade de iterações é avaliada em quatro famílias sem filtragem: originais, GPA, grafos e todas. Em cada família, o menor RMSE na subamostra de seleção escolhe as iterações; empate usa MAE e depois menos iterações. `originais_filtradas` herda a escolha de originais e avalia as 15 combinações não vazias dos quatro fenômenos. `todas_filtradas` herda a escolha de todas e aplica eliminação reversa dos 16 grupos fenômeno×produto, mantendo juntos seno/cosseno e conectividade/suporte. Toda tentativa é gravada em `selection_candidates.csv`.

Depois, **cada um dos seis braços** é treinado novamente com todas as amostras de ajuste mais seleção: até 600.000 pixels em H0 e 200.000 em H1. A validação completa, 300.000 pixels em H0 e 100.000 em H1, escolhe um vencedor por RMSE, MAE e menor número de canais. A subamostra muda a base da métrica de seleção em relação aos experimentos 01/02; a validação final usa a mesma amostragem e pode ser comparada entre eles. Todos os hiperparâmetros e canais são congelados antes do teste externo.

## Resultados e relatório

O relatório automático apresenta os 12 braços, as 30 combinações originais, todas as tentativas da filtragem, canais mantidos/removidos, busca de iterações, métricas internas, referências simples e teste externo completo dos vencedores. H0 é avaliado em 2020→2020, 2021→2021 e 2022→2022; H1 em 2021→2022. Quatro imagens PNG de 600 dpi comparam target original à esquerda e previsão à direita, na mesma máscara e escala por par. Os GeoTIFFs preservam a resolução original. O relatório inclui referências metodológicas por conceito.

`--check-only` verifica catálogo, grades e configuração sem treino ou leitura dos valores dos targets externos. `--report-only <pasta_da_execução>` regenera figuras e relatório após a inferência externa, sem retreinar. O relatório é exploratório: a região externa não entra neste treino, mas já foi examinada pelo projeto.

## Referências principais

- [HistGradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html) e [guia de gradient boosting com histogramas](https://scikit-learn.org/stable/modules/ensemble.html#histogram-based-gradient-boosting), documentação oficial do scikit-learn. Descrevem perda, iterações, histogramas e early stopping.
- Friedman, J. H. (2001). [Greedy Function Approximation: A Gradient Boosting Machine](https://doi.org/10.1214/aos/1013203451). *Annals of Statistics*.
- Valavi et al. (2019). [blockCV: An R package for generating spatially or environmentally separated folds](https://doi.org/10.1111/2041-210X.13107). *Methods in Ecology and Evolution*.
- Stock (2025). [Choosing blocks for spatial cross-validation: lessons from a marine remote sensing case study](https://doi.org/10.3389/frsen.2025.1531097). *Frontiers in Remote Sensing*.
- Definições locais e outras referências por conceito: `protocols/EXPERIMENTOS_ENRIQUECIDOS_2026.md`, `docs/GRADIENT_INPUTS.md`, `docs/GRAPH_INPUTS.md` e `methodology_appendix.md` de cada execução.
