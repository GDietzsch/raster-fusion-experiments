# Experimento 04 — XGBoost com camadas enriquecidas

`run_enriched_xgboost.py` repete o catálogo, 24 canais possíveis, seis conjuntos, H0/H1, zonas espaciais, pares temporais, amostras finais, máscaras, métricas e regra do vencedor dos experimentos 01–03. Cada execução tem sua própria pasta em `runs/enriched_xgboost/<data_hora>`; nenhum modelo anterior é reutilizado. O target da área externa só é aberto depois que os vencedores dos dois horizontes são salvos.

## Hipótese e estimador

O `XGBRegressor` ajusta árvores sequenciais para representar relações não lineares e interações entre canais originais, GPA e de grafo. O objetivo fixo é `reg:squarederror`, coerente com RMSE como critério primário. O método de árvore é `hist` em CPU. Todas as entradas são numéricas e não recebem padronização. O target conserva suas unidades; previsões negativas não são truncadas. A hipótese é preditiva, sem interpretação causal de atributos ou voos.

Para manter uma comparação controlada com HistGradientBoosting, a configuração fixa taxa de aprendizado 0,08, até quatro níveis (aproximadamente 16 folhas), peso mínimo de folha 80, regularização L2 de 1, L1 e `gamma` iguais a zero, 128 bins e sem amostragem estocástica de linhas/colunas. A grade de árvores é `[60, 120, 180]`. Esses valores foram registrados antes da execução XGBoost; não foram escolhidos com o target externo. A semente do modelo é 4204 e a versão instalada é XGBoost 3.4.1. `early_stopping_rounds` é `null` e não há `eval_set`, preservando as etapas espaciais/temporais explícitas.

## Busca interna e refit completo

A busca de árvores e a filtragem usam a mesma subamostra fixa do experimento 03: até 20.000 pixels por ano em cada etapa interna de ajuste e seleção, com `selection_seed=4203`, retirada das amostras completas. O script salva os índices, hashes e contagens. Cada candidato de uma família é ajustado somente na subamostra de ajuste e comparado por RMSE na subamostra de seleção; empates usam MAE, depois menos árvores. A grade é escolhida separadamente para originais, GPA, grafos e todas. `originais_filtradas` herda a escolha de originais e avalia as 15 combinações não vazias dos quatro fenômenos. `todas_filtradas` herda a escolha de todas e faz eliminação reversa dos 16 grupos fenômeno×produto, mantendo seno/cosseno e conectividade/suporte juntos. Todas as tentativas são registradas.

Os **seis braços** são depois ajustados novamente com todas as amostras de ajuste inicial e seleção: até 600.000 pixels em H0 e 200.000 em H1. A validação completa, de até 300.000 pixels em H0 e 100.000 em H1, escolhe um vencedor por RMSE, MAE e menor número de canais. O RMSE da busca não usa a mesma base amostral dos experimentos lineares; o RMSE da validação usa. Somente os vencedores congelados recebem teste externo.

## Relatório e integridade

O script salva `model.joblib` para a inferência desta execução e `model.ubj` no formato nativo XGBoost, com SHA-256 registrado. O relatório automático mostra os 12 braços, a busca de árvores, todas as combinações de originais e tentativas de filtragem, canais mantidos/removidos, métricas e referências. H0 é testado em 2020→2020, 2021→2021 e 2022→2022; H1 em 2021→2022. Quatro PNGs de 600 dpi apresentam target original à esquerda e previsão à direita, na mesma máscara e escala por par, além dos GeoTIFFs na resolução original.

`--check-only` confere a configuração e o catálogo sem treinar nem abrir os valores dos targets externos. `--report-only <pasta_da_execução>` regenera figuras e relatório de uma inferência externa concluída sem refazer o treino. A área externa não participa deste novo ajuste, embora já tenha sido examinada em investigações anteriores do projeto; os resultados são exploratórios.

Na execução completa, a auditoria SHA-256 dos 195 arquivos de entrada e targets de desenvolvimento ocorre antes do treinamento e informa o progresso no terminal e em `training.log`. Se a execução receber uma interrupção de teclado, `run.json` registra `interrupted` e a pasta parcial é preservada; uma nova chamada cria outra pasta.

## Referências principais

- Chen, T., & Guestrin, C. (2016). [XGBoost: A Scalable Tree Boosting System](https://doi.org/10.1145/2939672.2939785). *KDD '16*, 785–794.
- [Parâmetros XGBoost](https://xgboost.readthedocs.io/en/stable/parameter.html), [API Python](https://xgboost.readthedocs.io/en/stable/python/python_api.html) e [salvamento de modelos](https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html), documentação oficial.
- Valavi et al. (2019). [blockCV: An R package for generating spatially or environmentally separated folds](https://doi.org/10.1111/2041-210X.13107). *Methods in Ecology and Evolution*.
- Stock (2025). [Choosing blocks for spatial cross-validation: lessons from a marine remote sensing case study](https://doi.org/10.3389/frsen.2025.1531097). *Frontiers in Remote Sensing*.
- Conceitos adicionais de métricas, incerteza, rasters, GPA, grafos e figuras são associados às respectivas fontes no `methodology_appendix.md` do run. As definições próprias estão em `protocols/EXPERIMENTOS_ENRIQUECIDOS_2026.md`, `docs/GRADIENT_INPUTS.md` e `docs/GRAPH_INPUTS.md`.
