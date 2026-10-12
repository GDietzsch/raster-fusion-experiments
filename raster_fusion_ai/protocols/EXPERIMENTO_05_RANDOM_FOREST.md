# Experimento 05 — protocolo de exploração de hiperparâmetros do Random Forest

Protocolo definido antes do treinamento do experimento 05. O objetivo continua sendo regressão da intensidade do target, nas unidades originais. Target zero é ausência de registro naquele pixel e ano; não se deve converter pixels positivos em zero nem reequilibrar a amostra pelo valor do target no experimento principal.

## Pergunta e estrutura fixa

Repetir H0 e H1, a grade, os 24 canais possíveis, seis conjuntos de entrada, máscara comum, amostras de desenvolvimento, zonas espaciais, pares temporais, métricas e referências dos experimentos 01–04. H0: ajuste na zona 1, busca/filtragem na zona 2 e escolha do braço vencedor na zona 3, cobrindo targets 2020–2022. H1: alvo 2020 para ajuste, 2021 para busca/filtragem e 2022 para escolha do braço vencedor, sempre na zona de desenvolvimento 1. A região externa não entra em nenhuma dessas decisões. Nenhuma validação aleatória por pixel e nenhum KFold comum substitui essas separações.

O estimador será `sklearn.ensemble.RandomForestRegressor`, `criterion='squared_error'`, `bootstrap=True`, `oob_score=False`, quatro threads no máximo e `random_state=4205`. O sorteio de configurações usa `seed=4206`. O critério de escolha permanece **menor RMSE**, seguido por menor MAE e, no empate entre braços, menos canais. R², NMAD, viés, erro no target zero e positivo e captura da massa do target nos top 5%, 10% e 20% serão relatados, mas não usados para selecionar hiperparâmetros. CE90 não se aplica ao erro de intensidade raster.

## Busca interna pré-definida

1. Reusar a subamostra fixa dos experimentos 03/04: até 20.000 pixels por ano e por etapa interna, com índices e hashes preservados. O conjunto de validação final permanece completo e intocado durante busca e filtragem.
2. Para cada horizonte e cada um dos quatro braços base (`originais`, `gpa`, `grafos`, `todas`), avaliar **16 configurações**: uma configuração de referência fixa e 15 sorteadas sem reposição da grade abaixo. Usar a mesma lista de configurações nos dois horizontes e em todos os braços, as sementes acima e 96 árvores por tentativa. Ajustar em `ajuste` e pontuar em `selecao_camadas`; registrar RMSE, MAE, tempo e parâmetros de todas as tentativas.
3. Reavaliar as **três melhores configurações por braço** com 320 árvores nas mesmas etapas internas. Escolher menor RMSE, depois MAE e, em igualdade numérica, a configuração menos complexa (`max_leaf_nodes` menor e `min_samples_leaf` maior). Os braços filtrados herdam os parâmetros de seu braço base.
4. Com parâmetros congelados, testar todas as 15 combinações não vazias dos quatro fenômenos para `originais_filtradas` e a eliminação reversa de grupos fenômeno×produto para `todas_filtradas`. Seno/cosseno da direção e conectividade/suporte saem juntos. Usar 160 árvores para cada candidato de filtro e **a mesma regra RMSE→MAE**; guardar cada tentativa. O número de árvores final de 320 é igual nos seis braços.
5. Reajustar os seis braços com todas as amostras de `ajuste + selecao_camadas`, mantendo seus canais e parâmetros escolhidos internamente e usando 320 árvores. Avaliar uma vez na validação completa; escolher um vencedor por H0 e outro por H1 por RMSE→MAE→menos canais. A validação não serve para uma segunda rodada de hiperparâmetros.

| Parâmetro | Valores candidatos | Motivo |
|---|---|---|
| `max_features` | 0,50; 0,75; 1,00 | Varia a aleatoriedade e a interação entre canais. |
| `min_samples_leaf` | 0,0005; 0,002; 0,01 | Controla a suavidade usando fração, para manter escala comparável entre subamostra e reajuste completo. |
| `max_leaf_nodes` | 256; 1024; 4096 | Limita a complexidade e a memória de cada árvore. |
| `max_samples` | 0,50; 0,75; 1,00 | Varia a fração de amostras sorteada para cada árvore. |

A configuração de referência incluída nas 16 é `max_features=0.75`, `min_samples_leaf=0.002`, `max_leaf_nodes=1024`, `max_samples=0.75`. A grade tem 81 combinações; o sorteio de 15 adicionais e sua semente devem ser salvos antes do ajuste. `n_estimators` é tratado como orçamento de estabilidade (96 na triagem, 160 na filtragem e 320 no modelo final), não como parâmetro escolhido pelo target externo. Guardar a curva de erro por prefixo de árvores 96/160/320 na etapa interna como diagnóstico. Os valores são um orçamento de exploração reproduzível, não ótimos universais.

Executar **uma tentativa por vez**, usando até quatro threads dentro da floresta, para não multiplicar cópias de centenas de milhares de pixels em processos paralelos. Salvar progresso e métricas após cada tentativa, com retomada sem refazer candidatos já concluídos. Um piloto sem olhar métricas do target externo mediu cerca de 1–1,3 s para 32 árvores, 60.000 linhas e 24 canais em uma configuração limitada; o tempo total real será maior e depende dos parâmetros e dos reajustes completos.

## Validação final e relatório

Congelar os dois vencedores antes da avaliação externa. Para comparabilidade, testar H0 nos pares externos 2020→2020, 2021→2021 e 2022→2022, e H1 em 2021→2022. Acrescentar H1 2020→2021 como **avaliação complementar de 2021** decidida antes desta execução RF; ela não altera a escolha dos vencedores. Apresentar 2021 como recorte de leitura, mas manter todos os anos e a definição original do teste externo. A região externa já foi examinada em experimentos anteriores; portanto, os resultados entre modelos são exploratórios, mesmo que este RF não use seus targets no ajuste.

Para cada par externo, relatar RMSE, MAE, R², NMAD, viés, erros condicionais ao target zero/positivo, captura da massa nos top 5/10/20%, referências zero/média de treino/persistência, fração positiva e intervalos exploratórios por blocos espaciais. O relatório final deve listar todas as configurações e combinações de camadas, canais mantidos/removidos, hashes dos dados/modelos, versões, tempo e memória, CSVs, GeoTIFFs e pares target/previsão em PNG de 600 dpi. A importância por impureza pode ser salva como diagnóstico, mas não define relevância: canais derivados correlacionados podem dividir sua importância. Nenhum resultado externo provoca novo ajuste nesta execução.

## Fundamentação

- Breiman (2001), [Random Forests](https://doi.org/10.1023/A:1010933404324), *Machine Learning* 45, 5–32.
- scikit-learn, [RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html): parâmetros, bootstrap, tamanho de folha, subconjuntos de atributos e amostras.
- scikit-learn, [RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html): orçamento explícito de candidatos; a implementação aqui usa as etapas espaciais/temporais do projeto, não o `cv` padrão.
- scikit-learn, [GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html): grupos não sobrepostos, princípio que motiva evitar separação aleatória de pixels espacialmente dependentes.
- scikit-learn, [Permutation importance with correlated features](https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html): limites da leitura isolada de importância para canais derivados correlacionados.
- Os conceitos de métricas e divisão espacial já estão referenciados em `protocols/EXPERIMENTOS_ENRIQUECIDOS_2026.md` e nos relatórios dos experimentos 01–04.
