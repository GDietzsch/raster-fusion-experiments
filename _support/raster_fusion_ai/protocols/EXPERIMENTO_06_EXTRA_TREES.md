# Experimento 06 — Extra Trees

Protocolo definido antes de qualquer ajuste do experimento 06. O objetivo continua sendo prever a intensidade do target sem converter o zero em ausência permanente do fenômeno. O target externo da região de teste já foi examinado em experimentos anteriores; por isso as comparações entre modelos são exploratórias.

## Modelo e divisão dos dados

Usar `sklearn.ensemble.ExtraTreesRegressor` com perda `squared_error`, `bootstrap=False`, `oob_score=False`, 4 threads no máximo e sementes fixas 4207 (árvores) e 4208 (candidatos). Com `bootstrap=False`, cada árvore recebe a amostra completa do respectivo ajuste; `max_samples` não se aplica. Extra Trees aleatoriza também os limiares de divisão, distinguindo-se do Random Forest. O número de árvores é orçamento de estabilidade, não “épocas”.

Repetir sem alteração os 24 canais possíveis, os seis braços, a máscara comum, H0 e H1, a amostragem, os blocos espaciais, os pares temporais, as métricas, as referências e a regra de vencedor dos experimentos 01–05. H0 usa zona 1 no ajuste inicial, zona 2 na seleção interna e zona 3 na validação, nos targets 2020–2022. H1 usa targets 2020/2021/2022 nas três etapas dentro da zona de desenvolvimento 1, com entradas do ano anterior. Os dois vencedores são congelados antes de abrir qualquer target externo.

## Busca pré-definida

1. Para busca e filtragem, amostrar até 20.000 pixels por ano e etapa interna, com os mesmos índices determinísticos compartilhados por todos os candidatos de cada horizonte. Guardar índices e hashes. A validação final e o reajuste final usam as amostras completas originais.
2. Para cada horizonte e família base (`originais`, `gpa`, `grafos`, `todas`), avaliar 16 configurações da grade: a referência fixa e 15 sorteadas sem reposição. Treinar 96 árvores no ajuste inicial e medir RMSE/MAE na seleção de camadas. Reavaliar os três melhores de cada família com 320 árvores nas mesmas etapas. Escolher por RMSE, depois MAE, depois menor complexidade (`max_leaf_nodes` menor e `min_samples_leaf` maior). Registrar todas as tentativas, tempo, RSS após ajuste, número médio de nós, curvas de prefixo 96/160/320 e parâmetros.
3. Os braços filtrados herdam os parâmetros de suas famílias base. `originais_filtradas` avalia as 15 combinações não vazias de fenômenos; `todas_filtradas` usa eliminação reversa dos grupos fenômeno×produto, mantendo pares seno/cosseno e conectividade/suporte juntos. Cada candidato de filtragem tem 160 árvores e é comparado por RMSE/MAE apenas na seleção interna. Guardar cada tentativa.
4. Reajustar os seis braços com ajuste inicial + seleção completos e 320 árvores. Avaliar uma vez na validação completa. Selecionar um vencedor por horizonte por RMSE, MAE e menos canais; nenhuma curva ou métrica da área externa autoriza nova escolha nesta execução.

| Parâmetro | Grade | Papel |
|---|---|---|
| `max_features` | 0,50; 0,75; 1,00 | Fração de canais considerados por divisão. |
| `min_samples_leaf` | 0,0005; 0,002; 0,01 | Suavidade em fração das amostras do ajuste. |
| `max_leaf_nodes` | 256; 1024; 4096 | Limite de complexidade/memória da árvore. |
| `min_samples_split` | 0,001; 0,005; 0,02 | Fração mínima para dividir um nó. |

A referência incluída nas 16 configurações é `max_features=0.75`, `min_samples_leaf=0.002`, `max_leaf_nodes=1024`, `min_samples_split=0.005`. A grade completa tem 81 configurações. A lista sorteada é idêntica entre famílias e horizontes. Ajustar uma tentativa por vez, com até quatro threads internas, e manter checkpoint após cada tentativa para retomada. O orçamento 96/160/320 é fixo e comparável ao experimento 05; as diferenças entre Random Forest e Extra Trees incluem a construção das divisões e a escolha explícita de não usar bootstrap aqui.

## Avaliação e relatório

Depois de congelados os vencedores, testar H0 nos pares externos 2020→2020, 2021→2021 e 2022→2022; H1 em 2021→2022. Acrescentar H1 2020→2021 como avaliação complementar de 2021, decisão tomada antes desta execução e que não altera a seleção. Relatar RMSE, MAE, R², NMAD, viés, P90 do erro absoluto, resultados condicionais ao target zero/positivo, captura de massa no top 5/10/20%, referência zero/média/persistência e incerteza exploratória por blocos. CE90 não é apropriado para erro de intensidade. O relatório final deve mostrar as seis combinações em H0/H1, todos os candidatos de camadas e hiperparâmetros, canais mantidos/removidos, proveniência, durações, modelos, GeoTIFFs, imagens target/previsão de 600 dpi e curvas diagnósticas dos vencedores por prefixos de árvores. Essas curvas são calculadas dos modelos salvos em amostras de desenvolvimento e não selecionam o vencedor.

## Fontes

- Geurts, Ernst e Wehenkel (2006), [*Extremely randomized trees*](https://doi.org/10.1007/s10994-006-6226-1), *Machine Learning* 63, 3–42: método original.
- scikit-learn, [ExtraTreesRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html): parâmetros, divisões aleatórias, bootstrap e semântica de `max_samples`.
- scikit-learn, [RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html): conceito de orçamento fixo de candidatos. As divisões efetivas aqui são as espaciais/temporais do projeto.
- scikit-learn, [importância por permutação com atributos correlacionados](https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html): limites da interpretação individual de canais derivados. A seleção efetiva usa grupos e RMSE interno, não importância por impureza.
- Definições e referências de GPA, grafos, divisão espacial, métricas e figuras: `protocols/EXPERIMENTOS_ENRIQUECIDOS_2026.md`, `docs/GRADIENT_INPUTS.md` e `docs/GRAPH_INPUTS.md`.
