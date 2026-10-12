# Protocolo proposto para os experimentos com camadas enriquecidas

Estado: decisões metodológicas registradas antes dos novos treinamentos. Este arquivo não relata resultados novos.

## Objetivo e unidade de comparação

Cada família de modelo terá um experimento próprio, com H0 (inputs e target do mesmo ano) e H1 (inputs do ano anterior e target do ano seguinte) avaliados separadamente. Os 11 modelos previstos são regressão linear, Ridge, HistGradientBoosting, XGBoost, Random Forest, Extra Trees, MLP, CNN, ResNet compacta, U-Net e ResUNet. Um script de entrada por modelo poderá compartilhar funções de leitura, métricas e relatório, preservando o mesmo protocolo de comparação.

## Grupos de entradas

Quatro fenômenos: Garimpos, Ocorrencias_Minerais, Pistas_Regulares e Pistas_Irregulares. Em cada fenômeno, os candidatos de entrada são:

- original;
- GPA: magnitude, direcao_seno, direcao_cosseno;
- grafo: conectividade, suporte.

Assim, os seis braços por horizonte são: originais (4 canais); originais filtradas por relevância (subconjunto); GPA (12); grafos (8); todas (24); todas filtradas por relevância (subconjunto). O arquivo `suporte.tif`, hoje guardado em `auxiliares`, será carregado explicitamente nos braços com grafo. `direcao_graus.tif` não entra em modelos: seno e cosseno já codificam a direção circular. Inputs e targets são separados por ano; nenhum target é usado como entrada.

## Fluxo obrigatório por horizonte e modelo

1. Auditar origem, datas, hashes, CRS, resolução, transformada, NoData e zeros; alinhar sem sobrescrever fontes.
2. Fixar pares temporais, zonas espaciais, amostras/sementes e máscara comum de avaliação antes da primeira comparação. Nenhum atributo, escala ou limiar é ajustado com a região externa.
3. Treinar cada braço no conjunto de ajuste. Na regressão linear, `originais_filtradas` compara as 15 combinações não vazias dos quatro fenômenos; `todas_filtradas` faz eliminação reversa de grupos fenômeno×produto (original, magnitude, par seno/cosseno, par conectividade/suporte), aceitando uma retirada apenas se o RMSE na etapa de seleção não piorar. Essa busca é gulosa, não exaustiva. Registrar todos os candidatos e atributos escolhidos. Atributos correlacionados exigem interpretação conjunta. Treinar novamente o modelo com o subconjunto selecionado.
4. Medir os seis braços no mesmo conjunto de validação e na mesma máscara de pixels. Selecionar um vencedor para H0 e outro para H1 pelo menor RMSE de validação, com MAE como primeiro desempate e menor número de atributos como segundo. Não usar resultados da região externa para essa escolha.
5. Congelar modelo, atributos, parâmetros, normalizadores e regra de inferência. Avaliar o vencedor na região externa apenas após a seleção, mantendo a mesma definição temporal: H0 usa inputs e target do mesmo ano; H1 usa inputs do ano anterior para o target seguinte.
6. Salvar artefatos do modelo, previsões georreferenciadas, máscaras, métricas por braço/horizonte/ano e relatório específico. Ao fim dos 11 modelos, comparar os resultados sob o mesmo suporte e protocolo.

## Métricas e interpretação

RMSE é a métrica primária de seleção. Relatar MAE, R², viés médio, percentil 90 do erro absoluto e NMAD dos resíduos, definido como `1.4826 * mediana(|erro - mediana(erro)|)`. Quando o target da amostra for constante, R² é indefinido e deve ser registrado como tal. O NMAD é complementar: muitos zeros podem fazê-lo parecer favorável mesmo com erro grande em hotspots. CE90 mede deslocamento horizontal de coordenadas e não é métrica apropriada de erro de intensidade; não será calculado como se fosse o percentil 90 do erro do target.

Como o objetivo aplicado inclui priorização territorial, relatar a fração da massa do target capturada ao selecionar os 5%, 10% e 20% dos pixels de maior escore, com 10% destacado. Reportar resultados também nos subconjuntos de target zero, positivo e de intensidade alta, com limiar de alta intensidade definido apenas no treino. Incluir referências constantes zero e média do treino; persistência histórica pode aparecer como referência separada porque utiliza informação adicional.

## Teste externo e limites

A região chamada `teste_final` não entra em treinamento, normalização, seleção de atributos, ajuste de hiperparâmetros ou escolha do braço vencedor dos novos modelos. Seus targets estão referenciados no manifesto, fora da pasta de inputs. Ela foi analisada em investigações anteriores; portanto, os novos modelos não terão sido ajustados nela, mas o relatório distinguirá esse fato de um teste confirmatório inteiramente inédito para o processo de pesquisa. Resultados serão reportados mesmo se desfavoráveis.

Todos os relatórios deverão informar contagens, cobertura válida, seeds, versões, parâmetros, hashes das fontes e incerteza espacial por blocos quando aplicável. Pixels adjacentes e anos acumulados não são observações independentes.

## Padrão de relatório final — adendo de 25/09/2026

Após a primeira rodada, o usuário definiu este formato como padrão provisório para os relatórios finais dos próximos modelos. Esta exigência de apresentação foi acrescentada depois do experimento 01; não altera sua regra original de seleção ou as métricas já calculadas.

- Mostrar os resultados dos seis braços de H0 e H1, distinguindo RMSE de seleção, treino após novo ajuste e validação. Mostrar todos os candidatos efetivamente avaliados na filtragem e os canais mantidos/removidos. Candidatos provisórios devem ser separados dos modelos reajustados e validados.
- Para cada teste na área externa do vencedor, registrar horizonte, ano de entrada, ano do target, número e nome exato dos canais usados, pixels comuns, RMSE, MAE, R², viés, NMAD, percentil 90 do erro absoluto, captura top 5/10/20%, métricas por target zero/positivo/alto quando definidas, referências e intervalos exploratórios quando disponíveis. Deixar claro quais braços não receberam teste externo.
- Incluir **uma imagem por par de entrada e target externo** com o target original à esquerda e a previsão à direita. Usar a mesma cobertura válida e a mesma escala de cores nos dois painéis de cada imagem; identificar previsões negativas. Rotular horizonte e anos. Exportar PNG de alta resolução (600 dpi) e manter links aos GeoTIFFs georreferenciados originais, que preservam todos os pixels. A imagem pode ser reduzida para exibição, com essa redução declarada. Inspecionar visualmente cada figura antes de entregar o relatório.

## Experimento 02 — Ridge, decisão anterior à execução

Ridge repete dados, amostras, separações, seis braços, regra do vencedor e teste externo do experimento 01. A nova decisão é a regularização L2. `configs/enriched_ridge.yaml` fixa `lambda_por_amostra` em `[0,000001; 0,00001; 0,0001; 0,001; 0,01; 0,1; 1]`; o estimador usa `alpha = lambda_por_amostra × n_amostras` e padronização ajustada apenas no respectivo ajuste. A busca de lambda é interna, por RMSE da etapa de seleção de camadas, separada para originais, GPA, grafos e todas. Braços filtrados herdam a lambda do seu correspondente completo antes da filtragem. A validação escolhe o vencedor entre os seis braços após novo ajuste; os targets da área externa ficam fora de todas essas decisões. O solver é Cholesky e não há restrição de positividade nas previsões. As escolhas e todas as candidatas serão registradas no relatório. Referências e interpretação constam em `docs/ENRICHED_RIDGE.md`.

## Experimento 03 — HistGradientBoosting, decisão anterior à execução

O experimento 03 repete o catálogo, 24 canais possíveis, seis braços, pares H0/H1, zonas, amostras **finais** de ajuste/validação, máscaras, métricas e regra do vencedor. A perda é quadrática; a busca interna escolhe entre 60, 120 e 180 iterações separadamente nas quatro famílias de entrada não filtradas. Os dois braços filtrados herdam a escolha correspondente. Para viabilizar a busca de muitos candidatos de árvores, ela e a filtragem usam até 20.000 pixels por ano em cada uma das etapas internas de ajuste e seleção, retirados de forma fixa e reprodutível das amostras do protocolo. Os seis modelos finais são então reajustados em todas as amostras de ajuste mais seleção; a validação usa a amostra completa e continua comparável com os experimentos 01/02. `early_stopping=False` evita uma divisão interna aleatória que misturaria blocos ou anos. Nenhum resultado da área externa orienta parâmetros, camadas ou escolha do braço. A configuração e as referências estão em `configs/enriched_hgb.yaml` e `docs/ENRICHED_HGB.md`.

## Experimento 04 — XGBoost, decisão anterior à execução

XGBoost repete os mesmos dados, 24 canais possíveis, seis braços, H0/H1, zonas, amostras finais de ajuste/validação, métricas e regra do vencedor. `XGBRegressor` usa `reg:squarederror`, árvores `hist` em CPU, taxa 0,08, profundidade máxima 4, peso mínimo de folha 80, L2=1, L1=0, `gamma=0`, 128 bins e sem amostragem estocástica de linhas/colunas. A busca interna compara 60, 120 e 180 árvores nas quatro famílias não filtradas; os braços filtrados herdam essa escolha. Para manter a comparação de busca com HGB, até 20.000 pixels por ano em cada etapa interna são sorteados com a mesma semente 4203 e os índices são salvos. Os seis braços finais são reajustados nas amostras completas, e a validação completa decide o vencedor. Não há `eval_set` nem early stopping interno; nada da área externa decide árvores, atributos ou vencedor. A versão XGBoost fica registrada no run; a configuração e as referências estão em `configs/enriched_xgboost.yaml` e `docs/ENRICHED_XGBOOST.md`.
