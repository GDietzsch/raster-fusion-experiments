# Interpretação integrada dos experimentos de regressão raster

**Síntese descritiva, evidências empíricas e fundamentação metodológica — 29 de setembro de 2026**

Este documento reúne os experimentos 01–12, os controles 08A, 09A e 10A–C e a avaliação complementar do target 2021. Os números são extraídos dos artefatos congelados; não foram realizados novos treinamentos, mudanças de vencedores ou ajustes das previsões. A redação distingue resultados observados, interpretações plausíveis e conclusões que os dados ainda não sustentam.

## 1. Síntese dos achados

O principal resultado é a separação entre **capacidade de ajuste e validação no domínio de desenvolvimento** e **capacidade de transferência para outra região**. CNN e ResNet compacta obtiveram os menores erros internos, mas esse destaque não se manteve externamente. A U-Net apresentou os menores RMSE externos entre os 12 modelos principais em quatro dos cinco pares; a Inception venceu o par complementar H1 2020→2021. Essa liderança relativa não significa desempenho operacional satisfatório: nenhum dos 60 testes externos principais apresentou R² positivo, somente três superaram a previsão zero em RMSE e somente um superou a média do treino. Nenhum dos 48 pares principais com persistência disponível superou essa referência em RMSE.

As representações enriquecidas têm utilidade preditiva interna em vários contextos. Usar todas as camadas melhorou o RMSE frente às quatro originais em 11 dos 12 modelos H0 e em dez dos 12 modelos H1. Entretanto, a vantagem depende da arquitetura e do horizonte, e não demonstra melhoria externa de cada representação: somente os vencedores internos receberam os testes externos. Os resultados apoiam estudar o conjunto de variáveis, mas não sustentam que toda camada contribua sempre, que a direção seja dispensável ou que grafos sejam universalmente superiores.

O target deve ser entendido como intensidade espacial dos **registros disponíveis**. Zero representa ausência de registro na fotografia considerada, e não ausência definitiva do fenômeno. Essa definição afeta tanto a interpretação das previsões positivas em pixels zero quanto o significado das referências constantes. O trabalho avaliou regressão de intensidade e priorização espacial; não estimou probabilidades de ocorrência, causalidade ou contagem direta de voos.

## 2. Corpus, rastreabilidade e limites da comparação

A comparação principal contém 12 famílias, 24 vencedores internos, 148 reajustes finais e 60 avaliações externas. Os 148 correspondem a 11 modelos com seis conjuntos por horizonte e Inception com oito. Os controles acrescentam quatro modelos locais, 24 modelos U-Net e 67 avaliações externas, sendo dez dos controles locais e 57 dos controles U-Net. Portanto, esta síntese consolida **176 configurações finais e 127 registros externos**, sem somar como modelos finais as muitas tentativas provisórias de busca e ocultação. Os controles reutilizam referências e não são réplicas independentes de toda a investigação.

Execuções interrompidas ou incompletas de Ridge, XGBoost e MLP não entram no ranking. Os estudos anteriores ao reinício desta rodada foram definidos como investigação pelo pesquisador; não são agregados ao ranking dos experimentos enriquecidos, pois não possuem necessariamente os mesmos canais, orçamento e suporte. Os relatórios 01–12, seus adendos de interpretação, os controles e o complemento de 2021 constituem o corpus desta análise final.

{{INVENTORY}}

A extração usa métricas JSON e configuração registrada, evitando transcrever números arredondados dos relatórios. Foram verificadas unicidade dos pares, contagens, identidade dos hashes dos targets externos e consistência das referências e contagens de pixels. O manifesto [fontes_metricas_sha256.json](fontes_metricas_sha256.json) identifica os arquivos consultados. Esta síntese **não recalcula integralmente os 127 mapas nativos**, não reaudita todos os hashes de modelos e não constitui uma nova auditoria de origem de todos os dados. Auditorias específicas já realizadas permanecem nos relatórios individuais.

Os CSVs completos estão em [todos os ajustes](tabelas/todos_148_ajustes.csv), [testes externos principais](tabelas/externo_60_pares.csv), [vencedores](tabelas/vencedores_24.csv), [controle local](tabelas/controles_locais_externo.csv) e [controles U-Net](tabelas/controles_unet_externo_57.csv). O código [build_final_interpretation.py](../../build_final_interpretation.py) reproduz tabelas, gráficos e documento a partir dos arquivos salvos.

## 3. O que foi previsto e o que as entradas representam

### 3.1 Target, zeros e horizonte temporal

H0 significa estimar o target do mesmo ano das entradas. H1 significa estimar o target do ano seguinte às entradas. Esses rótulos **não designam hipótese estatística nula e alternativa**. H0 representa associação contemporânea; H1 acrescenta transferência temporal. Nenhum dos dois, sozinho, estabelece que as feições causaram os registros observados.

A densidade/intensidade raster é uma representação suavizada dos registros. A resolução de 250 m não implica 250 m de informação independente: pixels vizinhos compartilham a influência das mesmas feições. A contagem de milhões de pixels também não equivale a milhões de eventos independentes. Para os KDEs discutidos no projeto, o raio de 41 km corresponde a 164 pixels; o diâmetro, a 328 pixels. Essa diferença de escalas ajuda a explicar por que afastar apenas três pixels não torna os contextos independentes.

Uma previsão positiva onde o target é zero é erro em relação ao target observado, mas não prova que o local seja impossível ou que ali nunca tenha ocorrido um voo desconhecido. Reciprocamente, não se pode declarar que essa previsão descobriu ocorrências não registradas: seria necessária informação independente. Possíveis diferenças de cobertura de observação, atualização das fontes e disponibilidade de registros devem ser tratadas como hipóteses de qualidade dos dados, não causas já demonstradas.

### 3.2 GPA: nomenclatura e conteúdo efetivamente testado

O nome operacional GPA, mantido nos experimentos, abrange **magnitude e direção do gradiente por diferenças finitas**. Não foram usados G1, G2, triangulação ou uma medida completa de assimetria de padrões. Por fenômeno, magnitude ocupa um canal e direção ocupa dois, seno e cosseno. Assim, quatro fenômenos produzem 12 canais. O uso do par circular evita tratar ângulos próximos de −180° e +180° como numericamente distantes.

A magnitude descreve quão rapidamente a intensidade muda no espaço; a direção descreve para onde ela aumenta. O original representa o nível da intensidade, enquanto os derivados explicitam estrutura local e fronteiras. São transformações do raster e não novas observações independentes. Eles podem facilitar a aprendizagem mesmo sem acrescentar eventos à base. Nas áreas planas, a implementação usa magnitude zero e seno/cosseno zero, distinguindo neutralidade de orientação válida. Fórmulas, unidades, máscara e bordas estão no [documento dos gradientes](../../docs/GRADIENT_INPUTS.md).

### 3.3 Grafos: geometria de proximidade, não rede operacional comprovada

Os grafos foram construídos independentemente para cada fenômeno, ano e região, a partir de fontes de pontos. Cada coordenada distinta é um nó; as arestas unem a escolha de até oito vizinhos dentro de 50 km, com simetrização e peso exponencial da distância. O grau pode exceder oito após simetrizar. A força do nó é a soma dos pesos incidentes. O raster de conectividade interpola forças por vizinhos próximos; suporte indica se existem nós ao alcance.

Esses dois canais por fenômeno não são uma rede aprendida entre garimpos e target. Não usam target para definir arestas, não estabelecem rotas reais e não demonstram ligação operacional entre locais. O raio, a projeção e a troca de vizinhos podem criar transições abruptas. Conectividade zero com suporte um pode representar um nó isolado; conectividade e suporte zero significam ausência de nós próximos segundo a regra adotada.

Há uma ressalva de proveniência: a equivalência exata das feições de pontos usadas nos grafos às que originaram os KDEs não foi demonstrada, principalmente externamente. Portanto, diferenças entre grafos e originais podem refletir **representação e composição das fontes**. A afirmação genérica de que todos os produtos de grafo são transformações determinísticas do raster original não se aplica a esta implementação. O [documento dos grafos](../../docs/GRAPH_INPUTS.md) registra essa limitação. A construção antes da divisão em blocos utiliza contexto espacial de covariáveis de toda a região/ano, sem rótulos; isso precisa ser distinguido de previsão em locais sem qualquer informação contextual disponível.

## 4. Desenho experimental e comparabilidade

### 4.1 Ajuste, seleção, validação e área externa

O protocolo tem três etapas de desenvolvimento: ajuste inicial; seleção de hiperparâmetros/canais; e validação para escolher o conjunto vencedor após reajuste. A área externa recebe somente os modelos congelados. Em H0, ajuste e seleção usam zonas espaciais distintas, e a validação usa a terceira zona nos anos 2020–2022. Em H1, o ajuste usa target 2020, a seleção usa 2021 e a validação usa 2022 na zona espacial de desenvolvimento definida. Assim, H1 interno é predominantemente um teste temporal, e não uma nova região. As amostras finais disponíveis são 600 mil centros de ajuste e 300 mil de validação em H0; 200 mil e 100 mil em H1.

As divisões respondem a perguntas diferentes; um RMSE H0 não pode ser comparado diretamente a um RMSE H1 como se apenas a arquitetura mudasse. A literatura recomenda que a estratégia de avaliação corresponda à estrutura de dependência e ao destino da previsão, em vez de presumir independência entre observações próximas. Essa é a contribuição metodológica de [Roberts et al. (2017)](https://doi.org/10.1111/ecog.02881) para a leitura deste estudo.

O buffer interno de três pixels é aproximadamente 750 m; os blocos de 96 pixels medem 24 km. KDEs de 41 km, grafos de 50 km e halos densos de 56 km atravessam essas escalas. Sobreposição de contexto não demonstra uso indevido de rótulos, mas enfraquece a interpretação da validação como extrapolação para uma região independente. Não foram feitos novos testes para quantificar quanto do ganho se deve a esse contexto compartilhado.

### 4.2 Seis conjuntos comuns e duas adições na Inception

| Conjunto | Conteúdo | Canais antes da seleção |
|---|---|---:|
| Originais | KDEs dos quatro fenômenos | 4 |
| Originais filtradas | Subconjunto das quatro originais | 1–4 |
| GPA | Magnitude e direção seno/cosseno | 12 |
| Grafos | Conectividade e suporte | 8 |
| Todas | Originais + GPA + grafos | 24 |
| Todas filtradas | Subconjunto do conjunto completo | Variável |
| Originais + GPA | Controle de adição, somente Inception | 16 |
| Originais + grafos | Controle de adição, somente Inception | 12 |

Comparar GPA isolado com originais mede **substituição da representação**, e não o ganho marginal de acrescentar GPA. O mesmo vale para grafos. A comparação todas versus originais mede o enriquecimento conjunto; não separa os efeitos de GPA e grafos. Somente a Inception fornece os dois controles explícitos de adição, e não há um conjunto GPA+grafos sem originais para completar todas as combinações possíveis. A matriz experimental não é um fatorial completo comum a todos os modelos.

A Inception escolheu entre oito condições, e os demais modelos entre seis. Nesta execução, seus vencedores pertencem às seis comuns, mas o orçamento de busca continua diferente. Comparar vencedores estima o desempenho do procedimento completo, não um efeito isolado da arquitetura. [Cawley e Talbot (2010)](https://jmlr.org/papers/v11/cawley10a.html) mostram por que a própria seleção sobre uma amostra finita pode introduzir otimismo. Aqui não calculamos correção desse viés nem p-valores de superioridade.

### 4.3 Seleção de canais e orçamentos de treinamento

As famílias tabulares treinam candidatos de subconjuntos nas etapas internas; as redes convolucionais usam ocultação de grupos em um modelo base, seguida de reajuste do subconjunto escolhido. Ocultar um canal padronizado em zero corresponde ao valor médio padronizado, não necessariamente à ausência física do fenômeno. Ocultação não é permutação, e nenhuma das duas equivale automaticamente a uma intervenção causal.

Variáveis relacionadas podem se substituir no ajuste. Por isso, a retirada de um grupo não demonstra irrelevância científica, e a permanência não quantifica contribuição exclusiva. A [documentação de importância com atributos correlacionados](https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html) ilustra o problema de interpretar atributos isoladamente quando compartilham informação; neste projeto essa cautela também deve acompanhar as análises por ocultação e subconjuntos.

Também variam os orçamentos: HGB/XGBoost compararam poucas quantidades de iterações; RF/Extra Trees receberam buscas mais amplas; MLP trabalhou com arquitetura 64→32 e teto de 500 épocas; CNN/ResNet usaram patches esparsos por centro; redes densas usaram núcleos rotativos e até 30 épocas. Isso impede concluir que as configurações encontradas sejam ótimos globais de cada algoritmo.

## 5. Como interpretar as métricas

Se e = previsão − target, RMSE = √média(e²), MAE = média(|e|) e viés = média(e). RMSE enfatiza erros grandes; MAE descreve o erro absoluto médio com menor influência relativa da cauda. Um modelo pode reduzir MAE e aumentar RMSE quando concentra falhas severas em poucos locais. Nenhuma dessas métricas é universalmente superior: a escolha depende do erro que se deseja penalizar. [Hodson (2022)](https://gmd.copernicus.org/articles/15/5481/2022/) fundamenta essa distinção; mantivemos RMSE como critério prévio e usamos MAE como diagnóstico.

R² = 1 − SSE/SST compara o erro do modelo ao erro da constante igual à média do **target avaliado**. R² negativo significa erro quadrático maior que essa constante retrospectiva. Essa média do target externo não foi usada como preditor treinado. Ela não deve ser confundida com a referência média do treino, que é utilizável sem consultar o target externo. No subconjunto de target zero, a variância é nula e R² é indefinido. A definição é apresentada na [documentação de métricas de regressão](https://scikit-learn.org/stable/modules/model_evaluation.html#r2-score).

NMAD = 1,4826 × mediana(|e − mediana(e)|) mede dispersão robusta em torno da mediana do erro. Como remove essa mediana, pode ser pequeno mesmo quando existe deslocamento sistemático relevante; com grande massa de valores iguais pode chegar perto de zero. Portanto, NMAD baixo não compensa viés ou RMSE elevados. P90 do erro absoluto descreve um limiar de erro de intensidade; ele não é CE90, que é uma medida posicional. A base computacional do NMAD é documentada no [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.median_abs_deviation.html).

A captura top 10% soma a massa do target nos 10% de pixels com maior escore e divide pela massa total. Ela mede ordenação/priorização; não é precisão de classificação, porcentagem de voos encontrados nem probabilidade de evento. Uma escolha aleatória uniforme de 10% dos pixels tem captura esperada de 10%, mas uma realização específica pode diferir. Para previsões constantes, o desempate pela ordem dos pixels não é um ranking informativo: não interpretar a captura da referência zero como habilidade territorial real.

Não produzimos uma média global de RMSE entre anos como critério de vitória. Targets com escalas e distribuições distintas tornariam essa média difícil de interpretar. Os gráficos usam comparações por par e, quando apropriado, a razão RMSE_modelo/RMSE_zero. Os CSVs também oferecem skill = 1 − MSE_modelo/MSE_referência; esse skill não substitui o RMSE e não é R² quando a referência é zero ou média do treino.

## 6. Resultados internos: visão completa

### 6.1 Vencedores por modelo e horizonte

{{WINNERS}}

{{FIG_01}}

Em H0, CNN atinge RMSE 2,1842 e R² 0,9249, seguida da ResNet, com 2,6509 e 0,8894. MLP é o melhor modelo tabular por RMSE H0, com 5,4011. Em H1, ResNet lidera com 7,0480, seguida de CNN com 7,2209; entre os tabulares, HGB chega a 10,1787. Essas diferenças são grandes descritivamente, mas se referem à partição e ao procedimento adotados. Não estimamos sua estabilidade em novas divisões para todos os modelos.

U-Net, ResUNet e Inception possuem erros internos próximos dos lineares e R² modestos, sobretudo em H1. O encoder/decoder denso não superou automaticamente a regressão por centro. O regime de supervisão esparsa e exposição rotativa, além das arquiteturas e otimização, também mudou; não é possível atribuir toda a diferença apenas ao nome da rede.

### 6.2 Todos os resultados H0

{{ARMS_H0}}

### 6.3 Todos os resultados H1

{{ARMS_H1}}

As células ausentes representam condições não executadas. Resultados idênticos entre um conjunto completo e o filtrado podem ocorrer quando a filtragem não removeu canais; não constituem confirmações independentes. A U-Net H1 com todas filtradas merece atenção: RMSE 25,9542 contra 15,8147 com todas e 13,5381 com GPA. A seleção de canais não foi monotonicamente benéfica e pode interagir com reajuste e otimização.

{{FIG_02}}

## 7. Interpretação de cada experimento

### 7.1 Experimento 01 — regressão linear

A regressão linear estabelece uma referência aditiva: combina os canais com coeficientes constantes, sem interações aprendidas automaticamente entre eles. O melhor H0 usou todas as 24 entradas; H1, 22. Os RMSE de 7,2989 e 13,1141 são próximos aos obtidos com originais, indicando ganho interno limitado do enriquecimento neste modelo. O conjunto completo H0 reduz apenas cerca de 2,22% o RMSE das originais; em H1, cerca de 1,02%.

O modelo não apresentou boa transferência externa: em 2021, RMSE 3,9949 em H0 e 4,0891 no H1 complementar. Isso delimita a capacidade da relação linear estimada, mas não refuta relações não lineares. Os números de condição registrados nos vencedores foram aproximadamente 6,64 e 6,28, com posto completo. Esses registros não justificam diagnosticar singularidade severa como causa principal dos erros. A correlação entre representações ainda impede uma leitura causal ingênua dos coeficientes. A formulação de mínimos quadrados está na [documentação dos modelos lineares](https://scikit-learn.org/stable/modules/linear_model.html).

### 7.2 Experimento 02 — Ridge

Ridge acrescenta penalização L2, que restringe a magnitude dos coeficientes e altera o compromisso entre ajuste e estabilidade. Nesta implementação, alpha é a penalização por amostra multiplicada pelo número de amostras; os vencedores escolheram λ=0,1 em H0 e 0,01 em H1. Os conjuntos vencedores foram os mesmos tipos da regressão linear: todas e todas filtradas.

O ganho foi pequeno e consistente: RMSE interno 7,2490 e 13,1112; externamente, Ridge teve RMSE menor que Linear nos cinco pares. Em H0 2021, caiu de 3,9949 para 3,8529. A regularização ajudou relativamente, mas não superou as referências simples. Não há base para afirmar que penalização mais forte resolveria a transferência. A interpretação da penalização segue a [formulação Ridge](https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression-and-classification).

### 7.3 Experimento 03 — HistGradientBoosting

Boosting ajusta uma sequência de árvores para melhorar uma função de perda; HGB usa discretização por histogramas para tornar o ajuste eficiente. A referência conceitual é [Friedman (2001)](https://doi.org/10.1214/aos/1013203451). Neste experimento, os vencedores escolheram 180 iterações e todas as camadas. HGB foi o melhor tabular em validação H1: 10,1787, ganho de 24,73% de todas sobre originais.

Esse resultado não persistiu na região externa. H0 2021 teve RMSE 7,6173 e H0 2022, 9,5667, os maiores desses pares entre os 12 modelos principais. O contraste mostra que representar interações no desenvolvimento não garante transportá-las geograficamente. A causa exata — relações locais, distribuição das entradas, regularização ou outros fatores — não foi isolada. Como o melhor candidato atingiu o limite superior da pequena grade de iterações, o ótimo global tampouco foi demonstrado; isso não autoriza usar o teste externo para ampliá-la retrospectivamente.

### 7.4 Experimento 04 — XGBoost

XGBoost também usa boosting, com uma implementação que inclui regularização e controle explícito da construção das árvores. A referência original é [Chen e Guestrin (2016)](https://arxiv.org/abs/1603.02754). A configuração explorada foi restrita: taxa 0,08, profundidade máxima quatro, peso mínimo de folha 80, histogramas em CPU e candidatos de 60, 120 e 180 árvores.

Os vencedores usaram 180 árvores: todas filtradas, 20 canais, em H0; todas, 24, em H1. O H1 interno de 11,1100 ficou acima do HGB e RF. Externamente, XGBoost superou HGB em RMSE nos três pares H0 e em H1 complementar, mas perdeu em H1 2021→2022. Não existe ordenação fixa entre os dois algoritmos para todos os anos e horizontes. Nenhum desses resultados superou zero; a melhora relativa entre modelos não deve ser descrita como validação operacional.

### 7.5 Experimento 05 — Random Forest

Random Forest agrega árvores com aleatoriedade para reduzir dependência entre seus erros; o trabalho de [Breiman (2001)](https://doi.org/10.1023/A:1010933404324) fundamenta essa abordagem. O experimento empregou triagem e refinamento internos, totalizando 152 tentativas, e reajustes finais de 320 árvores. Esse orçamento é mais amplo que a busca dos primeiros modelos.

H0 usou todos os canais e atingiu RMSE 7,0182. H1 selecionou 12 canais e atingiu 10,8151, melhorando sobre todas, com 11,8341. Todos os grupos de direção foram removidos do vencedor H1, mas os quatro fenômenos continuaram representados. Isso indica que, para esse ajuste, magnitude, originais e parte dos grafos bastaram para melhorar o critério; não prova inutilidade geral da direção.

Grafos isolados superaram originais nos dois horizontes. Contudo, os cinco pares externos ficaram acima de zero. Em H0 2022, RF capturou 31,28% da massa no top 10%, o maior valor entre os modelos principais nesse par, apesar de RMSE 5,8433, superior ao da U-Net. Esse desacordo entre intensidade e ranking é um resultado útil, mas não constitui validação independente de uma aplicação de priorização.

### 7.6 Experimento 06 — Extra Trees

Extra Trees aumenta a aleatoriedade na escolha de atributos e pontos de corte, conforme [Geurts, Ernst e Wehenkel (2006)](https://doi.org/10.1007/s10994-006-6226-1). O orçamento de busca e o número final de árvores foram comparáveis aos do RF. A semelhança de desenho torna essa comparação particularmente informativa, embora cada família tenha escolhido seus parâmetros.

Os vencedores usaram 24 canais em H0 e 23 em H1. O ganho da filtragem H1 foi mínimo: de 11,8984 para 11,8816, após retirar magnitude de ocorrências minerais. Não é uma diferença suficiente para declarar que esse canal prejudica o fenômeno ou que o ganho se repetirá em novas amostras. RF obteve menores RMSE internos nos dois horizontes e menores RMSE externos em quatro dos cinco pares; Extra Trees venceu somente H0 2020.

No H1 complementar, a captura de 22,04% superou a do RF (14,88%) e da persistência (17,60%), mas o RMSE continuou alto. A evidência indica utilidade potencial de ordenação em um contexto específico, não superioridade generalizável da aleatorização adicional.

### 7.7 Experimento 07 — MLP

O MLP usa combinações não lineares dos atributos por pixel, sem receber diretamente um patch espacial. A arquitetura final possui duas camadas ocultas, 64→32, e os vencedores escolheram alpha 0,01 e taxa 0,003. O teto de 500 épocas com paciência 15 foi um limite; os reajustes vencedores usaram 37 épocas H0 e 20 H1. A busca parou antes do teto, de modo que falta de autorização para mais épocas não explica por si os resultados.

H0 com 23 canais atingiu RMSE 5,4011, melhor que todos os tabulares desta rodada; H1 com 24 atingiu 11,5090. Entretanto, em H1, a filtragem melhorou a seleção e piorou a validação final, de 11,5090 para 12,6196. O subconjunto escolhido não se transferiu tão bem quanto o conjunto completo.

A análise original documentou caudas importantes de erro externo: em H0 2021, previsão máxima 184,65 contra máximo do target 34,88, com 73,93% do erro quadrático concentrado no 1% de pixels de maior erro. Esses números são diagnósticos anteriores, não recalculados nesta síntese. Eles explicam por que RMSE pode piorar muito sem aumento proporcional do MAE. Não determinam, sozinhos, se a causa é capacidade excessiva, distribuição das entradas ou extrapolação. [Grinsztajn et al. (2022)](https://arxiv.org/abs/2207.08815) e [McElfresh et al. (2023; versão 2024)](https://arxiv.org/abs/2305.02997) mostram que comparação entre redes e árvores depende das propriedades dos dados e da busca; seus benchmarks não predizem o vencedor deste problema espacial.

### 7.8 Experimento 08 — CNN de duas escalas

A CNN combina uma janela local 5×5 com uma grade esparsa 5×5 que alcança ±41 km por eixo. Esse ramo amplo não é um patch denso de 328×328 pixels: ele amostra pontos separados no contexto. Os vencedores usaram 24 canais/25 épocas em H0 e 22 canais/34 épocas em H1. O ganho interno foi forte, chegando ao menor RMSE H0 da rodada, 2,1842.

Externamente, a CNN não superou zero em nenhum dos cinco pares. A discrepância sugere que padrões aprendidos no domínio de desenvolvimento são insuficientes para reproduzir a outra região. O contexto amplo pode facilitar discriminar configurações locais, inclusive a estrutura espacial associada aos KDEs, mas a análise atual não permite afirmar que a rede memorizou coordenadas ou que ocorreu vazamento de target. O controle 08A, discutido adiante, confirma utilidade relativa do ramo sob este orçamento, sem resolver a limitação externa.

### 7.9 Experimento 09 — ResNet compacta

Os atalhos residuais permitem aprender uma transformação acrescida à entrada, estratégia proposta por [He et al. (2016)](https://arxiv.org/abs/1512.03385) para facilitar a otimização de redes profundas. O experimento usa uma adaptação compacta de regressão, e não uma ResNet canônica pré-treinada. A referência original fundamenta a escolha dos atalhos, mas não garante benefício para estes rasters.

A ResNet venceu H0 com grafos isolados (oito canais, 17 épocas, RMSE 2,6509) e H1 com GPA (12 canais, 32 épocas, RMSE 7,0480). Foi o melhor H1 interno, mas seu RMSE externo H1 2021→2022 foi 8,1842, acima de CNN, lineares e U-Net. Já nos três pares H0, a ResNet teve RMSE externo menor que a CNN. A conclusão adequada é uma resposta dependente do horizonte; não uma superioridade geral das conexões residuais.

### 7.10 Experimento 10 — U-Net

A U-Net combina contração, expansão e conexões entre escalas para localizar saídas densas, conforme [Ronneberger et al. (2015)](https://arxiv.org/abs/1505.04597). Aqui a saída é intensidade contínua. O núcleo 512×512 é cercado por halo de 224 pixels, produzindo entrada 960×960; cinco reduções permitem combinar escalas. A perda supervisiona somente os centros autorizados, não todos os targets do patch. As épocas usam núcleos rotativos, e portanto não equivalem a percorrer todos os centros a cada época.

H0 selecionou oito canais e RMSE 7,6149; H1 selecionou GPA e 13,5381. Ambos foram reajustados por 11 épocas. Os erros internos não foram competitivos com CNN/ResNet, mas a U-Net teve os menores RMSE externos entre as famílias principais em H0 2020, 2021, 2022 e H1 2022. Essa inversão do ranking é central: menor RMSE de desenvolvimento não indicou automaticamente melhor transporte regional.

Mesmo assim, apenas H0 2022 superou zero e média do treino, e seu R² foi aproximadamente −0,0010. A U-Net pode ser a melhor entre alternativas insatisfatórias sem ser uma boa estimadora absoluta. Os controles 10A–C mostram que a exposição dos centros e a estabilidade entre sementes importam; não seria correto concluir apenas que a arquitetura densa é mais robusta em geral.

### 7.11 Experimento 11 — ResUNet

A ResUNet combina a estrutura encoder/decoder com blocos residuais, seguindo a ideia de [Zhang, Liu e Wang (2018)](https://arxiv.org/abs/1711.10684), aqui adaptada à regressão. H0 venceu com duas originais de pistas (regulares e irregulares), RMSE 7,4361, quase empatado com todas (7,4370). A diferença de cerca de 0,0008 é descritivamente muito pequena e não constitui evidência estável de superioridade de duas camadas.

H1 selecionou 15 canais, com RMSE 13,6362. Externamente, perdeu para a U-Net nos cinco pares. Os atalhos não trouxeram benefício geral nessa execução. Como canais, largura e época selecionados também diferem, comparar os vencedores não isola causalmente o atalho residual. O resultado deve ser preservado como evidência negativa desta configuração, e não como refutação da família ResUNet.

### 7.12 Experimento 12 — Inception densa

Inception processa caminhos convolucionais paralelos em diferentes escalas; [Szegedy et al. (2015)](https://arxiv.org/abs/1409.4842) e [Szegedy et al. (2016)](https://arxiv.org/abs/1512.00567) fundamentam os ramos e a fatoração de convoluções. A implementação combina quatro ramos em um encoder/decoder denso. Não reproduz GoogLeNet ou Inception-v3 canônica, nem usa pré-treinamento.

Os vencedores usaram apenas originais: quatro canais em H0 e garimpos + pistas regulares em H1, ambos por seis épocas. Na validação H0, adicionar GPA às originais piorou RMSE em 0,66%; adicionar grafos, em 10,57%. Em H1, melhorou em 5,14% e 4,29%, respectivamente; contudo, selecionar duas originais foi melhor. Logo, o efeito dos derivados depende do horizonte e da referência usada.

Externamente, Inception superou U-Net somente no complementar 2020→2021, reduzindo RMSE de 4,5604 para 3,6623. Nesse mesmo par sua captura top 10% foi 11,03%, contra 23,85% da U-Net. A melhoria no erro de intensidade não veio acompanhada de melhoria de ordenação. O custo de aproximadamente sete horas foi o maior da rodada, também por incluir duas condições adicionais; não houve ganho externo geral que justifique chamar esta configuração de superior.

## 8. O que GPA, grafos e filtragem acrescentaram

Os resultados favorecem uma interpretação condicional. GPA isolado superou originais em dez dos 12 H1, mas perdeu nos lineares. Grafos isolados foram especialmente competitivos em RF, XGBoost H1 e ResNet H0. A combinação completa foi frequentemente benéfica, sobretudo em MLP H0, CNN H0 e HGB H1. Porém, U-Net e ResUNet H1 pioraram com todas, e Inception H0 também piorou.

Não se pode decompor o ganho de todas em parcelas exclusivas de GPA e grafos com os seis conjuntos originais. Tampouco inferir que um fenômeno não selecionado deixou de contribuir fisicamente. Os canais de direção são mantidos juntos, assim como conectividade/suporte; estão fortemente vinculados a sua representação. No H1 RF, a direção desaparece, mas no H1 ResNet e U-Net o conjunto GPA inteiro vence. Essa divergência é incompatível com a decisão geral de eliminar direção com base somente no RF.

{{FIG_06}}

O mapa de presença acima mostra seleção, não importância. Modelos com todas selecionam automaticamente todos os fenômenos; contar sua presença como evidência independente de relevância inflaria a conclusão. Mesmo um subconjunto muito pequeno pode substituir sinais correlacionados. Para testar a hipótese de contribuição individual de cada camada, seria necessário um protocolo de retiradas de fenômenos com novo ajuste, sementes/divisões repetidas e avaliação apropriada. As avaliações existentes fornecem indícios preditivos, não uma confirmação universal da hipótese de que cada camada acrescenta informação exclusiva.

O controle 10C também não separa definitivamente magnitude e direção: foi executado apenas em H1, sob plano fixo de núcleos, que cobria menos de um terço dos centros. Magnitude teve melhor média de MAE e captura que direção nesse regime, mas a diferença média de RMSE foi pequena e o GPA completo rotativo permaneceu melhor. A conclusão é manter ambas as representações no registro científico, sem declarar uma indispensável ou dispensável em qualquer cenário.

## 9. Transferência externa, referências e distribuição do target

### 9.1 Mesma região, diferentes fotografias

{{TARGET}}

2020 contém cerca de 5,29% de pixels positivos; 2021, 50,08%; 2022, 59,29%. O ano de 2021 é mais equilibrado apenas na divisão zero/positivo. Isso não torna equilibrada a distribuição das intensidades positivas, nem resolve a dependência espacial. A repetição de desempenho desfavorável em 2021 mostra que a dificuldade externa não se explica apenas pelos muitos zeros de 2020.

A referência zero é forte quando a maior parte da massa está próxima de zero. Ela é necessária para evitar que um modelo complexo pareça útil somente por ter mapas elaborados. Mas não é uma proposta de abandonar a previsão territorial: uma constante não localiza áreas prioritárias. A média do treino mede a vantagem sobre um nível constante que pode ser definido sem consultar o teste. Persistência usa o target anterior e tem informação adicional; em targets acumulados, a sobreposição temporal também ajuda a torná-la competitiva. H1, portanto, não equivale automaticamente a prever apenas novos eventos: o target usado é a fotografia seguinte, não um incremento temporal explicitamente isolado.

### 9.2 Todos os RMSE externos principais

{{EXTERNAL}}

{{FIG_03}}

Somente U-Net, ResUNet e Inception em H0 2022 superaram zero. A U-Net foi a única a superar a média do treino nesse conjunto de 60 pares. Todos os R² externos foram negativos. Isso não é contraditório: em 2022, a média do target externo é uma referência quadrática mais forte que zero, e diferente da média do treino. Ganhar de zero por pouco não garante R² positivo.

A liderança externa da U-Net em quatro pares deve ser apresentada como ranking retrospectivo no domínio observado, e não como escolha validada para qualquer nova região. O complemento 2021 foi acrescentado aos modelos 01–04 após examinar a distribuição; nos posteriores, já fazia parte do protocolo. Todos usaram modelos congelados, mas o destaque desse target continua sendo uma decisão informada pela investigação.

### 9.3 MAE, viés e erro nos zeros

{{EXTERNAL_MAE}}

Os rankings mudam quando o critério passa de RMSE a MAE. Essa mudança não torna um critério errado: mostra que modelos distribuem seus erros de modos diferentes. MLP e HGB podem concentrar falhas grandes em poucos locais; redes densas podem gerar fundos positivos extensos e subestimar a cauda. Examinar apenas médias globais ocultaria essas diferenças.

{{FIG_05}}

Nos mapas e nos registros de subconjuntos, há erro relevante tanto em pixels zero quanto nos positivos. A proporção de zeros, isoladamente, não determina o desempenho. Os limiares de alta intensidade foram definidos no treino, aproximadamente 74,31 em H0 e 85,35 em H1; como excedem o máximo dos targets externos avaliados, o subconjunto alto externo fica vazio. Isso deve ser reportado como não aplicável, e não como erro zero ou ausência de dificuldade nos maiores valores locais.

### 9.4 Incerteza e limites do número de pixels

{{FIG_09}}

Os intervalos já registrados usam reamostragem espacial exploratória de 42 blocos ocupados de aproximadamente 96 km. Eles descrevem a variação sob esse esquema no domínio observado, não a variabilidade entre regiões novas. Sobreposição de intervalos individuais não é teste de equivalência; ausência de sobreposição tampouco substitui uma análise pareada da diferença. Não calculamos testes de significância entre os 12 modelos a partir de milhões de pixels tratados como independentes.

O histórico de uso da região externa importa. Os modelos desta rodada foram ajustados sem seus targets, mas o processo de pesquisa já conhecia a região e foi conduzido com leituras sucessivas de resultados. A distinção entre modelo não ajustado no teste e processo de pesquisa inteiramente cego ao teste é essencial para evitar uma conclusão confirmatória excessiva, coerente com o problema de seleção discutido por [Cawley e Talbot](https://jmlr.org/papers/v11/cawley10a.html).

## 10. Priorização territorial: um objetivo diferente

{{CAPTURE}}

Os valores da tabela são porcentagens da massa do target no top 10% da área válida. Diversos pares ficam acima dos 10% esperados para uma seleção aleatória, indicando associação útil para ordenação naquela amostra. Porém, essa comparação não recebeu uma distribuição nula específica que respeite a estrutura espacial; não a tratamos como teste de significância.

{{FIG_04}}

Em H0 2022, RF lidera a captura entre os modelos com 31,28%, mas perde para U-Net em RMSE. Em H1 2022, HGB captura 30,73%, enquanto U-Net tem menor RMSE. A persistência captura aproximadamente 41,14% nesse target, além de ter menor erro, mas usa informação histórica adicional. No H1 complementar, U-Net captura 23,85%, embora Inception tenha menor RMSE.

Esses contrastes permitem formular duas aplicações futuras distintas: previsão da intensidade e triagem territorial. A segunda exige definir custo de cobertura, custo de falsos alertas, disponibilidade temporal das fontes e validação independente da utilidade. Não é adequado escolher um modelo pelo RMSE e depois declarar sucesso operacional com base somente na métrica de captura que lhe foi mais favorável. Os resultados de ranking são promissores em alguns pares, porém ainda exploratórios.

## 11. O que os controles esclareceram

### 11.1 Controles 08A e 09A: retirar a escala ampla

{{LOCAL}}

Retirar o ramo amplo piorou substancialmente a validação da CNN e da ResNet nos dois horizontes. Na CNN, também piorou RMSE nos cinco pares externos. Na ResNet, melhorou somente H0 2020 e piorou os outros quatro. Portanto, a amostragem ampla contribuiu para as configurações testadas; removê-la não resolveu a transferência regional.

Esse contraste não isola apenas distância espacial: a retirada reduz parâmetros, e os canais centrais já incorporam o contexto dos KDEs ou dos grafos. Os controles mantiveram os vencedores, canais e épocas anteriores, sem uma nova busca otimizada para as redes locais. A conclusão vale para essa intervenção sob orçamento fixo, não para a melhor rede local possível.

### 11.2 Controles 10A–C: exposição dos dados, profundidade e componentes GPA

{{UNET_CONTROLS}}

{{FIG_07}}

10A demonstrou que curvas mais suaves não garantem melhor generalização. O plano fixo cobriu apenas 28,40% dos centros H0 e 29,36% dos H1. Em H1, apesar de fração positiva semelhante, o máximo do target nesse plano caiu de 398,47 para 122,07. A representação dos extremos mudou, explicando por que a fração zero/positivo é diagnóstico insuficiente de cobertura. Rotação melhorou a média e reduziu a dispersão entre as três sementes.

10B reduziu cinco para três níveis sob plano fixo. Piorou H0 e melhorou H1 frente à referência fixa instável; não superou o H1 rotativo completo. Contexto, capacidade e profundidade mudaram juntos, e a rede rasa com rotação não foi testada. Não há evidência de que reduzir profundidade seja solução geral.

10C comparou magnitude e direção isoladas sob o mesmo regime fixo. Magnitude teve RMSE médio 14,8390, direção 15,0220; o GPA completo rotativo teve 13,8501. O experimento não permite atribuir a diferença unicamente ao valor informativo da direção fora daquele plano de exposição. As médias são médias de métricas de três redes; não representam um ensemble e os desvios padrão não são intervalos de confiança.

Os 57 pares externos dos controles U-Net reforçam a limitação: apenas cinco superaram zero; nenhum dos 48 com persistência disponível a superou. Duas execuções tiveram R² externo ligeiramente positivo, mas não foram escolhidas retrospectivamente como novas vencedoras. Esse resultado não contradiz os R² negativos dos 60 pares principais: são conjuntos de experimentos distintos.

## 12. Curvas, épocas e custo computacional

Época não é uma unidade universal entre modelos. Linear e Ridge resolvem um problema de otimização diretamente; HGB e XGBoost somam etapas de boosting; RF e Extra Trees agregam árvores, cuja contagem não equivale a épocas. Curvas por prefixo reconstruídas dos modelos salvos são diagnósticos posteriores, não históricos medidos durante o ajuste e não foram usadas para alterar vencedores.

MLP e redes convolucionais registram perda por época e desempenho na seleção quando aplicável. Nas redes densas, os núcleos de treino mudam por rotação; oscilações da perda refletem tanto aprendizagem quanto mudança de amostra. Na seleção fixa, picos também ocorreram, então não é correto atribuir toda oscilação apenas à mudança dos núcleos. Tampouco é possível diagnosticar capacidade excessiva somente comparando uma perda padronizada de treino a um RMSE de seleção em unidades originais.

O teto de 500 épocas do MLP não foi alcançado. Nos vencedores densos, o reajuste usou de seis a 11 épocas nesta rodada; alguns candidatos Inception alcançaram o teto de 30 durante a busca. Reajustes finais percorrem a época escolhida internamente e não fazem seleção adicional pelo teste externo. O uso de AdamW nas redes convolucionais segue a separação entre atualização adaptativa e decaimento dos pesos discutida por [Loshchilov e Hutter (2019)](https://arxiv.org/abs/1711.05101); isso não elimina a necessidade de selecionar a intensidade da regularização.

{{FIG_08}}

A duração é custo observado de cada execução inteira, não uma medida normalizada da eficiência intrínseca do algoritmo. Inclui leituras, busca, reajustes, inferência e geração de arquivos; hardware, caches, quantidade de canais, candidatos e condições diferem. Inception custou cerca de sete horas; U-Net e ResUNet, aproximadamente quatro horas cada. HGB foi muito mais rápido, mas teve transferência desfavorável; custo menor ou maior não determinou qualidade externa.

A literatura recente oferece comparação de famílias e normalizações em protocolos organizados, como [TALENT, Liu et al. (2025)](https://jmlr.org/papers/v26/25-0512.html). Sua relevância aqui é metodológica: documentar orçamento, representação e avaliação. Não é evidência de que uma arquitetura tabular ainda não testada resolveria automaticamente o problema espacial deste projeto.

## 13. Comparação visual em escala comum

As figuras seguintes reúnem target e previsões dos 12 modelos para 2021, escolhido porque possui cerca de metade dos pixels positivos e foi avaliado em H0 e H1. Os painéis usam escala comum dentro de cada figura e unidades originais; azul representa intensidades negativas, mantidas sem truncamento. A escala simétrica/logarítmica permite visualizar simultaneamente valores pequenos e extremos, mas distâncias cromáticas não correspondem a diferenças lineares.

{{MAPS}}

As prévias são reduzidas por vizinho mais próximo para lado maior de até 500 pixels. Elas ilustram a estrutura espacial geral e não substituem os GeoTIFFs ou os mapas de 600 dpi dos relatórios individuais. As métricas da síntese usam registros calculados na resolução nativa. Ausência de um pequeno extremo na prévia não demonstra sua ausência no raster.

Os mapas permitem confrontar focos, fundos e contornos, mas não identificar causalmente a origem de cada padrão. Previsões podem apresentar estruturas visualmente parecidas com as covariáveis e ainda assim errar localização ou amplitude do target. Da mesma forma, um mapa mais suave pode reduzir RMSE sem concentrar tão bem a massa nas áreas prioritárias. A interpretação precisa combinar mapas, métricas globais, subconjuntos e finalidade de uso.

## 14. O que podemos afirmar sobre a hipótese do estudo

| Afirmação | Situação após os experimentos | Base e limite |
|---|---|---|
| As entradas contêm sinal preditivo no desenvolvimento | Sustentada descritivamente | R² interno positivo dos vencedores; ganhos sobre originais em vários modelos |
| Cada fenômeno tem contribuição exclusiva e estável | Não demonstrada | Seleção depende de modelo/horizonte; correlação e fontes impedem atribuição exclusiva |
| GPA melhora sempre | Não sustentada | Melhora vários H1, mas perde em outros modelos/condições |
| Grafos melhoram sempre | Não sustentada | Benefícios específicos; proveniência e representação podem mudar conjuntamente |
| Combinar todos os canais é sempre melhor | Refutada como regra universal nesta rodada | U-Net/ResUNet H1 e Inception H0 fornecem contraexemplos |
| Mais complexidade arquitetural melhora a transferência | Não sustentada | ResUNet/Inception não superam U-Net de modo geral; ranking interno muda externamente |
| O modelo reproduz bem a intensidade em outra região | Não sustentada pelas configurações atuais | R² externo negativo nos 60 pares principais; poucas vitórias sobre referências |
| Alguns escores podem auxiliar priorização | Indício exploratório | Captura acima da cobertura em vários pares; falta validação operacional independente |
| Zero significa ausência definitiva do fenômeno | Incompatível com a definição do target | Zero registra ausência de observação naquela fotografia |

Uma formulação estatística futura da hipótese nula poderia ser: acrescentar determinado grupo de entradas não melhora a perda esperada no domínio de interesse, sob um procedimento de ajuste definido. A alternativa seria redução dessa perda. A presente síntese não testa formalmente essas hipóteses: faltam replicações comparáveis e uma avaliação independente desenhada para as diferenças. Por isso, H0/H1 continuam sendo usados exclusivamente como horizontes, e os ganhos são tratados como descritivos.

## 15. Limitações e encaminhamento científico

O limite mais importante é a avaliação em uma única região externa já examinada. As imagens e métricas são suficientes para documentar a dificuldade de transferência dessas configurações, mas não para estimar desempenho em toda a população de regiões possíveis. A repetição de tentativas e a leitura sucessiva do teste exigem interpretar a comparação global como exploratória.

Também permanecem incertezas sobre equivalência das fontes dos grafos/KDEs, cobertura de observação do target, estabilidade entre sementes fora dos controles e efeitos do contexto compartilhado nas divisões internas. Não foi demonstrado um tipo específico de mudança de distribuição: a diferença observada pode envolver covariáveis, intensidade do target, relações condicionais ou mais de um desses componentes. Usar genericamente a expressão mudança de domínio descreve o problema de transferência, mas não resolve seu diagnóstico.

Esta rodada já permite uma interpretação geral consistente; não é necessário acumular arquiteturas para relatá-la. Se houver uma etapa confirmatória futura, a prioridade seria definir o destino da previsão, auditar a comparabilidade das fontes e avaliar poucas hipóteses pré-especificadas com replicação espacial/temporal apropriada. Separar ocorrência registrada de intensidade positiva, modelar incrementos temporais ou testar magnitude/direção sob rotação são possibilidades de novo protocolo, não resultados estabelecidos nem experimentos iniciados nesta análise.

**Conclusão científica:** as camadas enriquecidas e o contexto espacial melhoram a predição interna em várias configurações, mas esse ganho não se converteu em reprodução confiável da intensidade na região externa. A contribuição observada é dependente do modelo e horizonte. O principal avanço do conjunto de experimentos é mostrar onde há sinal preditivo, onde a seleção é instável e onde falha a transferência, delimitando uma próxima investigação mais focalizada em dados, domínio e objetivo de avaliação.

## 16. Apêndice: canais e hiperparâmetros dos vencedores

### 16.1 Canais exatos

{{WINNER_CHANNELS}}

O inventário de presença de todos os canais, inclusive nos conjuntos não vencedores, está em [presenca_canais.csv](tabelas/presenca_canais.csv). Esses registros permitem reconstruir as retiradas comparando cada conjunto filtrado com sua família completa; os relatórios individuais preservam todas as tentativas de seleção.

### 16.2 Hiperparâmetros e orçamento final

{{PARAMS}}

Traço indica campo não aplicável ou não registrado nesse formato de artefato, e não valor zero. Os JSONs e os arquivos config.yaml de cada execução são as fontes completas, incluindo parâmetros fixos não repetidos na tabela. A coluna de épocas descreve o reajuste final, não o total de épocas gasto por todos os candidatos da busca.

### 16.3 Viés externo em todos os pares

{{BIAS}}

Viés positivo significa superestimação média; negativo, subestimação. Viés próximo de zero pode coexistir com grandes erros que se compensam. Os CSVs externos preservam também NMAD, P90, captura em outras coberturas, previsões negativas e erros dos subconjuntos disponíveis.

## 17. Relatórios de origem

{{SOURCES}}

- [Controle 08A — CNN local](../../runs/enriched_cnn_local_control/20260927T010521698427Z/RELATORIO.md).
- [Controle 09A — ResNet local](../../runs/enriched_resnet_local_control/20260927T181122308509Z/RELATORIO.md).
- [Controles 10A–C — síntese e relatórios](../../runs/enriched_unet_controls/20260928T005339448925Z/ANALISE_10ABC.md).
- [Complemento do target 2021, modelos 01–04 congelados](../target_2021_frozen_models/20260926T162434415225Z/RELATORIO.md).
- [Protocolo comum](../../protocols/EXPERIMENTOS_ENRIQUECIDOS_2026.md), [gradientes](../../docs/GRADIENT_INPUTS.md) e [grafos](../../docs/GRAPH_INPUTS.md).

## 18. Referências bibliográficas e técnicas

As referências fundamentam os conceitos; os números experimentais são provenientes dos arquivos locais. Artigos clássicos foram mantidos por serem as fontes originais dos métodos. Trabalhos de 2022–2025 contextualizam avaliação e comparação de famílias. Fontes consultadas em 29/09/2026; a documentação on-line pode corresponder a versões diferentes das bibliotecas congeladas nos experimentos.

- Breiman, L. (2001). Random Forests. Machine Learning, 45, 5–32. [DOI](https://doi.org/10.1023/A:1010933404324).
- Cawley, G. C.; Talbot, N. L. C. (2010). On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation. JMLR, 11, 2079–2107. [Artigo](https://jmlr.org/papers/v11/cawley10a.html).
- Chen, T.; Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD. [Artigo](https://arxiv.org/abs/1603.02754).
- Friedman, J. H. (2001). Greedy Function Approximation: A Gradient Boosting Machine. Annals of Statistics, 29(5), 1189–1232. [DOI](https://doi.org/10.1214/aos/1013203451).
- Geurts, P.; Ernst, D.; Wehenkel, L. (2006). Extremely randomized trees. Machine Learning, 63, 3–42. [DOI](https://doi.org/10.1007/s10994-006-6226-1).
- Grinsztajn, L.; Oyallon, E.; Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on tabular data? NeurIPS Datasets and Benchmarks. [Artigo](https://arxiv.org/abs/2207.08815).
- He, K.; Zhang, X.; Ren, S.; Sun, J. (2016). Deep Residual Learning for Image Recognition. CVPR; preprint 2015. [Artigo](https://arxiv.org/abs/1512.03385).
- Hodson, T. O. (2022). Root-mean-square error (RMSE) or mean absolute error (MAE): when to use them or not. Geoscientific Model Development, 15, 5481–5487. [Artigo](https://gmd.copernicus.org/articles/15/5481/2022/).
- Liu, S.-Y. et al. (2025). Talent: A Tabular Analytics and Learning Toolbox. JMLR, 26(226), 1–16. [Artigo](https://jmlr.org/papers/v26/25-0512.html).
- Loshchilov, I.; Hutter, F. (2019). Decoupled Weight Decay Regularization. ICLR; preprint 2017. [Artigo](https://arxiv.org/abs/1711.05101).
- McElfresh, D. et al. (2023; versão atualizada 2024). When Do Neural Nets Outperform Boosted Trees on Tabular Data? NeurIPS Datasets and Benchmarks. [Artigo](https://arxiv.org/abs/2305.02997).
- Roberts, D. R. et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. Ecography, 40, 913–929. [DOI](https://doi.org/10.1111/ecog.02881).
- Ronneberger, O.; Fischer, P.; Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI. [Artigo](https://arxiv.org/abs/1505.04597).
- Szegedy, C. et al. (2015). Going Deeper with Convolutions. CVPR; preprint 2014. [Artigo](https://arxiv.org/abs/1409.4842).
- Szegedy, C. et al. (2016). Rethinking the Inception Architecture for Computer Vision. CVPR; preprint 2015. [Artigo](https://arxiv.org/abs/1512.00567).
- Zhang, Z.; Liu, Q.; Wang, Y. (2018). Road Extraction by Deep Residual U-Net. IEEE Geoscience and Remote Sensing Letters; preprint 2017. [Artigo](https://arxiv.org/abs/1711.10684).
- Scikit-learn. Modelos lineares, métricas de regressão e interpretação com atributos correlacionados. [Modelos](https://scikit-learn.org/stable/modules/linear_model.html), [métricas](https://scikit-learn.org/stable/modules/model_evaluation.html), [correlação e importância](https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html).
- SciPy. Median absolute deviation. [Documentação](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.median_abs_deviation.html).
