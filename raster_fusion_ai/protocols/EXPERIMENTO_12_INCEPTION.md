# Experimento 12 — Inception densa com oito conjuntos de entrada

Protocolo definido antes da execução. Perguntas: processar simultaneamente diferentes escalas em blocos convolucionais melhora a regressão raster? Acrescentar produtos GPA ou de grafos às camadas originais melhora a previsão? Target zero significa ausência de registro naquela fotografia, não impossibilidade do fenômeno.

## Arquitetura e fundamentação

Rede encoder/decoder para produzir intensidade contínua por pixel. Manter cinco reduções por fator dois, expansão bilinear e concatenações entre escalas, com saída linear, BatchNorm e dropout zero. Substituir os blocos convolucionais simples por blocos com quatro ramos paralelos:

1. Convolução 1×1.
2. Convolução 1×1 seguida de 3×3.
3. Convolução 1×1 seguida de duas 3×3 consecutivas.
4. Max-pooling 3×3, stride 1, seguido de convolução 1×1.

Cada ramo produz `max(1, largura_do_bloco // 4)` canais. Concatenar os quatro ramos e aplicar fusão 1×1 para retornar à largura do bloco. Cada convolução recebe BatchNorm e ReLU. Não usar atalhos residuais. As larguras por nível são 8,16,32,64,128,256 ou 16,32,64,128,256,512, conforme o candidato. O relatório registra parâmetros efetivos; igual largura não implica igual capacidade/custo entre famílias.

Szegedy et al. (2015; preprint 2014) fundamentam o processamento multiescala paralelo. Szegedy et al. (2016; preprint 2015) discutem fatoração de convoluções; duas 3×3 têm campo receptivo convolucional 5×5, mas não são numericamente equivalentes a uma única 5×5, pois incluem não linearidades. Esta é uma **adaptação inspirada em Inception com decoder e saída densa**, não GoogLeNet ou Inception-v3 canônica, nem Inception-ResNet. Não usa pré-treinamento de classificação.

Entrada 960×960, núcleo 512×512, halo 224 por lado; resolução 250 m. O núcleo mede 128 km e o halo 56 km. Um filtro pequeno não abrange sozinho o raio KDE de 41 km: a dependência espacial decorre do caminho completo, reduções e expansões. A geometria é mantida para comparar com os experimentos 10/11; validar as emendas com inferência sintética. BatchNorm usa estatísticas espaciais no treino, então localidade convolucional de inferência não caracteriza toda a dependência durante o ajuste.

## Oito conjuntos por horizonte

| Conjunto | Conteúdo | Número de canais |
|---|---|---:|
| originais | Quatro fenômenos originais | 4 |
| originais_filtradas | Subconjunto selecionado das originais | 1–4 |
| gpa | Magnitude, seno e cosseno da direção dos quatro fenômenos | 12 |
| grafos | Conectividade e suporte dos quatro fenômenos | 8 |
| todas | Originais + GPA + grafos | 24 |
| todas_filtradas | Subconjunto selecionado de todos os produtos | variável |
| originais_gpa | Originais + GPA, sem grafos | 16 |
| originais_grafos | Originais + grafos, sem GPA | 12 |

Repetir em H0 (inputs do mesmo ano) e H1 (ano anterior): **16 braços finais**. H0/H1 designam horizontes temporais, não testes estatísticos de hipótese nula/alternativa. Os seis primeiros reproduzem os conjuntos anteriores; os dois adicionais atendem à comparação com/sem os grupos derivados. Os produtos chamados GPA são os canais de gradiente já congelados, não novos índices G1/G2.

Relatar deltas de RMSE de validação: originais→originais+GPA; originais→originais+grafos; originais+GPA→todas (acréscimo de grafos); originais+grafos→todas (acréscimo de GPA). Delta negativo favorece a adição. Cada família seleciona seus hiperparâmetros/época, portanto os deltas avaliam o desempenho do procedimento com cada conjunto, não um efeito causal puro do canal. Produtos são derivados das camadas originais: não demonstram informação observacional independente só por melhorarem a representação. Não declarar significância a partir de uma semente.

## Treino, busca e seleção

Reutilizar os centros/labels congelados do MLP e os caches alinhados do experimento 08 somente para leitura. Preservar divisão espacial/temporal, máscara, targets, normalização treinada no respectivo conjunto de ajuste e perda MSE apenas nos centros autorizados. Nenhum target vizinho entra como input.

Manter rotação de até 64 núcleos por ano por época, semente 4210. Seleção e ocultação em 32 núcleos fixos por ano; validação final em todos os centros congelados (H0 300 mil, H1 100 mil). Oscilações na perda de treino também refletem a mudança do suporte. Não adotar o subconjunto fixo limitado dos controles 10A–C.

Buscar 8 ou 16 filtros iniciais para as seis famílias não filtradas. AdamW, taxa 0,001, weight decay 0,01, até 30 épocas, mínimo cinco e paciência seis épocas elegíveis sem melhora do critério (RMSE, MAE, época). São **24 tentativas de hiperparâmetros**: seis famílias × dois candidatos × dois horizontes. O orçamento por família é mantido, mas o orçamento total aumenta com os dois novos conjuntos.

Para originais filtradas, avaliar as 15 combinações não vazias de fenômenos por ocultação no modelo base. Para todas filtradas, testar remoções individuais de grupos fenômeno×produto e a remoção conjunta das favoráveis quando válida. Isso não corresponde a treinar todas as combinações. Reajustar do zero os subconjuntos escolhidos, herdando parâmetros/época da respectiva base; registrar todas as tentativas e canais mantidos/removidos.

Reajustar os 16 modelos finais do zero em ajuste+seleção, pela época escolhida internamente. Selecionar um vencedor por horizonte pelo menor RMSE completo de validação, depois MAE, menos canais e nome do braço como desempate determinístico. Escolher entre oito conjuntos amplia as oportunidades de seleção em relação aos seis dos experimentos anteriores; interpretar comparações por conjuntos comuns além dos vencedores. Uma semente por configuração preserva o orçamento anterior, mas não estima estabilidade. Não escolher parâmetros por desempenho externo.

## Teste externo e relatório

Após congelar ambos os vencedores, avaliar H0 2020→2020, 2021→2021, 2022→2022; H1 2021→2022 e complementar 2020→2021. São **cinco pares externos**, somente dos vencedores; comparações entre os oito conjuntos são feitas na validação interna. A região externa já foi examinada e é exploratória. O buffer interno de três pixels não separa contextos amplos, limitação preservada.

Registrar RMSE, MAE, R², viés, NMAD, P90 absoluto, captura top 5/10/20%, erros nos subconjuntos zero/positivo/alto, referências zero/média do treino/persistência, bootstrap espacial exploratório. NMAD é dispersão robusta dos resíduos e não substitui RMSE; CE90 posicional não se aplica à intensidade. Não normalizar pelo target externo, não truncar negativos. Persistência utiliza informação histórica adicional.

Relatório com 16 resultados finais, 24 buscas, todas as ocultações, canais usados/removidos, parâmetros, épocas realizadas/escolhidas, passos, curvas e os deltas dos grupos GPA/grafos. Curvas de busca mostram os candidatos escolhidos por família; o CSV guarda todas as épocas de todos os candidatos. Cinco mapas target/previsão a 600 dpi com escala comum dentro de cada par e GeoTIFFs na resolução/georreferência original. A inspeção visual será feita após a execução, sem alegar antecipadamente a qualidade dos mapas.

## Execução e retomada

Entrada `run_enriched_inception.py`, configuração `configs/enriched_inception.yaml`, saída separada `runs/enriched_inception/<timestamp>`. Salvar código, configuração, ambiente, hashes, planos de seleção, históricos, checkpoints e métricas. O nome interno `unet` no YAML identifica o motor compartilhado de regressão densa; `model_family: inception` e `block_type: inception` definem explicitamente a arquitetura usada e os artefatos.

Repetir a mesma linha retoma a última execução incompleta, conferindo configuração e código; uma época interrompida é refeita a partir do último checkpoint. Uma execução completa não é repetida automaticamente; `--new-run` solicita outra deliberadamente. O bloqueio evita duas execuções simultâneas na mesma saída. Solicitar ao Windows que evite suspensão automática durante o processo; manter tomada e terminal aberto. Não impede suspensão manual ou por fechamento da tampa.

## Verificação antes da execução — 28/09/2026

A conferência de configuração e catálogo passou: 195 entradas/targets e 24 canais, sem abrir targets externos nessa etapa. A suíte local passou em 26 testes, incluindo ajuste com supervisão esparsa, retomada de checkpoint, oito conjuntos, escrita de GeoTIFF com máscara comum e geração do relatório com 16 resultados e oito contrastes. Os testes também verificaram compatibilidade dos caminhos U-Net, controles e ResUNet.

Um teste sintético na GPU executou forward/backward com 24 canais, 16 filtros iniciais e entrada 960×960: pico de memória alocada pelo PyTorch de 3,19 GiB, sem incluir toda a memória reservada ou outros processos. A comparação sintética de inferência inteira versus blocos com halo de 224 pixels teve diferença máxima zero, com TF32 de cuDNN desativado nessa verificação. Isso verifica a implementação e não antecipa resultados científicos, tempo total ou estabilidade entre sementes. O treinamento experimental completo não foi iniciado durante a preparação.

## Referências

- [Szegedy et al. (2015), Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842), preprint 2014: processamento paralelo multiescala.
- [Szegedy et al. (2016), Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567), preprint 2015: fatoração e eficiência computacional.
- [Ronneberger et al. (2015), U-Net](https://arxiv.org/abs/1505.04597): decoder e concatenações entre escalas da estrutura adaptada.
- [Ioffe e Szegedy (2015), BatchNorm](https://arxiv.org/abs/1502.03167).
- [Loshchilov e Hutter (2019), AdamW](https://arxiv.org/abs/1711.05101).
- [Valavi et al. (2019), blockCV](https://doi.org/10.1111/2041-210X.13107): avaliação espacial.
- [scikit-learn, métricas de regressão](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics); [SciPy, desvio absoluto mediano](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.median_abs_deviation.html).
- [Protocolo do experimento 10](EXPERIMENTO_10_UNET.md), [protocolo comum](EXPERIMENTOS_ENRIQUECIDOS_2026.md), [gradientes](../docs/GRADIENT_INPUTS.md) e [grafos](../docs/GRAPH_INPUTS.md): construção das entradas e regras compartilhadas.
