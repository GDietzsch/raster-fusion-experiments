# Experimento 11 — ResUNet para regressão raster

Protocolo definido antes do treino. Pergunta: conexões residuais dentro dos blocos podem facilitar a otimização e melhorar a previsão, em relação à U-Net do experimento 10? Não presumir superioridade nem interpretar resultados exploratórios como confirmação causal.

## Arquitetura e justificativa

Preservar cinco reduções por dois, convoluções 3×3, BatchNorm, expansão bilinear, concatenações encoder/decoder, dropout zero e saída linear para intensidade. Em cada um dos onze blocos (seis no encoder incluindo o bottleneck e cinco no decoder), adicionar uma projeção aprendida 1×1 da entrada à saída do caminho de duas convoluções. Implementação: `ReLU(F(x) + P(x))`, com `F = Conv3×3 → BN → ReLU → Conv3×3 → BN → ReLU` e `P = Conv1×1`. A projeção é usada em todos os blocos para compatibilizar canais. As somas dentro dos blocos são distintas das concatenações entre escalas.

He et al. (2016) fundamentam a aprendizagem residual. Zhang, Liu e Wang (2018; preprint 2017) combinam unidades residuais e U-Net para extração de estradas em imagens aéreas. Aqui utilizamos uma **adaptação própria para regressão contínua**, não uma reprodução exata do bloco/arquitetura dos artigos, nem pesos pré-treinados. A posição das ativações está explicitada acima. A inclusão de projeções adiciona parâmetros, contabilizados no relatório.

Entrada 960×960; núcleo de saída 512×512 e halo 224 pixels (56 km por lado a 250 m). Halo, fase das reduções e resolução são os mesmos da U-Net. Atalhos 1×1 não ampliam o campo receptivo máximo convolucional. A consistência entre inferência em núcleos adjacentes e em uma grade maior deve ser verificada em teste sintético. Estatísticas espaciais de BatchNorm no treino não equivalem à localidade na inferência.

## Dados, treinamento e escolha

Usar todos os 24 canais possíveis nos respectivos conjuntos: originais; originais filtradas; GPA (magnitude/seno/cosseno); grafos; todas; todas filtradas. Repetir os seis braços em H0 e H1. H0 significa mesmo ano e H1 ano anterior; não são nomes de testes estatísticos nulo/alternativo. Target zero significa ausência de registro naquela fotografia, não impossibilidade do fenômeno.

Reutilizar somente para leitura os caches alinhados do experimento 08 e os centros/labels congelados do MLP. Preservar grade, divisão espacial/temporal, máscara e hiperparâmetros de amostragem. Ajustar a normalização exclusivamente no conjunto usado no respectivo treino. A perda MSE usa apenas os centros autorizados, não todos os pixels da superfície densa.

Manter **rotação de até 64 núcleos por ano por época**, semente 4210 e 32 núcleos fixos por ano para seleção/ocultação, como no experimento 10. Os controles 10A–C mostraram piora do subconjunto fixo testado e menor cobertura de centros; portanto não o adotamos. A rotação amplia cobertura ao longo das épocas; não garantir visita a todos os centros em treinos curtos. A perda de treino é medida em suportes que mudam por época, então suas oscilações não medem apenas mudança de pesos.

Busca por família base: 8 ou 16 filtros iniciais, AdamW, taxa 0,001, weight decay 0,01. Até **30 épocas**, mínimo cinco, paciência seis épocas elegíveis sem melhora no critério (RMSE, MAE, época). São 16 tentativas de hiperparâmetros (quatro famílias × dois candidatos × dois horizontes). A época é escolhida na seleção interna, antes do reajuste final. Não adotar as 11 épocas fixas dos controles 10A–C: aquela era uma comparação de orçamento fixo dos vencedores; aqui voltamos à busca do experimento 10.

Originais filtradas: pontuar as 15 combinações não vazias dos quatro fenômenos por ocultação. Todas filtradas: ocultar grupos fenômeno×produto, avaliar remoções individuais e a combinação das favoráveis quando válida. Ocultação mede dependência preditiva, não importância causal, e não equivale a treinar todas as combinações. Os dois braços filtrados herdam parâmetros/época da família base, são treinados do zero e também avaliados antes do reajuste final. Não alterar limiares com base no teste externo.

Reajustar cada um dos 12 braços do zero nos centros de ajuste+seleção por sua época escolhida. Calcular métricas nos centros completos de treino e validação. Escolher por horizonte menor RMSE de validação, depois MAE, menos canais e nome do braço como desempate determinístico. Uma semente por candidato mantém o orçamento do experimento 10; estabilidade entre sementes não será estimada nesta rodada. Como parâmetros, canais e épocas dos vencedores podem mudar, comparar vencedores não isola somente o efeito residual. Os efeitos de capacidade, otimização e seleção devem ser considerados.

## Avaliação externa e relatório

Somente após congelar os dois vencedores, testar H0 2020→2020, 2021→2021 e 2022→2022; H1 2021→2022 e complementar 2020→2021. A área externa já foi explorada em pesquisas anteriores e continua exploratória, embora não entre no ajuste destes modelos. O buffer interno de três pixels não separa halos de 56 km: possíveis dependências espaciais permanecem uma limitação.

Registrar RMSE, MAE, R², viés, NMAD, P90 absoluto, captura top 5/10/20%, subconjuntos zero/positivo/alto, referências zero/média de treino/persistência e bootstrap espacial exploratório. NMAD mede dispersão robusta de resíduos, não viés; CE90 posicional não descreve erro de intensidade. Não normalizar as previsões com estatísticas do target externo nem truncar negativos. Persistência usa informação histórica adicional.

Relatório com todos os 12 braços, todas as tentativas de hiperparâmetros e ocultação, canais mantidos/removidos, épocas realizadas/escolhidas, número de parâmetros e passos, curvas, métricas externas e cinco comparações target/previsão. PNGs 600 dpi, escala compartilhada dentro de cada par; GeoTIFFs na resolução e georreferência originais. Curvas de todos os candidatos também ficam em CSV; figuras de busca mostram o candidato escolhido por família. Inspecionar visualmente as cinco figuras antes da entrega da análise final.

## Execução e continuidade

Entrada: `run_enriched_resunet.py`; configuração: `configs/enriched_resunet.yaml`; saída separada em `runs/enriched_resunet/<timestamp>`. O motor comum recebe marcação explícita ResUNet nos parâmetros e artefatos; modelos U-Net anteriores preservam arquitetura sem atalhos.

Repetir o mesmo comando retoma a última execução incompleta; checkpoints salvam modelo, otimizador e histórico a cada época. Se a interrupção ocorrer no meio de uma época, refazê-la desde o último checkpoint. Configuração, código, fontes de amostras e modelos completos são conferidos na retomada. Após conclusão, repetir o comando informa o relatório existente; usar `--new-run` apenas para uma execução nova deliberada. Falhas ficam registradas e não são ocultadas. O bloqueio impede duas execuções simultâneas na mesma pasta de saída.

Solicitar temporariamente ao Windows que evite suspensão automática durante o processo. Isso não impede suspensão manual, fechamento da tampa, reinício ou falta de energia. O usuário inicia o treino; o assistente executa apenas verificações e testes sintéticos durante a preparação.

## Referências

Verificação de preparação: checagem de 195 inputs/targets de desenvolvimento e dos caches, GPU disponível; 20 casos de teste aprovados (incluindo regressão dos fluxos U-Net/controles, treino dos seis braços, checkpoints, relatório e GeoTIFF). Ensaio sintético ResUNet com 24 canais, 16 filtros e 960×960, ida/volta: aproximadamente 1,81 GiB de memória CUDA alocada no pico; não é garantia de consumo idêntico no experimento inteiro. A comparação geométrica entre núcleos foi verificada com TF32 desabilitado no teste para separar diferenças de precisão entre kernels de possíveis emendas; o treino mantém a política padrão de precisão da U-Net, cujas opções serão registradas no ambiente.

- [He et al. (2016), Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385), preprint 2015: aprendizagem residual.
- [Zhang, Liu e Wang (2018), Road Extraction by Deep Residual U-Net](https://doi.org/10.1109/LGRS.2018.2802944); [preprint 2017](https://arxiv.org/abs/1711.10684): combinação de unidades residuais com U-Net.
- [Ronneberger et al. (2015), U-Net](https://arxiv.org/abs/1505.04597): caminhos de contração/expansão e conexões entre escalas.
- [Ioffe e Szegedy (2015), Batch Normalization](https://arxiv.org/abs/1502.03167).
- [Loshchilov e Hutter (2019), AdamW](https://arxiv.org/abs/1711.05101).
- [Valavi et al. (2019), blockCV](https://doi.org/10.1111/2041-210X.13107): avaliação espacial.
- [Protocolo do experimento 10](EXPERIMENTO_10_UNET.md) e [protocolo comum](EXPERIMENTOS_ENRIQUECIDOS_2026.md): métricas, canais, critérios, gradação temporal e processamento dos rasters. As referências originais são usadas para a fundamentação; não representam uma revisão exaustiva das variantes recentes de ResUNet.
