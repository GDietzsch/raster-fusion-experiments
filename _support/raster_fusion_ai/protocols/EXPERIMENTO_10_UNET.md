# Experimento 10 — U-Net densa para regressão raster

Protocolo fixado antes da execução. O objetivo continua a intensidade do target por pixel; target zero significa ausência de registro naquela fotografia. H0 usa os inputs do mesmo ano, H1 os do ano anterior. O experimento avalia os seis braços originais, originais filtradas, GPA, grafos, todas e todas filtradas, em separado para H0/H1, com escolha do vencedor por menor RMSE de validação, depois MAE e menos canais. A área externa só é avaliada após congelar os dois vencedores.

## Por que esta U-Net e esta escala

Ronneberger, Fischer e Brox (2015) propuseram um caminho de contração para captar contexto e outro de expansão para localizar a saída, ligados por conexões entre níveis. Aqui empregamos a implementação U-Net existente no projeto com **cinco reduções por fator dois**, convoluções 3×3, BatchNorm, interpolação bilinear e saída linear para regressão. É uma adaptação ao target contínuo, não a tarefa original de segmentação. Não utiliza target vizinho como entrada.

Cada inferência recebe um tensor de **960×960 pixels**: núcleo de saída 512×512 mais halo de 224 pixels por lado. Em resolução de 250 m, o núcleo mede 128 km e o halo 56 km. Isso inclui o raio KDE de 41 km (164 pixels) e cobre o campo receptivo da rede para evitar emendas artificiais entre núcleos. O input 960 é divisível por 32, mantendo a fase das cinco reduções. Os rasters de 24 canais alinhados são reutilizados do experimento 08 somente para leitura. O requisito de GPU foi ensaiado na RTX 4060 Laptop de 8 GiB com 24 canais, 960×960, 16 filtros iniciais e passagem de ida/volta; o pico medido foi cerca de 1,54 GiB no teste sintético, sem garantir o mesmo pico em toda a execução.

## Amostras, custo e comparabilidade

Os centros, labels, anos e zonas são os mesmos arquivos `samples.npz` congelados desde o MLP. A perda é MSE apenas nesses centros, embora a rede produza uma superfície densa. Nenhum outro pixel de target entra no treino ou na escolha de hiperparâmetros. Para caber no tempo, cada época percorre até 64 núcleos de 512×512 por ano de input; a ordem circular determinística troca os núcleos a cada época e permite cobrir progressivamente o conjunto espacial completo. A seleção interna e a ocultação de canais usam 32 núcleos fixos por ano. Esses subconjuntos são salvos e são **diferentes do uso de todos os centros por época** nos modelos tabulares/CNN; a validação final volta aos mesmos centros congelados completos. A regra de escolha do vencedor permanece comparável, mas o regime de otimização não é idêntico.

Nas quatro famílias base, comparar 8 e 16 filtros iniciais, taxa AdamW 0,001, weight decay 0,01, até 30 épocas, mínimo cinco e paciência seis, pela seleção interna. Os dois braços filtrados herdam parâmetros e época da família base, mas são treinados do zero. Para originais, pontuar as 15 combinações não vazias de fenômenos por ocultação de canais no modelo base; para todas, testar remoções de grupos fenômeno×produto e a remoção conjunta dos grupos favoráveis, como na CNN. A ocultação mede dependência preditiva, não causalidade. Os seis modelos finais por horizonte são treinados do zero sobre centros de ajuste+seleção e avaliados na validação completa; todos os candidatos, curvas, canais, máscaras, seeds e hashes devem ser registrados. A seleção do braço usa somente a validação. A validação espacial de três pixels não separa contextos de 56 km; seus ganhos podem ser otimistas.

## Teste externo e relatório

Testar H0 2020→2020, 2021→2021, 2022→2022; H1 2021→2022 e o complementar 2020→2021. Calcular RMSE, MAE, R², viés, NMAD, P90 absoluto, captura top 5/10/20%, subconjuntos zero/positivo/alto, referências zero/média/persistência e intervalos espaciais exploratórios. CE90 posicional não se aplica à intensidade. Registrar canais e hiperparâmetros dos dois vencedores, todas as tentativas internas e os 12 resultados de validação. Salvar GeoTIFFs e PNG 600 dpi com target e previsão na mesma escala em cada par. Como a área externa foi examinada antes deste experimento, seu resultado é exploratório e não decide novos parâmetros.

## Referências

- [Ronneberger et al. (2015), *U-Net: Convolutional Networks for Biomedical Image Segmentation*](https://arxiv.org/abs/1505.04597): caminhos de contração/expansão e conexões entre níveis.
- [Ioffe e Szegedy (2015), *Batch Normalization*](https://arxiv.org/abs/1502.03167): normalização utilizada.
- [Loshchilov e Hutter (2019), *Decoupled Weight Decay Regularization*](https://arxiv.org/abs/1711.05101): AdamW.
- [Valavi et al. (2019), *blockCV*](https://doi.org/10.1111/2041-210X.13107): separação espacial e cautela na extrapolação.
- [Protocolo comum](EXPERIMENTOS_ENRIQUECIDOS_2026.md): seis braços, H0/H1 e métricas.
