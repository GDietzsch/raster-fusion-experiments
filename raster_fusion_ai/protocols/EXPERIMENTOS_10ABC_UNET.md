# Controles U-Net 10A, 10B e 10C

Protocolo definido antes das novas execuções. Origem: experimento 10, run `20260927T185613221209Z`. Os resultados e modelos originais são preservados. O usuário executa a sequência noturna; a análise e a inspeção visual dos mapas ocorrerão depois.

## Perguntas e condições

- **10A — estabilidade:** comparar núcleos rotativos e núcleos fixos, em H0 e H1, com três inicializações (4210, 4211 e 4212). Mantêm-se canais dos vencedores, cinco níveis, filtros, AdamW, taxa 0,001 e weight decay 0,01. A mesma semente espacial 4210 fixa planos e ordens, separada da inicialização. A primeira época usa os mesmos núcleos nas duas condições; posteriormente a rotação amplia a cobertura e o plano fixo repete os mesmos núcleos. Assim, o contraste envolve a política de exposição e a cobertura, não apenas ruído numérico. O plano fixo é sorteado uniformemente dentro de cada ano, sem consultar valores do target. Não se garante representatividade de hotspots: cobertura e resumos dos targets selecionados/disponíveis serão registrados. São 12 modelos novos.
- **10B — profundidade e contexto:** reduzir de cinco para três níveis, mantendo canais, filtros iniciais, otimizador, épocas e plano fixo de 10A, em H0/H1 e nas três sementes. Reutilizar os seis modelos profundos fixos do 10A como referência. São seis modelos novos. **Este é um controle conjunto de campo receptivo e capacidade**, pois a redução de profundidade também reduz parâmetros. Não atribuir causalmente qualquer diferença somente ao contexto. BatchNorm agrega estatísticas espaciais no treino; a localidade do campo receptivo convolucional refere-se à inferência em modo de avaliação. Não reduzir simplesmente o halo: manter núcleo 512 e halo 224 (entrada 960) para evitar confundir contexto com emendas de inferência.
- **10C — componentes GPA em H1:** comparar magnitude (quatro canais), direção codificada por seno/cosseno (oito canais) e ambas (12 canais). Mantêm-se cinco níveis, 16 filtros iniciais e plano fixo de 10A. Reutilizar os três modelos GPA completos fixos de H1 do 10A; treinar seis modelos novos, três por ablação. Não expandir esta comparação para H0: ali o vencedor é uma mistura de canais originais, gradientes e grafos. São derivados espaciais já congelados do gradiente, não novos índices G1/G2.

Total: **24 modelos novos**, com controles compartilhados explicitamente identificados. Não executar de novo os seis conjuntos de entradas: estes são controles dos vencedores, não uma repetição integral do experimento 10. Os canais mantidos/removidos são listados em cada relatório.

## Orçamento fixo, amostras e curvas

Cada modelo é inicializado do zero e treinado por **11 épocas**, número dos dois vencedores originais; não há escolha adaptativa de época nem parada antecipada nestes controles. Isso iguala o orçamento de exposição em 10A e facilita contrastes pareados. O objetivo não é encontrar o ótimo de cada arquitetura. Custos de computação diferem em 10B apesar do mesmo número de atualizações.

O ajuste usa os centros congelados de ajuste+seleção dos experimentos anteriores: H0 até 600.000 centros disponíveis e H1 até 200.000. A perda usa apenas esses centros, em até 64 núcleos por ano por época; a rotação/fixação determina quantos centros realmente entram nas atualizações. Todos os centros de ajuste+seleção são avaliados ao final, incluindo os não visitados no plano fixo. Por isso RMSE do conjunto completo de treino não equivale ao erro somente dos centros supervisionados. A escala é estimada apenas no conjunto de ajuste+seleção. A validação completa permanece em 300.000 centros H0 e 100.000 H1.

Para diagnóstico por época, avaliar um **monitor fixo de até 32 núcleos por ano da validação**, compartilhado entre todas as condições/sementes de um horizonte. Esse monitor não escolhe época, hiperparâmetros ou modelos e não entra no gradiente. O helper comum usa internamente o nome `selection_rmse`; o histórico exportado de cada controle o renomeia para `monitor_validation_rmse`. O RMSE completo de validação ao final é a métrica principal. A perda de treino é média dos erros nas atualizações, não uma avaliação em modo de inferência. No braço rotativo o suporte muda; no fixo o suporte permanece, mas pesos e ordem evoluem.

H0 significa mesmo ano; H1, defasagem de um ano. Esses nomes temporais não constituem, por si, hipóteses estatísticas nula/alternativa. Os contrastes avaliam se as intervenções reduzem RMSE de forma consistente entre sementes. Relatar diferenças pareadas, média e desvio padrão amostral. Três sementes não sustentam conclusões robustas sobre significância nem uma estimativa precisa de incerteza populacional. Não selecionar a melhor semente. A máscara e o buffer espacial de três pixels são os mesmos: contextos podem cruzar zonas, limitação preservada e registrada.

## Ordem, continuidade e resultados

Treinar sequencialmente 10A, 10B e 10C, em uma única GPU. Somente após terminar todos os modelos requisitados iniciar avaliações externas. Avaliar **todas as sementes**: três pares em H0 (2020→2020, 2021→2021, 2022→2022), dois em H1 (2021→2022 e complementar 2020→2021). Na sequência completa são **57 pares externos únicos**. Referências compartilhadas não são contadas como novas avaliações independentes. A área externa já foi explorada antes e continua exploratória; não orienta seleção ou ajuste nesta sequência.

Métricas: RMSE, MAE, R², viés, NMAD, P90 do erro absoluto, captura top 5/10/20%, subconjuntos de target zero/positivo/alto; referências zero, média do treino e persistência quando disponível; bootstrap espacial descritivo. CE90 posicional não se aplica à intensidade. Target zero denota ausência de registro naquela fotografia. Não normalizar previsões com estatísticas do target externo nem truncar negativos.

Salvar modelos, hashes, configuração, código, ambiente, planos espaciais por época, logs, histórico por época, métricas completas por modelo, resumo por condição e diferenças pareadas. Um relatório por controle e índice geral. Mapas de target e previsão lado a lado a 600 dpi, mesma escala dentro do par e GeoTIFFs com resolução/georreferência originais; mapas exibidos com passo dois, métricas calculadas no suporte integral. Não presumir escala idêntica entre figuras diferentes. Os relatórios são automáticos; não alegam inspeção visual antes de ela ocorrer.

Checkpoint ao final de cada época. Repetir o mesmo comando retoma a última execução correspondente e verifica hashes do código/configuração/amostras/modelos fonte. Se houver interrupção no meio de uma época, essa época é refeita. Modelos e avaliações já concluídos são reutilizados. Um bloqueio do sistema impede duas sequências simultâneas na mesma pasta. Uma execução completa não é repetida inadvertidamente; `--new-run` cria outra. Erro interrompe a sequência preservando artefatos; não suprimir falhas para aparentar conclusão.

A solicitação temporária ao Windows evita suspensão automática por inatividade durante a execução e é liberada ao terminar; não impede suspensão manual, fechamento da tampa, falta de energia ou reinício do sistema. Deixar o computador ligado à tomada e a sessão do terminal aberta. O tempo total depende do dispositivo; não há garantia de conclusão até a manhã.

## Referências

- [Ronneberger, Fischer e Brox (2015), U-Net](https://arxiv.org/abs/1505.04597): arquitetura de contração/expansão.
- [Araujo, Norris e Sim (2019), Computing Receptive Fields](https://distill.pub/2019/computing-receptive-fields/): relação entre arquitetura e campo receptivo.
- [Ioffe e Szegedy (2015), Batch Normalization](https://arxiv.org/abs/1502.03167): normalização empregada.
- [Loshchilov e Hutter (2019), Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101): AdamW.
- [Valavi et al. (2019), blockCV](https://doi.org/10.1111/2041-210X.13107): avaliação espacial.
- [PyTorch, Reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html): sementes e limites de reprodutibilidade.
- [scikit-learn, Model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics): métricas de regressão.
- [SciPy, median_abs_deviation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.median_abs_deviation.html): base do NMAD.
- [Protocolo original](EXPERIMENTO_10_UNET.md): dados, canais, desenho do experimento 10 e referências de processamento raster.
