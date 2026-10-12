# Experimento 09A — controle local da ResNet

Protocolo fixado após a conclusão do experimento 09 e antes de executar o controle. A ResNet de duas escalas obteve RMSE de validação 2,6509 em H0 (braço `grafos`, oito canais, 32 filtros, 17 épocas) e 7,0480 em H1 (braço `gpa`, doze canais, 16 filtros, 32 épocas). Seu RMSE externo ficou acima da referência zero nos cinco pares. O controle pergunta se a grade esparsa até ±41 km contribuiu para cada vencedor; não usa os resultados externos para escolher canais ou épocas.

## Contraste congelado

Treinar do zero dois modelos com **somente o ramo local residual 5×5**. O ramo conserva as duas convoluções 3×3, projeção de atalho 1×1, bloco residual pontual 1×1, BatchNorm e ReLU da referência. Não haverá parâmetros nem leitura do ramo amplo no ajuste ou na inferência. A arquitetura local usa menos parâmetros porque também perde metade da entrada da cabeça linear.

Fixar, separadamente para H0 e H1, o braço vencedor, a lista e ordem dos canais, os escaladores, a taxa de aprendizado, o weight decay, o número de filtros, o número de épocas, seed, ordem de lotes e tamanho do lote da ResNet concluída `runs/enriched_resnet/20260927T140640055671Z`. Ler os mesmos centros, targets e a primeira escala dos bancos de patches congelados em `runs/enriched_cnn/20260926T223539611207Z`. Conferir os hashes dos modelos de referência e amostras. Não repetir busca, seleção de canais ou parada antecipada.

Comparar RMSE de validação pareado, seguido de MAE, R², NMAD e captura top 5/10/20%, além de curvas de perda por época e métricas por ano. O sinal de Δ RMSE = RMSE local − RMSE original mostra qual arquitetura teve menor erro nos mesmos centros. Após registrar os dois modelos, avaliar H0 2020→2020, 2021→2021, 2022→2022; H1 2021→2022 e o complementar 2020→2021. Verificar hash do target e total de pixels contra a referência; produzir GeoTIFF e PNG target/previsão a 600 dpi em escala comum por par. O relatório deve registrar canais, parâmetros, épocas, métricas e a justificativa desta ablação.

## Interpretação

Os canais KDE centrais já codificam suavização espacial de aproximadamente 41 km. A ablação mede o ganho da **amostragem adicional** do ramo amplo, não remove o contexto KDE dos inputs. Como o ramo removido também reduz o número de parâmetros, o contraste não isola um único mecanismo causal. O buffer de três pixels não garante separação dos contextos amplos entre zonas de desenvolvimento. A área externa já foi examinada nos experimentos anteriores; seus resultados são exploratórios e não reorientam o controle. Target zero significa ausência de registro naquela fotografia, não impossibilidade de ocorrência; CE90 posicional não se aplica à intensidade.

He et al. (2016) fundamentam o uso de atalhos residuais para facilitar a otimização de redes profundas, mas não estabelecem que o ramo amplo será útil nestes dados. O controle local preserva esses atalhos e testa apenas a ablação arquitetural planejada.

## Referências

- [He et al. (2016), *Deep Residual Learning for Image Recognition*](https://arxiv.org/abs/1512.03385): aprendizado residual e atalhos.
- [Experimento 09 — ResNet de duas escalas](EXPERIMENTO_09_RESNET.md): referência congelada e desenho principal.
- [Experimento 08A — controle local da CNN](EXPERIMENTO_08A_CNN_CONTROLE_LOCAL.md): precedente da ablação pareada.
- [Valavi et al. (2019), *blockCV*](https://doi.org/10.1111/2041-210X.13107): limites da separação espacial.
