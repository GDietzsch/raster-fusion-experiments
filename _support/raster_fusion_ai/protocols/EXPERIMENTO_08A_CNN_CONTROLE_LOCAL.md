# Experimento 08A — controle da CNN sem o ramo amplo

Protocolo fixado antes de executar o controle. A CNN do experimento 08 teve RMSE de validação H0 2,1842 e H1 7,2209, mas não superou a previsão constante zero em nenhum dos cinco pares externos. Ela combinava uma janela local 5×5 com uma grade esparsa 5×5 que alcançava ±41 km por eixo. **O experimento 08 não treinou uma rede equivalente apenas com a janela local**. Portanto, seus resultados não permitem atribuir o ganho interno ao ramo amplo. Este controle isola essa pergunta.

## Contraste planejado

Treinar dois novos modelos, H0 e H1, com **somente o ramo local 5×5** da arquitetura do experimento 08: duas convoluções válidas 3×3, ReLU e cabeça linear 1×1. Não existe camada ou peso do ramo amplo no controle. Os rasters centrais e seus canais continuam iguais, inclusive KDEs cujo valor no próprio pixel já incorpora suavização de cerca de 41 km. Logo, o contraste testa o benefício da **amostragem espacial adicional** a ±20,5 e ±41 km, não a remoção de toda informação derivada de KDE.

Para reduzir confundimento, congelar previamente os dois vencedores do experimento 08: H0 `todas` com 24 canais, 32 filtros, AdamW com taxa 0,001 e decay 0,01, 25 épocas; H1 `todas_filtradas` com 22 canais, os mesmos parâmetros, 34 épocas. Usar os mesmos `samples.npz`, posições, labels, escaladores, seed, ordem de lotes e tamanho de lote 512. Ler a **primeira escala** dos bancos de patches já congelados pelo experimento 08, sem recriar nem selecionar exemplos. Os modelos locais serão treinados do zero; não haverá nova busca de hiperparâmetros, seleção de canais, parada antecipada nem escolha de vencedor. A arquitetura menor tem menos parâmetros por definição, pois o ramo e metade da entrada da cabeça foram removidos; o contraste mede o efeito total dessa ablação, não um efeito mantendo o número de parâmetros constante.

Validar H0/H1 nos mesmos centros congelados, reportando RMSE, MAE, R², NMAD, captura top 5/10/20%, subconjuntos de target zero/positivo/alto, curvas por época, canais, épocas e parâmetros. Comparar cada controle com seu respectivo vencedor original. Depois que ambos forem treinados e registrados, avaliar os mesmos cinco pares externos: H0 2020→2020, 2021→2021, 2022→2022; H1 2021→2022 e o complementar 2020→2021. Usar a mesma máscara de pixels válidos, georreferência e targets verificados por hash. Para cada par, mostrar tabela com RMSE local, RMSE original e referência zero, além de PNG 600 dpi target/previsão e GeoTIFF da previsão local. O critério de leitura primário é a diferença pareada de RMSE na validação; a área externa já foi examinada e suas comparações são exploratórias, sem uso para reajustar o controle.

## Limites de interpretação

O buffer espacial do experimento 08 era de três pixels: separa centros, mas não os contextos de ±41 km da CNN original. O controle local reduz essa sobreposição, o que torna a comparação de validação assimétrica e potencialmente favorável ao modelo amplo. Uma melhora local sugere que a segunda amostragem não ajudou sob este protocolo; uma piora local não provaria que 41 km generaliza, pois pode refletir essa sobreposição, a mudança no número de parâmetros ou a otimização. A área externa, já observada ao planejar este controle, não é um teste cego para escolher nova arquitetura. Não usar o target zero como afirmação de impossibilidade de ocorrência; ele significa ausência de registro naquela fotografia. CE90 posicional não se aplica ao erro de intensidade.

O código deve verificar identidade de configuração, hashes dos patches/amostras, canais e parâmetros da referência antes do treino, preservar checkpoints para retomada e registrar todo o código/configuração usado. Não se modificam arquivos do experimento 08.

## Referências

- [Experimento 08 — CNN espacial](EXPERIMENTO_08_CNN.md): arquitetura, dados e seleção original.
- [PyTorch Conv2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html): convoluções e parâmetros removidos na ablação.
- [PyTorch AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html): ajuste mantido entre os modelos.
- [Valavi et al. (2019)](https://doi.org/10.1111/2041-210X.13107): necessidade de separação espacial na avaliação.
