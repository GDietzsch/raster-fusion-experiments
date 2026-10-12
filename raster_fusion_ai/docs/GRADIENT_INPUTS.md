# Magnitude e direção dos inputs

Decisão do usuário em 24/09/2026: remover os produtos GPA anteriores e preparar
a geração somente de magnitude e direção. Substitui a decisão de incluir G2
descrita em GPA_INPUTS.md; nenhum treinamento GPA foi executado por este fluxo.

Script: generate_gradient_inputs.py. Fonte: os 32 inputs organizados em
data/regiao_*/ANO/inputs/*.tif, incluindo 16 da região de teste.

## Algoritmo

Diferenças centrais com halo de um pixel, nas unidades físicas do CRS:
gx=(direita-esquerda)/(2*passoX); gy=(linha seguinte-linha anterior)/(2*passoY).
Os passos têm sinal e são convertidos para metros. Em rasters north-up, passoY
é negativo. Magnitude=hypot(gx,gy), intensidade original por metro.
Direção=atan2(gy,gx), graus entre -180 e 180; 0 aponta para leste e 90 para norte
nos eixos da projeção (não é azimute geodésico).

Dois produtos conceituais, quatro GeoTIFFs por input:

- `_magnitude.tif`;
- `_direcao_graus.tif`, para mapas;
- `_direcao_seno.tif` e `_direcao_cosseno.tif`, para os modelos.

Em regiões planas, magnitude=0, graus=NoData e seno=cosseno=0. Isso fornece uma
codificação neutra sem descartar os zeros válidos nos treinamentos. Não usar
a máscara do raster de graus como máscara de treino; usar magnitude/seno/cosseno.
Nas demais regiões, seno²+cosseno²=1, sujeito ao arredondamento float32.
O ângulo em graus não deve ser usado diretamente como variável linear.

NoData/NaN/inf no centro ou em algum dos quatro vizinhos necessários invalida
todos os produtos naquele pixel. Bordas externas de um pixel ficam NoData.
Não há preenchimento artificial, suavização, janela GPA, triangulação ou G2.
Preserva a grade; exige CRS projetado e raster monobanda sem rotação.

O script não modifica dados de entrada, targets ou modelos. No treinamento
posterior, acrescentar magnitude/seno/cosseno aos originais, ajustar a escala
da magnitude somente no treino e comparar no mesmo suporte válido do controle.
A integração dessas camadas no treinamento não faz parte deste gerador.

## Execução

```powershell
& "C:\Gepeto_SandBox\raster_fusion_ai\.venv\Scripts\python.exe" "C:\Gepeto_SandBox\raster_fusion_ai\generate_gradient_inputs.py" --dpi 600
```

Saída: outputs/gradient_inputs/<data_hora_UTC>, separada por região/ano/inputs.
Pré-verificar: adicionar --dry-run. Retomar: --output "PASTA_DA_EXECUCAO" --resume.
A retomada confere hashes; um input interrompido é recalculado integralmente.

Além dos 128 GeoTIFFs, gera 64 PNGs (geral e detalhe para cada input), cada um
com original/magnitude/direção lado a lado, 9000x3600 pixels a 600 dpi.
Prévia geral até 2000 pixels no maior eixo, vizinho mais próximo; detalhe de
até 512x512 pixels nativos, escolhido pelo máximo de uma prévia do original.
A magnitude usa min/max da prévia; a direção usa escala circular fixa -180/180.
Manifesto registra inputs, hashes, parâmetros, limites das figuras e recortes.
