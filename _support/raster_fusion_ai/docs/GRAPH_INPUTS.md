# Grafos independentes e rasters de conectividade

## Escopo

32 casos: quatro camadas × anos 2019–2022 × regiões de treino e teste.
Camadas: Garimpos, Ocorrencias_Minerais, Pistas_Irregulares, Pistas_Regulares.
Nenhum Target é lido e não há arestas entre camadas, regiões ou anos.

Fontes dos nós:
- Treino: Projeto/Treinamento/Inputs/CAMADA_ANO.shp.
- Teste: Temporal/Test/CAMADA_ate_ANO.shp, cortes acumulados documentados em
  Temporal/Test/LEIA_ME_CORTES_TEMPORAIS.txt e relatorio_cortes_temporais.json.

Grades e máscaras: data/regiao_*/ANO/inputs/CAMADA.tif.
As fontes anuais estão explicitamente mapeadas no código; não há descoberta
recursiva de shapefiles que possa incorporar targets ou dados de anos futuros.
Audita as datas antes de gerar produtos. Ausência de data ou data futura interrompe
a execução. Pistas irregulares têm data inferida do nome da imagem; isso não
comprova surgimento ou operação. A equivalência exata dessas fontes de pontos
às feições usadas nos KDEs históricos **não foi demonstrada**, especialmente
na região externa. Trata-se de uma representação complementar com proveniência
própria, não transformação matemática do raster original.

## Método mantido dos exemplos

Cada coordenada distinta é um nó. Coincidências exatas são agrupadas; contagem
preservada em nos.csv, sem multiplicar pesos. MultiPoint é expandido em pontos,
com rastreio ao registro SHP em origem_nos.csv. Não converte áreas em centroides.
Transforma coordenadas ao CRS projetado métrico de cada raster.

Grafo não dirigido: união das escolhas de até oito vizinhos por nó, com distância
máxima de 50 km. Após simetrizar, o grau pode ultrapassar oito. Não força ligação
de componentes. Peso da aresta=exp(-distância/50.000). Força do nó=soma dos pesos
incidentes. Distâncias são euclidianas na projeção, não percursos reais.
Empates no limite do kNN seguem a implementação cKDTree; versão SciPy registrada.
Estes parâmetros são exploratórios, mantidos iguais em todos os casos.

Raster: para cada centro de pixel, busca até oito nós próximos em um raio de
50 km e calcula a média de suas forças, com pesos exp(-distância/50.000).
Fora do alcance de todos os nós, conectividade=0 e suporte=0. Dentro do alcance,
suporte=1; conectividade pode ser zero se os nós forem isolados. O limite do raio
e a troca de vizinhos podem produzir descontinuidades, como nos exemplos.

O grafo é calculado com todo o shapefile da região/ano, antes de dividir o raster
em blocos. Nós fora do retângulo do raster são contados na auditoria e preservados,
pois ainda podem influenciar sua borda. A restrição espacial das próprias fontes
pode causar efeitos de borda; não há integração de fontes da outra região.
Não interpretar componentes ou arestas como redes operacionais comprovadas.

## Saídas por caso

- nos.csv: ID, coordenadas métricas, multiplicidade, grau, força, componente e tamanho.
- arestas.csv: pares de IDs, distância em metros e peso.
- origem_nos.csv: número do registro original SHP → ID do nó.
- adjacencia.npz: matriz esparsa simétrica com pesos (scipy.sparse.load_npz).
- crs.wkt: CRS dos CSVs.
- nos.geojson e arestas.geojson: vetores em WGS84 para visualização em SIG.
- conectividade.tif: entrada adicional proposta para treinamento.
- suporte.tif: máscara 0/1 que distingue falta de suporte de conectividade nula.
- comparacao.png e detalhe.png: input, grafo e raster lado a lado, 600 dpi.

Os dois GeoTIFFs cobrem **toda a extensão nativa de cada input**, sem alterar
resolução, dimensão, origem ou CRS. NoData/NaN/inf do original continuam NoData;
zero é válido. O raster original não participa do cálculo das arestas ou forças.

Visão geral usa prévia por vizinho mais próximo de até 1800 pixels no maior lado.
Detalhe mostra até 1024x1024 pixels nativos centrados no nó de maior força; em
grafo vazio usa o centro da imagem. É ilustrativo, não amostra representativa.
Figuras de 9000x3600 a 600 dpi. Escala de força comum entre grafo e raster dentro
de cada figura, mas pode diferir entre figuras. Todos os nós/arestas visíveis são
desenhados, mesmo quando se sobrepõem.

Total previsto: 32 grafos, 32 rasters de conectividade, 32 de suporte e 64 PNGs,
mais tabelas, vetores, auditorias, hashes e cópia dos três scripts usados.

## Execução

```powershell
& "C:\Gepeto_SandBox\raster_fusion_ai\.venv\Scripts\python.exe" "C:\Gepeto_SandBox\raster_fusion_ai\generate_graph_inputs.py" --dpi 600
```

Saída: outputs/graph_inputs/<data_hora_UTC>/REGIAO/ANO/CAMADA/.
--check-only confere todos os casos sem gerar produtos.
Retomada: --output "PASTA_DA_EXECUCAO" --resume, com mesmos scripts/parâmetros.
Confere hashes e pula casos concluídos; caso interrompido é refeito integralmente.
Memória limitada por blocos de 256x256 para o raster; não cria grafo por pixel.
O processamento completo fica para o usuário; o assistente verifica casos pequenos.

## Uso experimental

Acrescentar conectividade aos originais; manter suporte como indicador separado
ou usá-lo para interpretar os zeros. Ajustar a normalização só no treino.
Conectividade e densidade podem ser correlacionadas; ganho precisa ser medido.
Aplicar o mesmo protocolo temporal e suporte de avaliação aos controles.
Grafo usa todas as feições disponíveis da região/ano: é contexto espacial
transdutivo, sem rótulos; não equivale a um teste de locais completamente
desconhecidos. Nenhuma integração ou treinamento de modelos ocorre neste script.
