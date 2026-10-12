# Experimento 07 — MLP

Protocolo do novo treinamento com 500 épocas, atualizado em 26/09/2026 após a interrupção da execução preliminar 60/12 (`runs/enriched_mlp/20260926T203402697385Z`), que não chegou ao teste externo. O objetivo é estimar a intensidade do target. Zero significa ausência de registro na fotografia considerada, não impossibilidade de ocorrência. Os targets da área externa já foram examinados em experimentos anteriores; comparações entre modelos serão exploratórias.

## Dados e comparação

Repetir as mesmas 24 entradas disponíveis, máscara comum, amostras, blocos espaciais, pares temporais, seis conjuntos, H0/H1, métricas, referências e regra de seleção dos experimentos 01–06. H0 usa zonas de desenvolvimento 1/2/3 para ajuste inicial/seleção de camadas/validação. H1 usa targets 2020/2021/2022, nessa ordem, dentro da zona de desenvolvimento. Nenhum target da região externa orienta arquitetura, época, camadas ou vencedor.

O estimador é `sklearn.neural_network.MLPRegressor`, saída linear, ativação ReLU, otimizador Adam, perda quadrática e uma semente fixa (4209). Cada conjunto de atributos recebe `StandardScaler` ajustado somente nas respectivas amostras de ajuste; o target recebe centralização e divisão pelo desvio-padrão dessas mesmas amostras para estabilidade numérica. Predições e métricas são revertidas para as unidades originais do target, sem truncar valores negativos. A padronização é reajustada na união de ajuste inicial + seleção antes do modelo final. Limitar BLAS a quatro threads. `early_stopping=False`: o holdout aleatório interno do estimador não substitui nossa separação espacial/temporal.

## Busca e histórico

1. Na busca e filtragem, usar até 20.000 pixels por ano e etapa, com seed de amostragem 4203, os mesmos índices fixos usados nos experimentos anteriores e compartilhados entre os candidatos do mesmo horizonte. Salvar índices e hashes. As amostras completas continuam no reajuste final e na validação.
2. Para cada família base (`originais`, `gpa`, `grafos`, `todas`), avaliar as quatro arquiteturas/configurações da tabela abaixo. `partial_fit` faz uma época por chamada; medir RMSE e MAE na seleção interna em cada época, desde a época 10 até no máximo 500. Parar após 15 épocas sem melhorar o RMSE de seleção. Guardar a curva completa por candidato. Escolher a configuração e a época de menor RMSE de seleção, com MAE, menor número de parâmetros e menor época como desempate. A etapa de validação não escolhe épocas.
3. Os braços filtrados herdam a configuração e a época da família base. `originais_filtradas` compara as 15 combinações não vazias de fenômenos; `todas_filtradas` faz eliminação reversa de grupos fenômeno×produto, mantendo seno/cosseno e conectividade/suporte juntos. Cada tentativa de filtragem treina até a época herdada e é avaliada apenas na seleção interna. Registrar todas as tentativas.
4. Reajustar os seis braços com ajuste inicial + seleção completos, usando a época previamente escolhida. Registrar a perda de treino por época e métricas finais; avaliar uma vez na validação completa. Escolher um vencedor por horizonte por RMSE, MAE e menor número de canais. Os dois vencedores são congelados antes de ler qualquer target externo.

| Candidato | Camadas ocultas | Alpha (L2) | Taxa inicial Adam |
|---:|---|---:|---:|
| 1 | 32 | 0,001 | 0,001 |
| 2 | 64 | 0,001 | 0,001 |
| 3 | 64, 32 | 0,001 | 0,001 |
| 4 | 64, 32 | 0,01 | 0,003 |

`batch_size=512`, `max_epochs=500`, `patience=15`, `min_epochs=10`. A paciência conta épocas consecutivas sem novo mínimo de RMSE depois da época 10; o limite de 500 continua valendo se houver melhoras suficientes. O número de épocas é transferido da subamostra de busca para o reajuste completo; isso fixa o protocolo, mas não garante o mesmo número de atualizações de pesos. O histórico de seleção mede RMSE em dados separados do ajuste inicial; a perda interna do Adam é em unidades padronizadas e não deve ser confundida com RMSE. Uma única semente não mede variabilidade entre inicializações.

## Avaliação e relatório

Testar H0 nos pares externos 2020→2020, 2021→2021 e 2022→2022; H1 em 2021→2022. Acrescentar H1 2020→2021 como avaliação complementar após congelar seu vencedor. Relatar RMSE, MAE, R², NMAD, viés, P90 do erro absoluto, erros condicionais a target zero/positivo, captura de massa top 5/10/20%, referências zero/média/persistência e incerteza exploratória por blocos. CE90 não é adequada para erro de intensidade. O relatório final terá as seis combinações em H0/H1, todas as tentativas de camadas e arquiteturas, canais retidos/removidos, parâmetros, escala, épocas, curvas reais da seleção e do ajuste final, modelos salvos, GeoTIFFs, imagens target/previsão em 600 dpi, código e proveniência.

## Fontes

- [MLPRegressor, documentação oficial do scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html): ReLU, Adam, `partial_fit`, perda e histórico.
- [Redes neurais supervisionadas, scikit-learn](https://scikit-learn.org/stable/modules/neural_networks_supervised.html): sensibilidade à escala das entradas.
- [StandardScaler, scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html): ajuste e transformação somente com dados de treinamento.
- Kingma e Ba (2015), [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980): otimizador Adam.
- Separação espacial, GPA, grafos, métricas e cautelas de interpretação: `protocols/EXPERIMENTOS_ENRIQUECIDOS_2026.md`, `docs/GRADIENT_INPUTS.md` e `docs/GRAPH_INPUTS.md`.
