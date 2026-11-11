# Avaliação do experimento Zarr v2

Resultados reais de 11/11/2026, usando contagens semanais **positivas e condicionais**. Não são métricas de detecção de ignição, de ocorrência incondicional nem de área queimada. O arquivo `runs/zarr_brasil_v2_20261111/evaluation_v2/metrics.json` contém métricas por região, estação, intensidade e suporte de versão; `prediction_test.png` apresenta resultados observados no teste.

## Seleção e teste de previsão

A variante com meteorologia foi escolhida antes do teste pelo CRPS de 148 exemplos de validação nas células de treino: 23,93, contra 24,25 sem meteorologia. No teste temporal de 2024–2025, com 403 exemplos elegíveis nas 16 células:

| Método | MAE, detecções/semana | RMSE | CRPS | Cobertura do intervalo 80% |
|---|---:|---:|---:|---:|
| Persistência | 41,06 | 95,91 | — | — |
| Referência sazonal | 45,50 | 101,26 | — | — |
| Regressão log-linear com MODIS | 36,93 | 91,78 | — | — |
| Pyro sem clima | 36,81 | 88,24 | 27,03 | 83,1% |
| Pyro com clima, selecionado na validação | 39,09 | 98,52 | 29,36 | 79,2% |

As referências sem distribuição não têm CRPS nem intervalo. A regressão log-linear sem MODIS teve MAE 37,21. Para o modelo com clima, a largura média do intervalo de 80% foi 84,88 detecções e a cobertura do intervalo de 95% foi 92,8%. A cobertura global próxima do nominal não garante calibração local. A variante sem clima teve resultados melhores no teste global; não se deve escolher o modelo pelo teste depois de observá-lo nem afirmar que a meteorologia melhorou a previsão neste experimento.

Nas 12 células usadas no ajuste, o modelo com clima teve MAE 30,68 e cobertura 80% de 82,7% em 329 exemplos. Nas quatro células reservadas, teve MAE 76,51 e cobertura 80% de 63,5% em 74 exemplos; ali a regressão simples com MODIS teve MAE 63,58. Nos 51 exemplos acima do percentil 90 do treino, o modelo com clima teve MAE 178,27 e cobertura 80% de 43,1%, indicando falha em situações de maior intensidade. Essas estimativas por grupo são condicionadas à existência dos registros de seleção.

Os 403 exemplos de teste têm código MODIS da semana anterior `61.03`, ausente do treino, onde aparece `6.03`. A saída os marca como `unseen_version`. Não há base para atribuir o erro somente à mudança de versão, e os intervalos não incorporam essa fonte de incerteza.

## Comparação entre sensores

O modelo Pyro de contraste entre `log(1 + VIIRS)` e `log(1 + MODIS/Terra)` usou 741 pares positivos de treino. No teste, 445 pares positivos tiveram MAE 0,587 na escala de diferença logarítmica, contra 0,608 da mediana global do treino. Nas quatro células reservadas, o modelo teve MAE 0,779 e a mediana 0,725 em 85 pares: a alternativa simples venceu. Essa métrica mede concordância relativa condicionada, não recuperação de incêndios reais.

## Exemplo de inferência

O arquivo `runs/zarr_brasil_v2_20261111/exports_v2/scenario_20251228.json` contém uma simulação retrospectiva na célula centrada em -11,5°, -46,875°. Sob climatologia fixa, o modelo estimou média de 8,61 detecções condicionais e intervalo preditivo de 80% [1, 19]; com as alterações hipotéticas de `config/scenario_example.json`, média 10,36 e intervalo [2, 21]. A comparação usa o mesmo modelo, não calcula efeito causal e carrega o aviso de versão MODIS desconhecida no treino. Como a ocorrência não está identificada, ambas as probabilidades são `null` e o estado geral é `insufficient_data`, embora o estado específico da contagem condicional seja `ok`.

A consulta simulada para 08/11/2026 retorna `insufficient_data` porque faltam arquivos nominais dos sensores para a semana anterior. Mais dados recentes e informação de cobertura são necessários antes de uma previsão operacional.
