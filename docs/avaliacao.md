# Avaliação do recorte brasileiro

> Este relatório registra o experimento inicial v1 em três células. A avaliação mais recente está em [avaliação Zarr v2](avaliacao_zarr_v2.md); os resultados têm recortes e seleções diferentes e não devem ser comparados diretamente.

Esta avaliação usa dados reais, mas é **retrospectiva e condicional**: o alvo é a contagem semanal de registros VIIRS/NOAA-20 em semanas positivas cuja semana anterior também teve registro na mesma célula. Os arquivos não permitem validar zeros, e as versões históricas de observações e meteorologia disponíveis na data de emissão não puderam ser reconstruídas. Nenhuma métrica abaixo avalia chance de ignição, fogo não detectado ou generalização global.

Foram usadas 235 semanas elegíveis de treino (2018–2021), 45 de validação (2022) e 75 de teste (2024–2025), em três células. Semanas que cruzam os cortes são excluídas e não há arquivo de 2023. A divisão é temporal; não foi realizada uma separação espacial independente. O horizonte é uma semana. Na validação e no teste, a meteorologia futura é uma climatologia regional/mensal calculada exclusivamente no treino; a reanálise anterior pode ter sido revisada posteriormente.

## Comparação pontual no teste

| Método | MAE, detecções/semana | RMSE, detecções/semana |
|---|---:|---:|
| Persistência | 120,93 | 265,91 |
| Média sazonal | 191,24 | 297,98 |
| Regressão log-linear regularizada | **105,32** | **231,14** |
| Pyro sem meteorologia, mediana | 108,69 | 240,90 |
| Pyro com meteorologia, mediana | 112,17 | 247,15 |

A regressão estatística simples teve o menor MAE e RMSE entre os métodos comparados no teste. O modelo Pyro com meteorologia superou persistência e a referência sazonal em MAE, mas não a regressão log-linear. Esses são resultados para semanas selecionadas, não para todas as semanas.

## Distribuição e ablação

As duas variantes foram especificadas antes do teste. A variante com meteorologia foi selecionada pelo CRPS de validação, 43,49 contra 45,95 sem meteorologia; essa seleção foi gravada antes da avaliação do teste. CRPS menor é melhor e está em detecções/semana. O log score negativo menor também é melhor.

| Variante | CRPS teste | Log score negativo | Cobertura 80% | Largura média 80% | Cobertura 95% | Largura média 95% |
|---|---:|---:|---:|---:|---:|---:|
| Sem meteorologia | 90,83 | **5,43** | 80,0% | 473,11 | 94,7% | 895,31 |
| Com meteorologia | **87,11** | 5,52 | 74,7% | **283,71** | 89,3% | **499,35** |

O menor CRPS com meteorologia nesta amostra não demonstra superioridade geral: a variante teve pior log score e menor cobertura. Os intervalos são mais estreitos, mas ficaram abaixo das coberturas nominais de 80% e 95%; a amostra é pequena e as semanas não são independentes. As coberturas são condicionais ao critério de inclusão do alvo e à trajetória meteorológica usada.

No Pantanal da demonstração, a variante com meteorologia teve MAE 218,06 e cobertura de 80% igual a 65,6% em 32 semanas de teste. Na célula centrada em (-10; -64,375), teve MAE 26,73 e cobertura 80% de 80,6% em 31 semanas; na célula (-7,5; -61,875), MAE 50,50 e cobertura 80% de 83,3% em 12 semanas. Nas oito semanas acima do percentil 90 de intensidade definido no treino, o MAE foi 578,63 e a cobertura do intervalo 80% foi 62,5%. Os tamanhos são pequenos, sobretudo nas caudas. Métricas por trimestre e por faixa de intensidade estão no JSON completo.

No treino, a razão amostral variância/média das contagens positivas por célula variou aproximadamente de 31 a 586. Essa dispersão agregada é compatível com a necessidade de considerar variação além de uma Poisson simples, mas também reflete sazonalidade, heterogeneidade e a seleção positiva; não é um teste formal de ajuste da distribuição. Foi usada binomial negativa truncada em zero, sem inflação de zeros.

## Exemplo real de cenário, com ressalvas

Na célula nominal centrada em (-18,5; -57,5), com emissão **simulada** em 7/1/2024, a climatologia fixa produziu média condicional de 17,12 detecções/semana e intervalo preditivo 80% de 2 a 43. Com +1 °C, -3 pontos percentuais de umidade, precipitação ×0,9 e vento ×1,05, a média foi 22,49 e o intervalo 80% de 2 a 57. São simulações do mesmo modelo; os intervalos se sobrepõem, a trajetória meteorológica não é uma previsão e a diferença não é um efeito causal. `event_probability` permanece `null`: não há base para calcular a probabilidade de ao menos uma detecção.

Os valores são aproximações Monte Carlo, com semente registrada. O resultado completo, metadados e figura estão em `runs/brasil_20261111/exports/scenario_pantanal_20240107.json` e `scenario_pantanal_20240107.png`. Consulte `runs/brasil_20261111/evaluation/metrics.json`, `selection.json` e os Parquets de previsões para métricas completas. O modo `quick_` é apenas um teste de funcionamento e não foi usado nos números desta página.
