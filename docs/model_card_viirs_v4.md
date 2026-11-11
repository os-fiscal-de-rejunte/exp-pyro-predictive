# Model card — VIIRS diário com meteorologia v4

**Modelo:** `vj114a1_daily_weather_v4b`, regressão logística bayesiana em Pyro/PyTorch. O checkpoint `runs/viirs_daily_v4b_20261111/models/mask_weather.pt` tem SHA-256 `b316b22b569e5915c4f3e11ed00c2c7004c8b375517db3e132bef2fe504cecb0` e registra configuração, dados, código, estado probabilístico e transformações.

**Evento estimado:** ao menos um pixel de foco em um composto diário `VJ114A1.002`/VIIRS NOAA-20 de 1 km, na área de 41 × 41 pixels, condicionado à máscara futura ser válida em todos os pixels. As classes de nuvem, dados não processados e ausência não são negativas. Não é uma probabilidade de ignição ou de fogo não observado; o produto não é o FIRMS de 375 m.

**Dados:** seis áreas brasileiras para ajuste e duas reservadas, amostra sistemática de 2018–2025. O treino usou 230 exemplos com 25 eventos, a validação 130 com dez e o teste 131 com 14. Houve 570 registros área-dia com cobertura insuficiente, excluídos dos rótulos. A amostra de validação foi ampliada por calendário depois de uma contagem preliminar revelar poucos positivos; o teste não participou dessa decisão.

**Entradas:** sazonalidade, posição, histórico disponível de focos MODIS e VIIRS/FIRMS, indicadores de ausência e histórico NASA POWER de temperatura, umidade, precipitação, vento e umidade modelada do solo. A última data meteorológica é D−3 para alvo em D. Os históricos de satélite podem ser versões reprocessadas posteriormente e não provam disponibilidade operacional passada.

**Desempenho:** no teste, Brier 0,09088 contra 0,08628 da referência sazonal e 0,08380 de uma referência de transição a partir de MODIS. Uma regressão estatística com clima obteve 0,09132 contra 0,09966 da mesma referência sem clima; isso não é ablação exata do modelo Pyro. Na validação, as áreas de treino não atingiram o critério de liberação; o grupo reservado o atingiu com apenas 37 exemplos e cinco positivos. Resultados detalhados em [REPORT_VIIRS_V4.md](../REPORT_VIIRS_V4.md).

**Uso e limitações:** pesquisa e demonstração retrospectiva identificada. O grupo espacial pequeno não autoriza extrapolação para o Brasil ou outros locais. A disponibilidade futura da máscara não é conhecida e os intervalos são sobre parâmetros, não sobre contagens. Não usar para alertas operacionais, evacuação, propagação, área queimada ou dano. A variante v4 só usa meteorologia anterior e não implementa resposta a uma trajetória meteorológica futura.
