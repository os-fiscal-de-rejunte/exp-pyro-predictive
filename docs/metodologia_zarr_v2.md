# Metodologia do experimento Zarr v2

## Pergunta e unidade

Para cada célula selecionada de 0,5° × 0,625°, emite-se uma distribuição para a contagem de registros VIIRS/NOAA-20 na semana UTC seguinte. O alvo é **condicional**: a semana-alvo tem pelo menos um registro VIIRS e a semana anterior contém registros tanto de VIIRS/NOAA-20 como de MODIS/Terra. Não se estima a probabilidade de a semana-alvo ser positiva. O horizonte é uma semana de segunda a domingo; a emissão simulada ocorre no domingo anterior.

Não se inferem incêndios independentes, ignições, área queimada, propagação ou dano. Como `type` não é filtrado, o alvo inclui anomalias térmicas que podem não ser fogo de vegetação. Ausência de linha não equivale a zero: a cobertura, condições de observação e completude de cada extração não são conhecidas.

## Região e divisões

As 16 células foram escolhidas deterministicamente entre 1.411 células elegíveis na partição Brazil, usando 2018–2021 e pelo menos 25 semanas com registros coincidentes dos sensores, com estratificação por blocos geográficos. São áreas selecionadas por atividade observada, não amostra representativa do Brasil. Quatro células foram reservadas antes do ajuste; sua elegibilidade usou o próprio histórico anterior, de modo que sua avaliação espacial é condicional a essa seleção.

Treino: semanas encerradas até 2021-12-31 em 12 células, 683 exemplos. Validação temporal: 2022, 148 exemplos nas células de treino e 56 nas reservadas. Teste: 2024–2025, 329 exemplos nas células de treino e 74 nas reservadas; 2023 não tem arquivo VIIRS. As 196 semanas das células reservadas no período de treino não participam dos ajustes. Oito semanas que cruzam limites temporais foram excluídas. As previsões retrospectivas usam origens semanais sucessivas; intervalos que cruzam as divisões não entram como alvo de dois conjuntos.

## Fontes e variáveis

Os arquivos FIRMS foram lidos em lotes, com hashes, datas/horas, versões, plataforma e campos originais preservados. Somente duplicatas exatas são removidas; proximidade espaço-temporal não identifica um incêndio. MODIS/Terra e VIIRS/NOAA-20 têm séries separadas.

O recorte NASA POWER diário UTC inclui `T2M`, `T2M_MAX`, `T2M_MIN` em °C; `RH2M` em %; `PRECTOTCORR` em mm/dia; `WS10M` em m/s; e `GWETROOT`, `GWETTOP` como índices adimensionais de umidade modelada do solo. Na agregação semanal, precipitação é somada e as demais variáveis são promediadas. No caso de `T2M_MAX` e `T2M_MIN`, trata-se da média semanal dos extremos diários, não dos extremos de toda a semana. A grade é uma representação meteorológica grosseira, não observação exata no ponto de um hotspot.

O histórico climático usa a semana terminada pelo menos sete dias antes da emissão e janelas completas anteriores de quatro semanas; a semana imediatamente anterior da atividade inclui as contagens de cada sensor. Seno e cosseno do ano e coordenadas compõem as demais entradas. Há normalização ajustada somente no treino.

A variante meteorológica aprende, nos exemplos de treino, também uma relação condicional com o clima observado da semana-alvo. Na validação e teste, esses valores futuros reais são mascarados e substituídos por climatologia mensal estimada apenas nas 12 células e no período de treino. Para células reservadas, usa-se climatologia agrupada das células de treino. Isso introduz uma diferença entre condições de treino e de inferência e não equivale a validação usando previsão meteorológica histórica. POWER e os arquivos finais de satélite podem ter sido revisados após as datas retrospectivas; a disponibilidade exata à época não foi comprovada.

## Harmonização e previsão

A comparação simples usa a mediana, calculada no treino, da diferença `log(1 + VIIRS) - log(1 + MODIS/Terra)` em semanas positivas nos dois sensores. Um modelo hierárquico exploratório em Pyro usa intercepto, componente sazonal e efeitos regionais com compartilhamento parcial; seu posterior e a mediana constituem comparações na escala logarítmica, não contagens verdadeiras. A documentação por dia/noite, confiança e tipo é mantida por produto, sem converter suas escalas. O modelo condicional não identifica separadamente atividade real e detectabilidade com cobertura desconhecida. Concordância entre sensores não valida a realidade subjacente.

O preditor é uma regressão binomial negativa bayesiana truncada em zero, com PyTorch/Pyro e inferência variacional, escolhida para representar contagens positivas sobredispersas. Dois modelos idênticos no recorte de treino são ajustados com e sem as variáveis meteorológicas. As contagens defasadas de cada sensor entram diretamente como variáveis explicativas. A harmonização é um diagnóstico separado: sua incerteza **não** é propagada como estado latente na previsão. Isso evita transformar uma pseudo-contagem harmonizada em verdade, mas não corrige a falta de identificabilidade.

Persistência, referência sazonal e regressão log-linear regularizada são avaliadas nas mesmas semanas. A escolha foi registrada antes do teste pelo CRPS da validação temporal nas 12 células de treino; a variante com clima venceu nesse critério, mas não no teste. O teste também mede regiões reservadas, estação, intensidade e versões de sensor. `MAE`/`RMSE` estão em detecções por semana; CRPS avalia a distribuição e é melhor quando menor; cobertura e largura descrevem intervalos preditivos de observações condicionais, não intervalos para a quantidade verdadeira de incêndios.

## Cenários e rastreabilidade

`earthfire.dual_scenario.predict_week` reutiliza modelos e transformações salvos, recupera histórico e recebe alterações de temperatura em °C, umidade em pontos percentuais e multiplicadores para precipitação e vento. Sem um arquivo válido de previsão futura, usa climatologia fixa; a umidade do solo permanece fixa. A mesma função probabilística avalia os dois cenários. As diferenças são condicionais, não causais; o contrato retorna probabilidades de ocorrência como `null`. A faixa de extrapolação é um critério prático baseado nos extremos do treino mais um desvio-padrão, não garantia de calibração.

Os exemplos, configurações, versões, seeds, cache Zarr, checkpoints, métricas e contratos de saída ficam sob `runs/zarr_brasil_v2_20261111/`. A ferramenta testa compatibilidade por hashes antes de retomar um modelo. Outros ambientes e versões de bibliotecas podem alterar resultados numéricos, mesmo com a mesma semente.
