# Model card — candidato geográfico AppEEARS

Versão: `geography_annual_v1`, treinada em 12/11/2026. Estado: **experimental, sem promoção operacional**. Pesos e manifesto: `artifacts/geography_annual_v1`.

## Evento e suporte

Probabilidade de ao menos um pixel MOD14A2 classificado como foco no próximo composto de até oito dias, condicionada a todos os 41×41 pixels da célula possuírem máscara válida. Não estima ignição, incêndios independentes, área queimada, propagação ou danos. A área é aproximadamente 1.443 km²; as nove amostras geográficas não representam todos os seus pixels. Oportunidade futura de observação é desconhecida.

## Entradas e inferência

Regressão Bernoulli em Pyro/PyTorch, prior normal regularizador e posterior variacional AutoNormal. Coordenadas, seno/cosseno sazonal, interações com latitude; opcionalmente frações das 17 classes MCD12Q1, porcentagens MOD44B e média/amplitude NASADEM. Imputação, escala e indicadores de ausência aprendidos somente no treino. NDVI/EVI não estavam nessa extração anual e são ignorados pelo modelo, mesmo quando exibidos na consulta.

Não usa clima atual nem previsão meteorológica futura. Os modelos meteorológicos anteriores são preservados separadamente. Intervalo posterior de 95% representa incerteza variacional dos parâmetros, não toda a incerteza de transferência espacial. Intervalo preditivo Bernoulli é sobre o evento binário; nenhum deles é intervalo de número de incêndios.

## Dados e divisões

Coorte por disponibilidade congelada antes da comparação nova: 11 células, África/Ásia/Europa, 2.597 exemplos. Treino temporal 2020–2022, validação 2023, teste 2024–2025. Ajuste efetivo: 720 exemplos em seis células; seleção: 234 exemplos nas mesmas regiões; cinco células inteiras reservadas. Teste: 844 exemplos, sendo 380 nas regiões reservadas. Seleção pelo Brier da validação, salva antes do teste. O teste histórico v6 já era conhecido; não é confirmação independente ou prospectiva.

A emissão retrospectiva deste experimento ocorre no início do composto: features geográficas com disponibilidade assumida estritamente anterior a `period_start`, e alvo no composto que começa naquele instante (antecedência zero, horizonte de até oito dias). A consulta pode selecionar um composto que começa dias depois; nesse caso usa apenas geografia disponível na emissão, mas essa antecedência adicional não foi validada. O JSON distingue `evaluation_lead_days: 0` de `additional_lead_not_validated`.

## Resultados e decisão

Com geografia: acurácia 78,08%, recall 29,67%, Brier 0,17852. Sem geografia: 77,25%, 23,44%, 0,17572. A geografia melhorou acurácia nesse recorte, mas piorou a qualidade probabilística. Nas regiões reservadas, o candidato geográfico teve recall 0% no limiar 0,5 e Brier 0,21628, inferior à frequência simples, Brier 0,17911. **Não sustenta melhorar previsões em todo o globo.** Detalhes por continente e baselines em `REPORT_APPEEARS.md` e nos JSONs do pacote.

A previsão principal v6 continua preservada. O candidato tem `deployment_status: experimental_only` e `global_accuracy_established: false`; fora das faixas aprendidas, retorna estado `out_of_distribution` com estimativas `null`. Isso não transforma o candidato em um detector confiável de todos os tipos de extrapolação.

## Limitações e uso

Seleção por disponibilidade e poucas regiões; produtos reprocessados; datas originais de publicação não verificadas; atrasos de disponibilidade assumidos; amostragem espacial esparsa; dados estáticos/anuais, sem meteorologia atual; posterior aproximado pode subestimar incerteza; nenhum teste operacional prospectivo. A precisão alta em grupos com poucos positivos pode esconder falha em detectar fogo. Não usar para alertas de segurança, decisões de emergência ou afirmações causais.

Uso recomendado: estudar camadas, testar contratos, reproduzir diagnósticos e orientar a próxima coleta. Próximo experimento: mais regiões, clima alinhado e períodos novos, preservando uma avaliação espacial independente. Nenhuma garantia de ganho com novas features.
