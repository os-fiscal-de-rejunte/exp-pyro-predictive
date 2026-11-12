# Integração NASA AppEEARS — 12/11/2026

## Entrega implementada

O projeto consulta camadas oficiais de cobertura do solo (MCD12Q1), vegetação (MOD13Q1), cobertura arbórea (MOD44B) e altitude (NASADEM). O cliente autentica localmente, limita requisições, conserva tarefas retomáveis, valida os downloads por checksum e não guarda credenciais no repositório. A consulta preditiva pode anexar o contexto geográfico e gerar um relatório HTML, JSON e GeoJSON. Não é um mosaico completo: nove amostras representam cada célula MODIS de aproximadamente 1.443 km².

O token foi configurado com sucesso no cofre Windows. Para renová-lo, use `scripts/configurar_appeears.py`; não envie a senha pelo chat. O token AppEEARS é distinto do token Earthdata de outros serviços.

## Execução real verificável

A demonstração em latitude −21,229167, longitude −54,341825 contém quatro CSVs NASA e 1.719 observações em formato longo, referentes a nove coordenadas. Após os filtros e atrasos documentados, em 12/11/2026 as médias amostrais são NDVI 0,6604, cobertura arbórea 21,56% e altitude 450,44 m. Não são médias de todos os pixels nem medições meteorológicas nesse ponto.

O pacote offline está em `artifacts/geography_demo`: `index.html`, gráfico, dados estruturados, CSVs, metadados, auditoria e manifesto SHA-256. Os 144 valores de vegetação recusados pela qualidade continuam rastreáveis; não viram observações de ausência de fogo. CSVs AppEEARS já vêm escalados, e o pipeline não aplica novamente o fator NDVI.

```powershell
.\.venv\Scripts\python.exe scripts/geografia_local.py --latitude -21.229167 --longitude -54.341825 --reference-date 2026-11-12
.\.venv\Scripts\python.exe scripts/testar_expansao_real.py --latitude -21.229167 --longitude -54.341825 --geography --geography-view runs/minha_consulta
```

A consulta real de integração conservou a probabilidade v6 de 24,02% e anexou geografia com estado `ok`. Esse número estima detecção MOD14A2 no próximo composto de até oito dias, condicionada à máscara inteiramente válida; não é uma probabilidade incondicional de incêndio. A simples exibição das camadas não muda a previsão: `primary_prediction_uses_geography` permanece `false`.

## Experimento científico

Foi pré-selecionado um piloto de 24 células em seis continentes, com nove pontos por célula, 216 coordenadas. Os IDs foram escolhidos pela ordem e pelo suporte do treino, preservando grupos espaciais. Extrações históricas 2018–2025; rótulos MOD14A2 2020–2025 já existentes. A extração completa com NDVI permaneceu em processamento; preparou-se uma variante anual mais leve, sem NDVI/EVI, reutilizando os trabalhos de relevo.

Em 15:28 UTC, após liberação do primeiro lote anual de 100 pontos, foi decidido usar somente células/datas com pelo menos cinco amostras válidas de cobertura do solo ou NDVI. A decisão antecedeu a comparação nova e foi congelada por hashes em `runs/geography_annual_v1/cohort_policy.json`. Não se acrescentam posteriormente novos dados ao modelo concluído. A seleção por disponibilidade é uma limitação, não representatividade global.

As duas regressões probabilísticas Pyro/PyTorch compartilham coordenadas, sazonalidade e interações latitude × estação; uma recebe também geografia. Treino 2020–2022, validação 2023 e teste 2024–2025. Regiões reservadas não entram no ajuste ou na escolha pelo Brier. Imputação, normalização e identificação de variáveis ausentes são aprendidas somente no treino. Esse experimento não inclui clima atual; não é uma nova validação conjunta de todas as fontes. O teste histórico v6 já era conhecido: os resultados são retrospectivos.

O experimento foi concluído em 15:31 UTC. Coorte final: **2.597 exemplos em 11 células da África (4), Ásia (4) e Europa (3)**. Seis células de ajuste e cinco reservadas. O ajuste efetivo usou 720 exemplos de treino; a seleção usou 234 exemplos de validação das regiões de ajuste. O teste tem 844 exemplos, 209 positivos, dos quais 380 exemplos e 88 positivos pertencem às cinco regiões reservadas. Os 1.311 registros classificados temporalmente como treino incluem regiões reservadas que **não** entram no ajuste.

Foram executados exatamente dois treinamentos Pyro de 1.200 passos. A variante geográfica venceu na validação pelo Brier (0,16403 contra 0,17175), e a escolha foi salva antes de abrir o teste. Resultados no teste comum:

Neste experimento geográfico, a emissão retrospectiva é no início do composto; usa-se somente geografia com disponibilidade assumida anterior a essa data. As métricas avaliam o composto de até oito dias, com antecedência zero. Antecipar sua emissão em vários dias, como pode ocorrer na consulta sazonal, requer outra validação e é sinalizado no candidato por `additional_lead_not_validated`.

| Método | Acurácia, limiar 0,5 | Recall | Brier, menor é melhor |
|---|---:|---:|---:|
| Frequência de treino | 75,24% | 0,00% | 0,18663 |
| Sazonal mensal | 75,24% | 0,00% | 0,18234 |
| Persistência do último evento conhecido | 73,46% | 39,71% | 0,25451 |
| Pyro sem geografia | 77,25% | 23,44% | 0,17572 |
| Pyro com geografia | 78,08% | 29,67% | 0,17852 |

Os baselines foram calculados no mesmo recorte, usando apenas treino para frequências. Essa análise é um diagnóstico posterior à seleção congelada; não foi usada para escolher outro vencedor após observar o teste. O ganho de 0,83 ponto percentual de acurácia **não demonstra melhora probabilística**: o Brier e a log loss pioraram com geografia (log loss de 0,64585 para 0,82219). Nas regiões reservadas, o Brier piorou de 0,21278 para 0,21628 e o recall caiu de 10,23% para **0%**. Nesse grupo, a variante geográfica classificou todos os casos como negativos no limiar 0,5. A frequência simples obteve Brier 0,17911, superior aos dois Pyro nessas regiões.

Por continente no pequeno teste geográfico: África, acurácia 91,26% e recall 0%; Ásia, 65,08% e recall 12,26%; Europa, 72,68% e recall 69,01%. A alta acurácia africana resulta de muitos negativos, não de boa detecção dos positivos. Não substitui a avaliação africana v6, que usa outra coorte, nem estabelece acurácia continental completa.

**Decisão: manter v6 como saída principal.** O novo modelo é um candidato experimental separado. Fora das faixas das features de treino, esse candidato retorna `out_of_distribution` e estimativas `null`, preservando a resposta principal. Na demonstração brasileira, a geografia foi recuperada, mas o candidato geográfico foi corretamente recusado por extrapolação. O JSON indica `global_accuracy_established: false`.

Pesos, exemplos, predições, métricas, seleção congelada, CSVs, metadados, ambiente, proveniência e checksums estão em `artifacts/geography_annual_v1` (aproximadamente 0,9 MB). Gráficos: `evaluation.png` e `reliability.png`. Consulte também `docs/model_card_geography.md`.

Teste completo offline, sem conta NASA, novas consultas ou retreinamento:

```powershell
.\.venv\Scripts\python.exe scripts/testar_geografia_real.py
.\.venv\Scripts\python.exe scripts/avaliar_geografia.py --model-dir runs/geography_annual_v1/models
```

O primeiro comando salva `runs/geography_demo_offline/forecast.json` e `view/index.html`. O segundo reproduz os diagnósticos no diretório local da execução. Para conferir as métricas do pacote, use um diretório de saída próprio antes de regenerar figuras; não altere os artefatos congelados.

## Verificações executadas

- Suíte final: **97 testes aprovados em 31,77 segundos**, incluindo testes sintéticos identificados, reprocessamento dos CSVs reais, recálculo das métricas e recusa da extrapolação geográfica. A suíte anterior também passou com 94 testes.
- Os **31 arquivos originais de DATA** tiveram seus hashes recalculados e permaneceram idênticos ao inventário anterior.
- Verificador v6 aprovado: artefatos e 898 células de features preservados; 498 células liberadas, 465 atualmente projetáveis. Essa verificação usou `--skip-originals`, e a conferência dos originais foi feita separadamente.
- Pacote geográfico offline conferido por manifesto, unidades, qualidade, datas e agregações reais.
- Dois segredos locais conferidos contra os arquivos preparados para commit, sem exposição nos logs e sem ocorrência no código versionado.

Os resultados dos testes sintéticos comprovam contratos de software; não são evidência de desempenho preditivo real.

## Limites e retomada

Metadados de produtos reprocessados não comprovam sua data histórica de publicação. Os atrasos conservadores são hipóteses declaradas; `availability_verified` é `false`. NASADEM não cobre todo o globo e amplitude de altitude entre nove pontos não equivale a declividade. Cobertura arbórea não distingue sozinha mata nativa, combustível seco ou causa do fogo.

O cliente limita três tentativas para falhas transitórias, não repete submissões ambíguas e interrompe em 429. O cache deve ser usado em série. O orçamento local separa uma janela inicial de 90 minutos, encerrada sem treinamento, e outra de encerramento de 90 minutos dentro das oito horas gerais autorizadas. CPU limitada a duas threads, memória a 60% da disponibilidade inicial da janela e novos arquivos a 1 GB. As tarefas NASA remotas já aceitas podem continuar no serviço; isso não significa haver processo local em segundo plano.

O treinamento registrou pico amostrado de memória de aproximadamente 343 MB, abaixo do limite de 1,23 GB da janela de encerramento. Esse pico se refere aos processos registrados no orçamento, não a toda a memória do computador. A expansão geográfica maior permanece separada e não altera a cobertura validada v6 de 0,140922% da superfície global.

Para iniciar uma retomada explícita da coleta completa em outra execução, preservando o cache:

```powershell
.\.venv\Scripts\python.exe scripts/treinar_geografia.py --run-dir runs/geography_pilot_v1 --config config/geography.json --budget-id geography_full_resume_01 --collect-only --no-wait
```

Mantenha o mesmo ID durante a execução. Não gere IDs automaticamente para contornar o tempo. Se o modelo anual parcial já estiver concluído, o comando correspondente retorna o resultado congelado sem retreinar. Um experimento maior exige outro diretório e orçamento explícito.

O próximo passo científico de maior impacto é concluir a coleta geográfica maior e avaliar em regiões e períodos novos, com meteorologia alinhada e disponibilidade histórica verificável. Uma melhora neste pequeno recorte não estabelece acurácia mundial.

Fontes e contratos: [guia AppEEARS](docs/appeears_geografia.md), [decisões](docs/decisions.md), [API NASA](https://appeears.earthdatacloud.nasa.gov/api/). Consultas oficiais realizadas em 12/11/2026.
