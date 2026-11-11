# Relatório — cobertura VIIRS/NOAA-20 e modelo diário v4

Execução de 11/11/2026. O acesso Earthdata fornecido pelo usuário foi utilizado em memória para obter arquivos protegidos da NASA; a credencial não faz parte dos arquivos do projeto nem dos relatórios. O experimento anterior continua em [REPORT_NASA_V3.md](REPORT_NASA_V3.md).

## Dados obtidos e verificados

- Foram baixados 1.287 grânulos `VJ114A1.002`: 1.016 da amostra principal, 128 datas adicionais de validação e 143 tiles de borda. Os hashes de todos foram verificados. A pasta da execução ocupa cerca de 450 MB.
- A amostra principal utiliza os dias 1, 8, 15 e 22 de janeiro, abril, julho e outubro de 2018–2025, para oito áreas do Brasil. O catálogo não retornou 01/01/2018 nas oito áreas; essas oito lacunas foram registradas.
- Para aumentar a quantidade de exemplos de validação, foram acrescentados os mesmos dias de fevereiro, maio, agosto e novembro de 2023. Essa ampliação foi definida após verificar que a contagem agregada de positivos na validação preliminar era pequena, e antes do treinamento. Datas individuais e teste de 2024–2025 não foram escolhidos pelo resultado.
- A área `norte` cruza dois tiles. O mosaico foi conferido com dados reais; os 287 pixels que faltavam no arquivo central passaram a ser obtidos do tile adjacente. Se faltar um tile, os pixels ficam desconhecidos.
- Os grânulos geraram 1.144 registros área-dia, dos quais 570 tinham cobertura/classificação insuficiente e não foram usados como rótulo. Não houve pixels sem tile após o mosaico nem classes inválidas explícitas detectadas pelo leitor.

`VJ114A1.002` é um composto **diário de 1 km**, derivado do VIIRS/NOAA-20. O campo `FireMask` diferencia classes de foco, não fogo, nuvem e dados indisponíveis. Ele não é a máscara das detecções FIRMS de **375 m** e não demonstra observação contínua. O evento modelado é ao menos um pixel classificado como foco no recorte durante um dia, **condicionado a todos os pixels terem classes válidas**. Isso não equivale a chance de ignição, incêndio presente, propagação ou dano.

Também foram obtidos 22 grânulos diários de **01–22/10/2026** para a área centrada em (-11,5°, -46,875°), sem dias de arquivo faltante nesse recorte. Em 11 dias a área foi integralmente classificada; em 11 houve cobertura insuficiente. Entre os dias integralmente classificados, seis apresentaram algum pixel de foco. Isso não comprova completude para o Brasil ou para todo o ano de 2026. Esses arquivos não foram usados no treinamento ou no teste v4. Inventário: `runs/viirs_oct2026_20261111/recent_coverage/`.

## Modelo, recorte e separação

O modelo é uma regressão logística bayesiana em Pyro/PyTorch, com latitude, longitude, sazonalidade, histórico separado de MODIS e de registros positivos VIIRS/FIRMS, além de temperatura, umidade, precipitação, vento e umidade modelada do solo do NASA POWER. O clima é estritamente retrospectivo; não foram apresentadas observações futuras como previsão meteorológica. As transformações foram ajustadas somente no treino.

| Conjunto | Áreas | Exemplos elegíveis | Positivos |
|---|---:|---:|---:|
| Treino 2018–2022 | 6 | 230 | 25 |
| Validação 2023 | 6 de treino + 2 reservadas | 130 | 10 |
| Teste 2024–2025 | 6 de treino + 2 reservadas | 131 | 14 |

Os registros de treino das duas áreas reservadas não participam do ajuste. Os dados são reprocessados e podem não ter estado disponíveis na data histórica simulada; o experimento não valida um serviço operacional em tempo real. A amostra é pequena, sistemática e não representa todo o Brasil.

## Resultados reais

Brier mede erro quadrático das probabilidades: menor é melhor. Todas as referências e o modelo foram avaliados nos mesmos 131 exemplos elegíveis de teste, com 14 positivos.

| Referência ou modelo | Brier no teste |
|---|---:|
| Transição a partir do MODIS anterior | 0,08380 |
| Referência sazonal | 0,08628 |
| Modelo Pyro com clima | 0,09088 |
| Regressão logística com clima | 0,09132 |
| Frequência global / persistência disponível | 0,09546 |
| Regressão logística sem clima | 0,09966 |

O **modelo Pyro não superou a referência sazonal no agregado**. A comparação entre as regressões estatísticas com e sem clima favoreceu a versão com clima nesse teste; isso não é uma ablação exata da inferência bayesiana nem prova causal do efeito meteorológico. A persistência coincidiu com a frequência global porque não havia máscaras anteriores elegíveis para estimar transições na amostra de treino sob a regra de data de produção.

Como cálculo descritivo posterior, usando 50% como limiar convencional para classificar um evento, a acurácia foi **87,8%** (115 de 131); a precisão dos alertas positivos foi **41,7%** (5 de 12), e a sensibilidade foi **35,7%** (5 de 14 eventos). Uma regra que classificasse todos os casos como negativos teria 89,3% de acurácia, pois a amostra é desbalanceada. Esse limiar não foi calibrado como critério operacional e os números incluem somente os casos com máscara válida.

Os oito recortes de 41 × 41 pixels de aproximadamente 926,6 m abrangem nominalmente cerca de 11.547 km², ou **0,0023% da superfície total terrestre**; seis recortes usados no treino abrangem aproximadamente 0,0017%. Essas proporções representam somente a área das amostras, não cobertura temporal contínua ou capacidade preditiva validada fora delas.

Na validação das seis áreas de treino, o Brier do modelo foi 0,06235 contra 0,05761 da sazonal: o critério pré-definido não foi atingido e a função de demonstração não libera a probabilidade para esse grupo. Nas duas áreas reservadas, a validação teve Brier 0,09691 contra 0,11477 da sazonal, em 37 exemplos e cinco positivos; o critério mínimo foi atingido, mas esse grupo pequeno não valida genericamente áreas novas. No teste reservado espacial, os valores foram 0,10360 e 0,12197, respectivamente, em 35 exemplos e cinco positivos. Métricas por área, mês, log loss e calibração estão em `runs/viirs_daily_v4b_20261111/evaluation/`.

Os resultados v3 e v4 usam alvos e períodos distintos e não devem ser comparados diretamente por Brier. A incerteza da disponibilidade futura da máscara permanece sem identificação; `event_probability` continua `null`. Intervalos sobre parâmetros não são intervalos preditivos de contagem.

## Teste com dados reais

```powershell
.\.venv\Scripts\python.exe scripts\testar_viirs_real.py
```

O exemplo usa uma área reservada em (-31°, -52,5°) e simula a previsão para 22/10/2025. A probabilidade condicional foi 2,9%, com intervalo de 80% sobre parâmetros de aproximadamente 0,5%–6,1%; no produto, os 1.681 pixels foram classificados e nenhum foi marcado como foco. Um caso não mede calibração, e esse valor não é chance de incêndio real. O resultado estruturado está em `runs/viirs_daily_v4b_20261111/evaluation/exemplo_holdout_viirs.json`. Outros exemplos salvos podem ser escolhidos por `--latitude`, `--longitude` e `--data`; se o grupo não satisfaz a validação ou se a máscara é desconhecida, a saída indica insuficiência.

## Verificações e retomada

O piloto real e testes sintéticos verificaram mosaico, datas, nuvens, dados ausentes, autenticação, falhas e salvamento de modelos. A suíte final teve **52 testes aprovados**. `scripts/verificar_viirs_v4.py` releu os 1.287 grânulos e confirmou hashes, divisão temporal, datas das features, coerência dos eventos e 131 probabilidades de teste. Os 31 arquivos originais de `DATA/` foram novamente lidos e seus hashes conferiram com as auditorias. O checkpoint completo foi reaberto sem novo treinamento. A verificação local de 258 arquivos textuais da execução não encontrou a credencial fornecida. Durante a coleta houve uma falha de rede e uma resposta do catálogo truncada; a primeira foi retomada e a segunda foi resolvida com consultas trimestrais. Os registros persistentes estão nos manifestos e no `pipeline.log`.

Para retomar com os arquivos já obtidos, na raiz do projeto:

```powershell
$env:PYTHONPATH = 'src'
$env:EARTHFIRE_BUDGET_ID = 'minha_nova_janela'
.\.venv\Scripts\python.exe -m earthfire.viirs_daily_data
.\.venv\Scripts\python.exe -m earthfire.viirs_daily_model readiness
.\.venv\Scripts\python.exe -m earthfire.viirs_daily_model baselines
.\.venv\Scripts\python.exe -m earthfire.viirs_daily_model train --steps 1000
.\.venv\Scripts\python.exe -m earthfire.viirs_daily_model evaluate --samples 400
.\.venv\Scripts\python.exe scripts\verificar_viirs_v4.py
.\.venv\Scripts\python.exe -m pytest -q
```

O checkpoint completo compatível é reutilizado. Para baixar novamente, os comandos `earthfire.viirs_daily_yearly_collect`, `earthfire.viirs_daily_validation` e `earthfire.viirs_daily_halo` usam `EARTHDATA_TOKEN` somente a partir do ambiente; não inclua a credencial em arquivos ou no comando. Novos parâmetros ou insumos exigem uma nova execução e novo orçamento. Os três treinamentos completos previstos para esta janela já foram usados.

Fontes, hipóteses e contratos: [metodologia](docs/metodologia_viirs_v4.md), [dicionário](docs/dicionario_viirs_v4.md), [decisões](docs/decisions.md) e [fontes NASA](docs/fontes.md). Próximos passos: obter observações por passagem e cobertura do produto 375 m, ampliar áreas/datas, construir histórico de previsões meteorológicas realmente emitidas e avaliar calibração prospectiva. O classificador v4 usa clima anterior e não implementa um cenário de meteorologia futura; o simulador v2 permanece separado e experimental.
