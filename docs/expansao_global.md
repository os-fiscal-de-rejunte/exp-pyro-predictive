# Expansão multicontinental v5

## Escopo e definição

Esta implementação amplia o processamento e a avaliação para 12 recortes em seis continentes. A versão VIIRS diária v4 permanece independente. O novo alvo é **ao menos um pixel com foco no composto MOD14A2, de até oito dias, condicionado à máscara integralmente válida no recorte**. Não é uma probabilidade global de ignição. O composto final do ano pode ser menor que oito dias.

A mudança de referência foi necessária porque a máscara VIIRS 375 m, compatível com FIRMS, ainda não foi obtida para o conjunto mundial. Usar MODIS como oportunidade de observação de VIIRS criaria negativos inválidos. A expansão MODIS é um experimento próprio; seus números não podem ser comparados diretamente com a acurácia da v4 diária.

## Dados e divisões

- Máscaras NASA MODIS/Terra pela API pública ORNL, 2020–2025, consultas de no máximo dez datas, recortes de aproximadamente 40 km. Respostas brutas, metadados, produção e hashes ficam no cache.
- Registros MODIS e VIIRS/NOAA-20 em ZIPs locais, 2019–2025 quando presentes. Extração por país, em blocos de 10 mil linhas, com validação de coordenadas, aquisição e confiança específica do produto. As cópias `(1)` não são reprocessadas. O inventário identifica duplicatas por SHA-256. VIIRS 2023 não está nos ZIPs disponíveis; ausência de registros na janela continua desconhecida.
- NASA POWER Daily UTC: T2M (°C), RH2M (%), PRECTOTCORR (mm/dia), WS10M (m/s) e GWETROOT (adimensional). Unidades verificadas na resposta real. Mesmo produto meteorológico no treino e na consulta histórica; não há correção entre fontes.
- O JSON local `global_countries.json` contém indicadores de cenários/anomalias; não foi usado como clima diário observado. Sua origem não foi confirmada.
- Treino: 2020–2022 em oito áreas. Validação: 2023. Teste: 2024–2025, incluindo quatro áreas reservadas. Compostos que atravessam os cortes ficam excluídos. Há áreas de treino em todos os seis continentes; isto não constitui teste de um continente inteiramente desconhecido.
- Seleção intencional de localidades, sem representar proporcionalmente biomas ou países. Natural Earth 1:110m fornece somente contexto administrativo aproximado; estados e fronteiras precisas permanecem pendentes.

## Grade e observabilidade

`global_grid.locate` reconhece coordenadas finitas em todo o globo, trata os polos e o antimeridiano e retorna célula MERRA-2, geometria e área. A célula meteorológica tem 0,5° × 0,625° e sua área varia com a latitude. A estimativa fica vinculada à geometria sinusoidal **real** do recorte MODIS, não a toda a célula climática. A consulta verifica ambos.

Todas as classes válidas do produto devem estar presentes sem pixels desconhecidos para que um zero seja elegível. Nuvens e classificação incompleta permanecem fora dos rótulos. Datas solicitadas sem imagem aparecem em `coverage_gaps.parquet` como `missing_unknown`. Zero de detecções não comprova ausência de fogo.

Os dados por país são usados como histórico positivo e não como comprovação de cobertura; uma célula pode cruzar fronteiras e o extrator não estabelece completude dos registros de países vizinhos. Contagens por sensor não são somadas nem interpretadas como incêndios independentes. `sensor_counts.parquet` é um diagnóstico descritivo, sem conversão universal MODIS/VIIRS. A harmonização probabilística já existente no projeto não é declarada calibrada globalmente por este piloto.

## Features, modelos e seleção

As features incluem posição, seno/cosseno da sazonalidade, última máscara MODIS produzida antes da emissão, detecções VIIRS anteriores e indicadores de ausência. O clima usa oito dias terminados três dias antes da emissão. Este atraso é uma hipótese conservadora; sem arquivo de disponibilização histórica, não demonstra reprodução operacional completa. Não se usa clima futuro observado.

São treinadas duas regressões Bayesianas Bernoulli em Pyro/PyTorch, com e sem clima, com regularização e transformações aprendidas somente no treino. Persistência, frequência global, sazonalidade e regressões logísticas regularizadas usam os mesmos exemplos. O vencedor é selecionado por Brier na validação das áreas de treino **antes** da avaliação de teste. Um baseline pode ser o modelo publicado. O teste não altera a política de seleção ou liberação.

A liberação por área usa somente validação: pelo menos 30 exemplos e cinco de cada classe, erro médio de calibração ≤0,10 e Brier não superior ao sazonal. São requisitos demonstrativos, não uma certificação. Valores meteorológicos fora dos limites do treino produzem `out_of_distribution`. Demais locais e datas sem features preparadas recebem `insufficient_data` e probabilidade nula.

Relatam-se Brier, log loss, ROC AUC, average precision, calibração, acurácia no limiar 0,5 e desempenho por região/continente. O Brier e sua diferença para o sazonal têm intervalos por bootstrap de blocos regionais; isso não mede incerteza sobre todas as regiões do mundo.

## Incerteza e consulta

`predict(latitude, longitude, reference_date, run_dir)` carrega artefatos prontos. Não retreina. Registros, código e artefatos têm integridade verificada. A consulta atual é uma reconstrução histórica das datas preparadas; não há serviço operacional de atualização em tempo real implementado nesta expansão.

O intervalo sobre a probabilidade é obtido da posterior variacional do Pyro, da aproximação de Laplace da logística ou da posterior Beta do baseline selecionado. No baseline, o compartilhamento do valor global é um procedimento empírico e sua incerteza não é propagada integralmente. Os intervalos preditivos do evento binário, com suporte {0,1}, são separados dos intervalos sobre a probabilidade.

O calendário usa somente imagens anteriores já produzidas, com referência do mesmo mês e ao menos dez observações anteriores; períodos insuficientes têm referência/anomalia ausentes. Anomalia não é atribuição causal.

## Execução e retomada

```powershell
$env:PYTHONPATH = 'src'
$env:EARTHFIRE_BUDGET_ID = 'global_v5_20261111'
.\.venv\Scripts\python.exe -m earthfire.global_pipeline records --config config/global_v5.json --run-id global_v5_20261111
.\.venv\Scripts\python.exe -m earthfire.global_pipeline collect --config config/global_v5b.json --run-id global_v5b_20261111
.\.venv\Scripts\python.exe -m earthfire.global_pipeline prepare --config config/global_v5b.json --run-id global_v5b_20261111
.\.venv\Scripts\python.exe -m earthfire.global_pipeline train --config config/global_v5b.json --run-id global_v5b_20261111
```

O orçamento original continua contado no identificador compartilhado. Após seu encerramento, uma nova execução exige outro identificador e uma decisão explícita de orçamento. `collect` verifica e reaproveita o cache; falhas persistentes permanecem no manifesto. Os treinamentos completos compatíveis e a avaliação final pronta são reutilizados. Uma alteração em dados, configuração ou código exige nova execução.

```python
from earthfire.global_pipeline import predict
resultado = predict(39.5, -8.125, "2024-01-01", "runs/global_v5b_20261111")
# Pode retornar insufficient_data se a região não passou na validação.
```

Não há necessidade de token para ORNL ou POWER. O cliente FIRMS e as credenciais Windows existentes permanecem disponíveis para etapas posteriores. Os limites centrais de CPU, RAM, disco, tempo, requisições, retentativas e treinamentos estão nos JSONs de configuração. A coleta começou em série e passou a duas requisições simultâneas após observar respostas válidas; HTTP 429 interrompe a coleta. Resultados e caches de execuções interrompidas são preservados.

## Arquivos de saída

| Arquivo | Contrato |
|---|---|
| `coverage/manifest.json` | Consultas, hashes, falhas, origem de caches |
| `processed/manifest.json` | Hash dos exemplos, fontes, grupos e lacunas |
| `processed/mask_examples.parquet` | Uma linha por recorte e composto, UTC convencional, features retrospectivas, alvo anulável e split |
| `processed/areas.geojson` | Geometria real do recorte e célula climática associada |
| `processed/sensor_counts.parquet` | Contagens positivas por sensor, período e célula climática; sem preenchimento de zeros |
| `release_policy.json` | Campeão e elegibilidade por região, fixados na validação |
| `model_registry.json` | Versão e hashes de todos os artefatos de inferência |
| `evaluation/metrics.json` | Avaliação final, grupos e incerteza das métricas |
| `evaluation/predictions.parquet` | Probabilidades do teste por candidato |
| `evaluation/calendar.parquet` | Observações, qualidade, referência histórica e anomalias |

## Fontes e pendências

Consultadas em 12/11/2026 UTC: [serviço oficial ORNL](https://modis.ornl.gov/data/modis_webservice.html), [NASA POWER Daily](https://power.larc.nasa.gov/docs/services/api/temporal/daily/), [Natural Earth — domínio público](https://www.naturalearthdata.com/about/terms-of-use/). Cada resposta de dados mantém data de obtenção e hash. A resolução nominal não implica precisão uniforme.

Pendentes: máscaras e geolocalização VIIRS 375 m, harmonização identificável para a nova abrangência, cobertura global sistemática de terras, previsões meteorológicas históricas realmente emitidas, latências verificadas, avaliação prospectiva, generalização para continentes inteiramente reservados, atualização regular, horizontes diários de até sete dias com dependência temporal e subdivisões estaduais. Os resultados medidos ficam no relatório da execução; esta descrição não afirma que todas as regiões passarão nos critérios.
