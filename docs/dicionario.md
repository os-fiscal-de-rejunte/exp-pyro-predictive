# Dicionário e contratos

Os tipos físicos originais são texto CSV; as interpretações abaixo dependem do produto e de validação. `source_archive`, `source_member`, `source_row` e `source_sha256` apontam para a origem sem alterar os ZIPs.

| Campo | Significado e unidade | Tratamento |
|---|---|---|
| `latitude`, `longitude` | Centro do pixel de detecção, graus | Numérico validado em [-90,90] e [-180,180]; não é precisão de localização do fogo. |
| `acq_date`, `acq_time` | Data e hora de aquisição em UTC | Preservadas como texto; `acquired_utc` é convertido com zeros à esquerda e validação de hora/minuto. |
| `satellite`, `instrument`, `product` | Plataforma, instrumento e identificação derivada do nome do ZIP | Preservados separadamente; não somados entre sensores. |
| `confidence` | Confiança de detecção do produto | MODIS numérico 0–100; VIIRS `l`, `n`, `h`. Escalas não são equivalentes e não são confiança do modelo. |
| `brightness`, `bright_t31` | Temperaturas MODIS, K | Preservadas com nome original. |
| `bright_ti4`, `bright_ti5` | Temperaturas VIIRS I4 e I5, K | Preservadas com nome original. |
| `scan`, `track` | Dimensões da observação, km conforme descrições FIRMS | Preservadas; variam com geometria. |
| `frp` | Potência radiativa do pixel, MW | Preservada; os produtos têm limitações e resolução diferentes. |
| `daynight` | D ou N | Preservado e auditado. |
| `type` | Tipo inferido de anomalia | Preservado; não tratado como incêndio confirmado. A NASA relata inconsistências do campo em alguns produtos VIIRS Collection 2; versão/origem exata precisa de confirmação. |
| `version` | Indicador textual de versão e possível sufixo de processamento | Os valores são relatados literalmente; sua relação com a coleção exata dos arquivos não foi confirmada. |

`records.parquet` contém linhas selecionadas e origem. `weekly_positive.parquet` contém apenas combinações com pelo menos um registro. `calendar.parquet` inclui semanas sem registros com `detections = null` e `observation_state = missing_unknown`; `archive_present_for_all_calendar_years` apenas indica existência de ZIP nominal, não cobertura. `historical_reference` e `observed_anomaly` usam semanas positivas anteriores. `examples.parquet` documenta cortes e features; `climatology.parquet` é ajustado no treino.

POWER Daily em UTC: `T2M`, `T2M_MAX`, `T2M_MIN` em °C; `RH2M` em %; `PRECTOTCORR` em mm/dia; `WS10M` em m/s. A resposta bruta e seus metadados ficam no cache. As features semanais somam precipitação em mm/semana e calculam médias das demais variáveis diárias, inclusive médias das máximas e mínimas diárias.

## Contrato atual da simulação Zarr v2

`earthfire.dual_scenario.predict_week(latitude, longitude, reference_date, run_dir, scenario_overrides=None, forecast_path=None, variant='selected', n_samples=600)` usa o modelo já treinado da execução. A data deve ser domingo UTC; o horizonte é fixo de sete dias. A localização precisa pertencer ao recorte configurado. As regras do arquivo opcional de previsão e das alterações meteorológicas permanecem as descritas abaixo; `GWETROOT` e `GWETTOP` são adimensionais e permanecem na climatologia de treino do v2 mesmo quando se fornecem as seis variáveis meteorológicas do cenário.

A saída `schema_version=2.0` fornece `location`, limites nominais da célula, `emission_utc`, `target_week_start`, `horizon_days`, definição do evento, versões e hashes do modelo e das features, fonte e trajetória meteorológica, estados de qualidade, cenários e intervalos preditivos. `sensor_version_support` pode indicar `unseen_version`, caso em que os intervalos não incluem incerteza da mudança. `emission_mode=retrospective_simulation` identifica uma data de emissão anterior ao ajuste efetivo do modelo. `event_probability=null` e `status=insufficient_data` registram que a ocorrência não é identificável; `conditional_count_status=ok` só se aplica à contagem condicionada. O GeoJSON, calendário e previsões retrospectivas usam igualmente a versão 2.0 em `exports_v2/`.

## Simulação da execução anterior v1

Entrada `spatial_config` tem `run_dir` e `power_cache`. `reference_date` é domingo UTC; `horizon_days` deve ser 7. O local deve pertencer a uma célula do treino. `scenario_overrides` aceita `temperature_delta_c`, `relative_humidity_delta_percentage_points`, `precipitation_multiplier` e `wind_multiplier`. Valores impossíveis geram erro de validação; combinações fisicamente possíveis ainda podem ser meteorologicamente implausíveis.

`forecast_path`, se fornecido, é um JSON com `issued_at` ISO 8601 com fuso, `source`, `location` com `latitude` e `longitude` na mesma célula, `units` exatamente iguais a `earthfire.power.EXPECTED_UNITS` e `daily`, lista com exatamente sete objetos contendo `date` (`AAAA-MM-DD`) e as seis variáveis POWER. A data de emissão tem de ser anterior ou igual à simulação. Esse contrato valida estrutura, datas e unidades; não verifica independentemente a qualidade da fonte.

Saída `schema_version=1.0`: `requested_location`, `analysis_area`, `emission_utc`, `horizon_days`, `event_definition`, `weather_history`, `weather_trajectory`, `reference_scenario`, `modified_scenario`, `model_version`, `preprocessing_version`, `status` e `conditional_count_status`. Em cada cenário, `expected_count_conditional` e intervalos são contagens condicionadas, em detecções/semana. `event_probability` e diferença em pontos percentuais são `null` se não identificáveis. `insufficient_data` não equivale a zero; `weather_unavailable` identifica falha meteorológica; `out_of_distribution` bloqueia a estimativa extrapolada. `ok` no campo condicional não torna disponível a probabilidade incondicional.
