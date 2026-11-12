# Contratos de consulta e exportação v6

## Unidade e evento

Cada célula canônica contém 41×41 pixels do MOD14A2, aproximadamente 1.443,365 km² na projeção equivalente MODIS. A área efetiva vem em `cell.geometry` e `cell.area_km2`; a latitude consultada não representa um hotspot previsto naquele ponto exato.

`probability` está entre zero e um e estima o evento: pelo menos um pixel marcado como foco no composto de até oito dias, condicionado à classificação válida de todos os pixels. Essa condição não garante observação em todos os dias. O resultado não estima ignição, propagação, área queimada ou todo fogo presente. Uma detecção não equivale a um incêndio independente.

## Funções públicas

```python
from earthfire.expansion_predict import forecast, predict

# Próximo composto, usando referência sazonal/estática validada.
result = forecast(latitude, longitude, "artifacts/maxcoverage_v6")

# Reconstrução retrospectiva em um início de composto já preparado.
past = predict(latitude, longitude, "2024-01-01", "artifacts/maxcoverage_v6")
```

As funções carregam artefatos treinados e verificam seus hashes. Não treinam novamente nem consultam a rede. `forecast` aceita `reference_date` UTC opcional. O calendário do composto reinicia em 1º de janeiro; `valid_until` é exclusivo. `lead_days` separa a espera até o próximo composto da duração `horizon_days`, que pode ser menor que oito no fim do ano. Não é uma previsão dos sete dias imediatamente seguintes à consulta.

`predict` usa somente features retrospectivas preparadas. A consulta não exige conhecer o rótulo do período futuro. Meteorologia indisponível ou fora dos limites de treino exige uma referência alternativa validada; na ausência dela o resultado fica indisponível. A latência histórica do POWER não foi integralmente verificada, portanto o replay não comprova operação em tempo real.

`forecast` usa a distribuição histórica sazonal/estática; não incorpora meteorologia atual ou uma previsão meteorológica futura. Os meses permitidos aparecem em `static_projection_months`. As probabilidades dependem da hipótese de estabilidade histórica, sem validação prospectiva. Esse modo permanece separado do simulador meteorológico das versões anteriores; ele não representa uma simulação climática global validada.

## Campos principais

| Campo | Interpretação |
|---|---|
| `schema_version` | `native41-1.0` |
| `status` | `ok`, `insufficient_data`, `weather_unavailable` ou `out_of_distribution` |
| `requested_location` | Coordenadas solicitadas em graus WGS84 |
| `cell` | Identificador nativo, polígono GeoJSON e área efetivamente analisada |
| `event` | Definição observacional da probabilidade |
| `probability` | Probabilidade do evento; `null` quando indisponível |
| `probability_interval_95` | Intervalo sobre o parâmetro probabilístico; não sobre contagem de incêndios |
| `predictive_event_interval_95` | Intervalo para o resultado binário futuro, frequentemente `[0, 1]` |
| `issued_at` | Data/hora UTC da emissão reconstruída ou projetada |
| `valid_from`, `valid_until` | Intervalo temporal de validade, fim exclusivo |
| `horizon_days`, `lead_days` | Duração do composto e antecedência até seu início |
| `selected_model` | Método efetivamente usado, incluindo referências simples |
| `weather_context` | Disponibilidade e interpretação da meteorologia |
| `future_observability` | `unknown`; não se conhece a máscara futura |
| `history` | Número de compostos observados e positivos no treino; sua frequência não é uma previsão |
| `model_run_id` | Execução que gerou os artefatos |
| `model_registry_sha256` | Hash do registro exato de modelos e arquivos |
| `preprocessing_sha256` | Hash do código de features registrado no treinamento |
| `data_assembled_at` | Quando os artefatos finais ficaram disponíveis |
| `historical_projection` | Emissão solicitada anterior à montagem dos artefatos: reconstrução, não operação comprovada |

Campos específicos de um modo podem não aparecer em outro. Falhas de integridade geram erro explícito, em vez de inferência com arquivos incompatíveis. Coordenadas inválidas geram `ValueError`. Células reconhecidas sem validação retornam `insufficient_data`, não zero. Nenhuma probabilidade é somada para representar um país ou vários períodos.

## Catálogo, calendário e avaliação

- `coverage.geojson`: polígonos na área real, indicadores de liberação, suporte de projeção estática e disponibilidade de histórico. Não se preenche um país inteiro porque há algumas células avaliadas nele.
- `evaluation/cells.json` e `.parquet`: catálogo resumido para consulta espacial e integração posterior. O nome `cells` não significa que todas ofereçam previsões.
- `evaluation/calendar_parts/*.parquet`: datas UTC, classe observacional, evento binário ou ausência, pixels de fogo/válidos/nublados, referência e anomalia quando sustentadas. `fire_pixels` é contagem de pixels marcados no composto; não é número de incêndios ou área queimada.
- `evaluation/calendar_manifest.json`: unidades, regras e hashes. Anomalias usam somente compostos anteriores com produção anterior à emissão. Menos de três referências válidas resulta em `insufficient_data` e valores ausentes.
- `evaluation/validation_predictions.parquet`: previsões usadas na seleção; seu hash fica na política congelada antes do teste.
- `evaluation/test_parts/*.parquet`: resultados do teste 2024–2025. `release_eligible` indica suporte da política e dos inputs, não verdade ambiental.
- `evaluation/metrics.json`: desempenho global, por continente/grupo, regiões novas e transferência espacial sem filtragem local. `diagnostics.json` acrescenta calibração, comparação na mesma coorte, bootstrap por bloco e calendário.

Dados derivados são Parquet, JSON ou GeoJSON. Valores ausentes numéricos do JSON público usam `null`; não são convertidos em probabilidade zero. Os hashes do registro identificam os arquivos de treino/inferência; `package_manifest.json` cobre os demais arquivos distribuídos. Os caches brutos permanecem nos diretórios de execução locais.
