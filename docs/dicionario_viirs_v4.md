# Dicionário — dados diários VIIRS v4

O arquivo `runs/viirs_daily_v4b_20261111/processed/mask_examples.parquet` contém uma linha por área e dia efetivamente coletado; dias ausentes do catálogo constam no manifesto de coleta e não são preenchidos automaticamente.

| Campo | Significado e unidade |
|---|---|
| `region_id`, `region_name`, `spatial_group` | Identificação da área e separação espacial `train` ou `holdout` |
| `latitude`, `longitude` | Centro solicitado, graus geográficos; não é localização exata de fogo |
| `period_start`, `period_end_exclusive` | Início e fim exclusivo do composto diário UTC |
| `class_0` a `class_9` | Número de pixels por classe `FireMask` do produto diário |
| `total_pixels`, `valid_pixels`, `fire_pixels`, `cloud_pixels`, `unknown_pixels` | Contagens de pixels na área; desconhecidos incluem não processados e bordas faltantes |
| `masked_pixels`, `invalid_pixels`, `uncovered_pixels` | Pixels indisponíveis na leitura, com classe inválida explícita ou sem tile correspondente; são desconhecidos, não zeros |
| `event` | 1 para ao menos um pixel com foco ou 0 para nenhum, somente se a máscara for integralmente válida; caso contrário é nulo |
| `source_tiles_complete` | Se os tiles informados cobrem toda a geometria; não estabelece ausência de nuvens |
| `source_files`, `source_hashes`, `production_time` | Grânulos NASA, hashes SHA-256 e produção mais tardia dos tiles necessários |
| `past_viirs_records` | Número de registros positivos FIRMS de 375 m encontrados na janela retrospectiva; `null` se não há registros, pois a ausência é desconhecida |
| `past_viirs_mask_event` | Evento da última máscara diária amostrada produzida antes da emissão, quando utilizável; `null` sem suporte |
| `past_modis_event` | Evento do composto MODIS anterior quando disponível antes do alvo; `null` para ausência ou validade desconhecida |
| `weather_T2M`, `weather_RH2M`, `weather_PRECTOTCORR`, `weather_WS10M`, `weather_GWETROOT` | Médias retrospectivas de temperatura em °C, umidade em %, velocidade em m/s e umidade de solo adimensional; precipitação é soma em mm |
| `weather_last_date`, `viirs_history_end` | Datas finais das fontes retrospectivas, sempre anteriores ao dia previsto |
| `split` | `train`, `validation`, `test`, `unknown_coverage` ou `excluded` |

Os arquivos `evaluation/viirs_predictions_*.parquet`, quando produzidos, acrescentam probabilidade condicional, limites de 80% sobre parâmetros e referências. Probabilidade incondicional e chance de ignição não são deriváveis da máscara futura sem um modelo de observabilidade independente.
