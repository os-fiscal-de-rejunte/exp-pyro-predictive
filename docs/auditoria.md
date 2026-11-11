# Auditoria dos dados

> Esta é a auditoria da execução inicial v1, anterior aos oito ZIPs MODIS de 2018–2025. Para a atualização e a sobreposição atualmente disponível, consulte [auditoria Zarr v2](auditoria_zarr_v2.md).

Estado: **concluída**; 22 ZIPs inventariados. Cada CSV foi lido integralmente apenas nos ZIPs processados; os relatórios JSON preservam detalhes por país, coluna, categoria e SHA-256.

| Arquivo | Tamanho (bytes) | Registros | Data mínima | Data máxima | Membros CSV | Duplicatas exatas |
|---|---:|---:|---|---|---:|---:|
| modis_2000_all_countries.zip | 4,658,057 | 261,363 | 2000-11-01 | 2000-12-31 | 163 | 0 |
| modis_2001_all_countries.zip | 31,545,201 | 1,769,987 | 2001-01-01 | 2001-12-31 | 197 | 0 |
| modis_2002_all_countries.zip | 65,275,966 | 3,699,859 | 2002-01-01 | 2002-12-31 | 199 | 0 |
| modis_2003_all_countries.zip | 90,378,077 | 5,149,401 | 2003-01-01 | 2003-12-31 | 204 | 0 |
| modis_2004_all_countries.zip | 89,357,956 | 5,092,534 | 2004-01-01 | 2004-12-31 | 202 | 0 |
| modis_2005_all_countries.zip | 89,941,181 | 5,125,152 | 2005-01-01 | 2005-12-31 | 208 | 0 |
| modis_2006_all_countries.zip | 82,443,496 | 4,688,046 | 2006-01-01 | 2006-12-31 | 202 | 0 |
| modis_2007_all_countries.zip | 90,015,477 | 5,134,642 | 2007-01-01 | 2007-12-31 | 208 | 0 |
| modis_2008_all_countries.zip | 84,590,452 | 4,818,997 | 2008-01-01 | 2008-12-31 | 205 | 0 |
| modis_2009_all_countries.zip | 78,288,906 | 4,456,446 | 2009-01-01 | 2009-12-31 | 205 | 0 |
| modis_2010_all_countries.zip | 82,694,466 | 4,715,142 | 2010-01-01 | 2010-12-31 | 203 | 0 |
| viirs-jpss1_2018_all_countries.zip | 259,896,539 | 14,524,817 | 2018-04-01 | 2018-12-31 | 221 | 0 |
| viirs-jpss1_2019_all_countries.zip | 370,806,690 | 20,742,304 | 2019-01-01 | 2019-12-31 | 220 | 0 |
| viirs-jpss1_2020_all_countries.zip | 365,551,116 | 20,385,806 | 2020-01-01 | 2020-12-31 | 223 | 1 |
| viirs-jpss1_2021_all_countries.zip | 358,433,386 | 20,059,802 | 2021-01-01 | 2021-12-31 | 219 | 0 |
| viirs-jpss1_2022_all_countries.zip | 324,844,239 | 18,043,578 | 2022-01-01 | 2022-12-31 | 220 | 0 |
| viirs-jpss1_2024_all_countries.zip | 401,156,384 | 22,511,092 | 2024-01-01 | 2024-12-31 | 219 | 1 |
| viirs-jpss1_2025_all_countries.zip | 328,649,496 | 18,376,493 | 2025-01-01 | 2025-12-31 | 217 | 1 |
| modis_2001_all_countries (1).zip | 31,545,201 | 1,769,987 | 2001-01-01 | 2001-12-31 | 197 | cópia integral de modis_2001_all_countries.zip |
| modis_2005_all_countries (1).zip | 89,941,181 | 5,125,152 | 2005-01-01 | 2005-12-31 | 208 | cópia integral de modis_2005_all_countries.zip |
| modis_2010_all_countries (1).zip | 82,694,466 | 4,715,142 | 2010-01-01 | 2010-12-31 | 203 | cópia integral de modis_2010_all_countries.zip |
| viirs-jpss1_2018_all_countries (1).zip | 259,896,539 | 14,524,817 | 2018-04-01 | 2018-12-31 | 221 | cópia integral de viirs-jpss1_2018_all_countries.zip |

**Registros em ZIPs únicos processados:** 179,555,461. Duplicatas exatas verificadas dentro e entre CSVs do mesmo ZIP único: 3. Registros com ano de aquisição diferente do ano nominal do ZIP: 0.
As quatro cópias integrais entram na tabela, mas não nos totais únicos. Duplicatas exatas são igualdade de todos os campos CSV, não associação de eventos.

**Extremos geográficos observados**, antes de excluir coordenadas inválidas: latitude -89.14716 a 81.9249; longitude -179.9902 a 179.99451 graus.

## Campos e qualidade

Os CSVs são texto. Os campos comuns são latitude, longitude, scan, track, acq_date, acq_time, satellite, instrument, confidence, version, frp, daynight e type. MODIS usa brightness e bright_t31; VIIRS usa bright_ti4 e bright_ti5. Os tipos de análise e unidades constam no [dicionário](dicionario.md).

Valores ausentes (texto vazio), totais por coluna nos ZIPs únicos processados:

```json
{"acq_date": 0, "acq_time": 0, "bright_t31": 0, "bright_ti4": 0, "bright_ti5": 0, "brightness": 0, "confidence": 0, "daynight": 0, "frp": 0, "instrument": 0, "latitude": 0, "longitude": 0, "satellite": 0, "scan": 0, "track": 0, "type": 0, "version": 0}
```

Valores inválidos por regra, contagens independentes que podem se sobrepor:

```json
{"bright_t31_invalid": 0, "bright_ti4_invalid": 0, "bright_ti5_invalid": 0, "brightness_invalid": 0, "confidence_invalid": 0, "date_invalid": 0, "daynight_invalid": 0, "frp_invalid": 120, "latitude_invalid": 0, "longitude_invalid": 0, "scan_invalid": 0, "time_invalid": 0, "track_invalid": 0}
```

- `modis` — satélites: `{'Terra': 17081176, 'Aqua': 27830393}`; valores literais de `version`: `{'6.2': 44901153, '6.03': 10416}`.
- `viirs-jpss1` — satélites: `{'N20': 134643892}`; valores literais de `version`: `{'2': 134643892}`.

Os valores de `version` são registrados literalmente: sem metadados do download não se deve inferir com segurança coleção exata, fonte de processamento ou mistura NRT/histórica. O relatório por arquivo contém categorias de confiança, dia/noite e tipo e os valores ausentes por coluna.

## Cobertura, sobreposição e desconhecidos

Os nomes dos ZIPs disponíveis têm MODIS em 2000–2010 e VIIRS/NOAA-20 em 2018–2022 e 2024–2025; não há sobreposição temporal local entre os produtos. O arquivo de 2023 e os anos 2011–2017 estão ausentes. A presença de arquivos anuais e de detecções em um dia não comprova completude, cobertura orbital ou ausência de nuvens. Não é possível reconstruir dias verdadeiramente sem detecção e não se inventam zeros. Se houver registros fora do ano nominal, duplicatas entre anos não foram verificadas globalmente; fora isso, anos distintos não podem conter linhas totalmente idênticas com a mesma data. Pares espaço-temporais de eventos não foram associados.

O JSON `global_countries.json` tem 27.008.745 bytes; a inspeção encontrou 246 códigos de país, percentis, cenários SSP e chaves temporais de 2000-07 a 2100-07. A variável física, unidade e proveniência não são identificadas no conteúdo. Não é utilizado como meteorologia observada.

A demonstração leu 14.313.853 linhas dos arquivos Brazil e reteve 61.041 registros em três células; não removeu linhas selecionadas por horário inválido ou duplicação exata. Ver [metodologia](metodologia.md), [fontes](fontes.md) e os manifestos da execução.
