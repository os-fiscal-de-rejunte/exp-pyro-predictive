# Auditoria incremental dos novos dados — Zarr v2

Auditoria executada em 11/11/2026. O inventário detalhado, incluindo limites, tipos, ausências, versões, hashes e valores inválidos, está em `runs/audit_zarr_v2_20261111/audit_added/`. Os resultados da auditoria anterior, ainda aplicáveis aos arquivos presentes naquela execução, estão em `runs/full_20261111/audit_full/`. A seleção e o treinamento não modificam `DATA/`; os ZIPs são percorridos diretamente, sem extração.

## O que foi observado

| ZIP MODIS novo | Registros globais nos CSVs | Datas observadas | Código de versão observado |
|---|---:|---|---|
| `modis_2018_all_countries.zip` | 4.184.262 | 2018-01-01 a 2018-12-31 | 6.03 |
| `modis_2019_all_countries.zip` | 4.578.188 | 2019-01-01 a 2019-12-31 | 6.03 |
| `modis_2020_all_countries.zip` | 4.440.623 | 2020-01-01 a 2020-12-31 | 6.03 |
| `modis_2021_all_countries.zip` | 4.384.343 | 2021-01-01 a 2021-12-31 | 6.03 |
| `modis_2022_all_countries.zip` | 3.789.164 | 2022-01-01 a 2022-12-31 | 6.03 |
| `modis_2023_all_countries.zip` | 4.664.302 | 2023-01-01 a 2023-12-31 | 61.03 |
| `modis_2024_all_countries.zip` | 4.848.247 | 2024-01-01 a 2024-12-31 | 61.03 |
| `modis_2025_all_countries.zip` | 3.693.555 | 2025-01-01 a 2025-12-31 | 61.03 |

Os oito ZIPs acrescentam 34.582.684 linhas. Nenhuma duplicata exata foi encontrada dentro dos arquivos novos; cinco valores de FRP foram sinalizados como inválidos nas verificações. A verificação das regras implementadas não encontrou datas, horas, coordenadas, confiança ou dia/noite inválidos nem campos em branco nesses arquivos. Veja os relatórios individuais para colunas, dimensões e abrangência; a existência de observações nas datas extremas não comprova completude.

Na auditoria anterior, os arquivos únicos continham 179.555.461 linhas antes de remover três duplicatas identificadas. O total dos arquivos únicos antigos e novos é 214.138.145 linhas, antes dessa remoção. Quatro ZIPs com conteúdo duplicado já conhecidos foram ignorados na preparação. SHA-256 dos 30 ZIPs conferiu entre as auditorias e o manifesto de preparação; isso dá evidência de preservação entre esses pontos, não prova sobre períodos anteriores à auditoria.

Na partição denominada `Brazil`, a preparação percorreu 17.151.487 linhas dos arquivos disponíveis e reteve 155.580 registros válidos nas 16 células configuradas: 22.297 MODIS/Terra, 40.651 MODIS/Aqua e 92.632 VIIRS/NOAA-20. O treinamento e a comparação principal usam Terra e NOAA-20 separadamente; Aqua é preservado nas agregações, sem ser somado aos dois. Nenhum registro selecionado foi removido por invalidez ou duplicidade exata. As células são retângulos da grade meteorológica; o conteúdo vem dos arquivos denominados Brazil e não de um recorte por polígono de fronteira nacional.

O campo `type` foi preservado: no recorte há 57.823 MODIS tipo 0 e 5.125 tipo 2; no VIIRS há 74.092 tipo 0, 18.519 tipo 2 e 21 tipo 3. São códigos do produto, não incêndios confirmados; algumas anomalias podem ser fontes térmicas estáticas. A confiança numérica MODIS e as categorias VIIRS permanecem distintas. O arquivo conserva as observações por plataforma, produto, data/hora UTC e origem.

## Sobreposição e lacunas

Há registros MODIS e VIIRS/NOAA-20 em 2018–2022 e 2024–2025; não há arquivo VIIRS/NOAA-20 de 2023 nem de 2026 na cópia examinada. O teste usa 2024–2025. Nas 403 amostras de teste da previsão, o MODIS da semana anterior tem código `61.03`, enquanto no treino usado tinha `6.03`; o código VIIRS observado é `2`. A interpretação exata e o efeito desses códigos não foram estabelecidos. Ano, versão, processamento e outras mudanças podem estar confundidos.

Observações positivas não identificam exposição do sensor, nuvens ou completude das extrações. Uma célula/semana sem linhas conserva estado desconhecido. Há mistura potencial de tipos de processamento; os arquivos não fornecem proveniência suficiente para atribuir com segurança processamento histórico ou quase em tempo real a cada registro.

## Meteorologia NASA POWER

O acervo Zarr foi consultado para 17 células, incluindo uma célula de controle, entre 2018-01-01 e 2025-12-31. Foram salvos 49.674 registros diários para oito variáveis e 7.089 semanas completas; não houve valores ausentes nas variáveis desse recorte. Foram feitas 143 requisições para metadados e blocos, com 169.313.956 bytes em blocos e metadados armazenados no cache. O conjunto preservado tem resolução nominal de 0,5° de latitude por 0,625° de longitude.

Na célula de controle, os seis parâmetros compartilhados foram comparados à resposta preservada da API NASA POWER: 2.922 dias pareados por variável, diferença absoluta máxima de aproximadamente 0,0000293 nas unidades de cada variável. Isso confirma a concordância da amostra consultada, não uma equivalência universal entre versões ou datas futuras. A origem, os atributos e o hash dos metadados constam em `runs/zarr_brasil_v2_20261111/weather_zarr/provenance.json`; a comparação completa está em `api_comparison.json`.
