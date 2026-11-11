# Estado da execução

## Atualização final de 11/11/2026, 16h26 — acesso VIIRS

- Etapa atual: concluída; não há coleta ou treinamento em execução. Credencial usada apenas em memória; a busca local em 258 arquivos textuais não encontrou seu conteúdo.
- Concluído: 1.287 grânulos VJ114A1.002 em oito áreas, incluindo 128 de validação complementar e 143 de borda. As oito datas não encontradas são 01/01/2018. A seleção e a ampliação da validação estão em `docs/decisions.md`.
- Preparação: 1.144 registros área-dia, dos quais 570 com classificação insuficiente; um mosaico conferido recuperou as 287 posições que atravessam a borda. Produto de 1 km mantido separado das detecções de 375 m.
- Modelo com clima: 230 exemplos de treino; Brier no teste de 131 exemplos foi 0,09088 versus 0,08628 da referência sazonal. A liberação foi bloqueada nas regiões de treino pelo critério de validação; o grupo reservado passou no critério mínimo, com amostra pequena.
- Dados recentes: 22 grânulos diários em 01–22/10/2026 para uma área, sem arquivos ausentes nesse intervalo; 11 dias tinham classificação incompleta. Não foram usados para treinar o modelo.
- Testes e integridade: 52 testes aprovados; 1.287 grânulos e 31 originais de `DATA/` tiveram os hashes rechecados. A retomada carregou o checkpoint sem um quarto treinamento. Os três treinamentos previstos foram usados.
- Recursos: orçamento inicial de 12 CPUs lógicas, até seis threads, 1,08 GB de RAM dos processos registrados e 12,11 GB de novos arquivos; a execução v4 ocupa aproximadamente 450 MB. Na última leitura havia cerca de 3,24 GB de RAM disponível e 59,72 GB livres.
- Próxima ação: ampliar as amostras e áreas, obter cobertura específica do produto VIIRS 375 m e um histórico de previsões meteorológicas realmente emitidas, e validar prospectivamente. Relatório, comandos e checkpoints estão em [REPORT_VIIRS_V4.md](REPORT_VIIRS_V4.md).

## Atualização final de 11/11/2026, 12h10

- Etapa atual: encerrada; resultados, falhas de rede e checkpoints preservados.
- Concluído: VIIRS/NOAA-20 Brasil 2023; retratos MODIS e VIIRS NRT da América do Sul; auditoria; 3.000 composições MOD14A2 em oito áreas; amostra POWER e GEOS-S2S-3; modelos com e sem clima, baselines, avaliação e consultas reais.
- Teste 2024–2025: 697 exemplos; Brier 0,08289 para a variante selecionada sem clima e 0,09237 para a referência sazonal no agregado. Nas seis áreas de treino, a referência sazonal foi melhor. A versão com clima foi pior na validação e não foi selecionada.
- Integridade: os 31 arquivos originais de `DATA/` foram relidos e seus hashes conferem com as auditorias; três arquivos FIRMS baixados também conferem. Resultado em `runs/nasa_complement_v3b_20261111/verification_v3.json`.
- Testes: 40 aprovados. Nenhum processo da execução ou treinamento permanece ativo; os checkpoints foram retomados sem novo treinamento.
- Bloqueios: o catálogo NASA identificou dois grânulos VJ114A1.002 para o exemplo, mas seu download protegido requer `EARTHDATA_TOKEN`, não disponível; os arquivos FIRMS recentes cobrem sete dias e a chave FIRMS Area API não foi fornecida. As tentativas de séries GEOS-FP tiveram HTTP 502 e timeout. A amostra GEOS-S2S-3 é reforecast produzido em 2026, sem prova de disponibilidade em 2021.
- Limites científicos: máscara MODIS não equivale à cobertura VIIRS. A probabilidade v3 é condicional à futura máscara válida, não probabilidade incondicional de incêndio. O recorte de oito áreas não valida resultados para todo o Brasil. Ausência de registro continua desconhecida.
- Recursos no início: 12 CPUs lógicas; orçamento compartilhado de até 6 threads, 1,08 GB de RAM dos processos registrados e 12,11 GB de novos arquivos; aproximadamente 60,55 GB livres; sem GPU CUDA. Dois treinamentos completos de aproximadamente 26 e 20 segundos, em CPU. Orçamento e processos registrados: `runs/budget_nasa_v3_20261111.json` e `runs/processes_nasa_v3_20261111/`.
- Próxima ação: com um token Earthdata configurado no ambiente, executar a coleta VJ114A1 descrita no relatório, auditar e integrar sua máscara ao modelo; ampliar dados recentes e arquivo de previsões meteorológicas realmente emitidas.
- Checkpoints: `runs/nasa_complement_v3b_20261111/coverage/`, `processed/`, `models/` e `evaluation/`; reforecast e manifesto em `runs/s2s_nasa_v3_20261111/s2s_hindcast/`. Resultados e comandos estão no [relatório v3](REPORT_NASA_V3.md) e em [OVERNIGHT_REPORT.md](OVERNIGHT_REPORT.md).

## Histórico anterior

- Etapa atual: aquisição NASA complementar, máscaras de cobertura e avaliação de possibilidade de previsão em mais locais.
- Atualização da nova etapa: CSV público VIIRS/NOAA-20 para o Brasil em 2023 e CSVs recentes dos dois sensores foram baixados. A coleta inicial de máscara MODIS/Terra `MOD14A2` foi interrompida após constatar poucos eventos de treino no recorte pequeno; seus blocos foram preservados. A execução seguinte amplia o recorte para raio de 20 km, sem alterar as divisões. A máscara não comprova cobertura VIIRS.
- Progresso em 2026-11-11 11:26, horário de São Paulo: auditoria completa dos três CSVs novos concluiu; arquivo VIIRS Brasil 2023 contém 1.312.648 registros e os recortes NRT cobrem datas registradas de 04–11/11/2026. Coleta de máscaras ampliadas segue em `runs/nasa_complement_v3b_20261111/coverage/`, com checkpoints por consulta; ver `pipeline.log` para o avanço atualizado.
- A auditoria e o retrato NRT corrigidos estão em `runs/audit_nasa_v3b_20261111/` e `runs/recent_nasa_v3b_20261111/`. A primeira versão foi preservada e está superada porque tratava incorretamente as categorias longas de confiança VIIRS como inválidas. Uma consulta de Brasília reconheceu a célula e devolveu ausência desconhecida, sem inventar zero.
- Última atualização: 2026-11-11 08:43, horário de São Paulo.
- Entregas: auditoria dos oito ZIPs MODIS adicionados; recorte NASA POWER Zarr para 17 células, incluindo controle; preparação em 16 células; comparação exploratória MODIS/Terra–VIIRS/NOAA-20; dois modelos probabilísticos e baselines; validação temporal/espacial; simulador, JSON, GeoJSON, Parquet e gráficos; documentação da execução v2.
- Verificações: 22 testes passaram; 30 hashes de ZIPs das auditorias conferem com os registrados na preparação; 140 objetos do cache meteorológico foram novamente verificados; 403 previsões e 43.456 linhas de calendário passaram na checagem de contratos; retomada dos checkpoints não iniciou outro treinamento.
- Processo em execução: coleta de máscaras MOD14A2 em `runs/nasa_complement_v3b_20261111/`; progresso e checkpoints no `pipeline.log` e no manifesto de cobertura. Treinamento v3 ainda não iniciado.
- Bloqueios científicos: os CSVs de detecções não demonstram cobertura nem completude; o retrato 2026 obtido cobre apenas os sete dias recentes, e os arquivos anuais de 2026 não foram localizados publicamente. A máscara MODIS não estabelece cobertura VIIRS. A chance incondicional permanece indisponível; versões de produto diferem e não há validação global ou operacional.
- Recursos no início da atualização: 12 CPUs lógicas, cerca de 2,25 GB de RAM disponível, sem CUDA e 65,16 GB livres. Limite de novos arquivos: 13,03 GB; foram usadas 143 requisições NASA e as três vagas de treinamento. O orçamento está em `runs/budget_zarr_v2_20261111.json`.
- Recursos na última verificação: aproximadamente 2,49 GB de RAM disponíveis e 64,73 GB livres; as pastas específicas da atualização ocupam aproximadamente 202 MB, sem contar os artefatos anteriores ou o ambiente virtual.
- Próxima ação: obter cobertura/oportunidade de observação verificável e dados recentes, controlar versões dos produtos e reavaliar desempenho por região e intensidade antes de estimar uma probabilidade de ocorrência.
- Checkpoints: `runs/audit_zarr_v2_20261111/audit_added/`, `runs/sat_zarr_v2_20261111/`, `runs/zarr_brasil_v2_20261111/weather_zarr/` e `runs/zarr_brasil_v2_20261111/models_v2/`. Relatório atual: `REPORT_ZARR_V2.md`; `OVERNIGHT_REPORT.md` preserva a execução anterior.
