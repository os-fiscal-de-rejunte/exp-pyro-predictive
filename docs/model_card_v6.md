# Model card v6

# Expansao maxima v6 - relatorio final

Atualizacao UTC: 2026-11-12T10:37:31+00:00.

## Resultado executado

Foram preparados 72 blocos, 898 celulas e 245,957 celula-compostos de 2020 a 2025. 23,229 alvos ficaram desconhecidos; nao foram preenchidos com zero. Existem 1140 celulas com algum historico observado. A selecao leu 12.614.130 registros Terra MODIS de 2020-2022, somente tipo 0. Os 31 originais foram relidos e seus hashes conferem.

A validacao liberou **498 celulas**, **718,795.7 km2**, **0.140922% da superficie global**. Isso equivale a **99.6 vezes** a area liberada do piloto v5, sem somar areas sobrepostas. Para a emissao atual, a restricao de mes permite 465 celulas, 671,164.7 km2. Area validada em algum periodo nao e igual a area consultavel em toda estacao.

| Continente | Celulas liberadas | Area km2 |
|---|---:|---:|
| Africa | 138 | 199184.4 |
| Asia | 83 | 119799.3 |
| Europe | 25 | 36084.1 |
| North America | 52 | 75055.0 |
| Oceania | 96 | 138563.0 |
| South America | 104 | 150110.0 |

O alvo e ao menos um pixel MOD14A2 com foco em composto de ate oito dias, condicionado a todos os pixels 41x41 terem classificacao valida. Nao e probabilidade incondicional de incendio, ignicao ou area queimada. Uma classificacao valida no composto nao comprova cobertura diaria completa.

## Treinamento e teste

Treino 2020-2022, validacao 2023, teste 2024-2025. Dois modelos Pyro/PyTorch de 1.500 passos foram treinados com os mesmos 12.000 exemplos reais, 2.992 positivos, com e sem meteorologia NASA POWER. Normalizacao, imputacao, priors e parametros usam somente treino. A politica foi congelada em 2026-11-12T10:34:10+00:00, antes do teste iniciado em 2026-11-12T10:34:10+00:00. O campeao global escolhido na validacao foi `seasonal`; o teste nao foi usado para promove-lo.

O teste total tem 74,408 casos. Na coorte liberada: 42,968 casos, 10,176 positivos, acuracia **78.86%**, recall **39.55%**, precisao positiva **57.86%**, Brier **0.15080**. O recall baixo impede apresentar o modelo como detector completo de queimadas. Os 86,25% do v5 nao foram preservados nesta coorte mais ampla e diferente; uma comparacao direta nao isola o efeito da expansao.

Comparacao justa na mesma coorte com clima completo (Brier menor e melhor):

| Modelo | Casos | Brier | Acuracia | Recall |
|---|---:|---:|---:|---:|
| global_frequency | 68603 | 0.17607 | 77.26% | 0.00% |
| seasonal | 68603 | 0.14391 | 80.41% | 34.24% |
| regional_seasonal | 68603 | 0.16919 | 76.99% | 5.04% |
| persistence | 68603 | 0.15334 | 77.86% | 48.42% |
| pyro_no_weather | 68603 | 0.15389 | 78.73% | 31.44% |
| pyro_weather | 68603 | 0.15200 | 79.23% | 31.68% |
| logistic_no_weather | 68603 | 0.15384 | 78.73% | 31.73% |
| logistic_weather | 68603 | 0.15186 | 79.27% | 33.83% |

O clima melhorou o Pyro em relacao a variante sem clima, mas o Pyro nao superou a referencia sazonal nesta coorte. As referencias foram mantidas na inferencia quando sustentadas pela validacao. A amostra computacional dos modelos globais e menor que o historico usado pelas referencias.

Comparacao nas mesmas celulas e periodos liberados:

| Modelo | Brier | Acuracia |
|---|---:|---:|
| global_frequency | 0.18084 | 76.32% |
| seasonal | 0.15301 | 78.98% |
| regional_seasonal | 0.17255 | 76.07% |
| persistence | 0.16143 | 76.06% |
| routed_probability | 0.15080 | 78.86% |

Generalizacao espacial sem filtragem local: 13,034 casos novos reservados. Brier sazonal 0.17203; persistencia 0.15267. A liberacao local usa validacao 2023 da propria celula e nao deve ser chamada de transferencia sem rotulos. Todos os resultados, inclusive falhas regionais, calibracao, bootstrap por bloco e meses, estao em `evaluation/metrics.json` e `diagnostics.json`.

## Entregas e verificacoes

- Pipeline particionado, grade nativa sem sobreposicao, validacao de mascaras, clima UTC, baselines regionais, Pyro, consultas historicas e projecoes sazonais.
- GeoJSON, catalogos JSON/Parquet, calendario com anomalias estritamente retrospectivas e manifestos de integridade. Calendario: 245,957 registros; 182,689 anomalias sustentadas pelas datas de producao.
- 74 testes automatizados aprovados; verificacao real aprovada; consultas historica e futura com estado `ok`; coordenada sem suporte retorna probabilidade nula.
- Documentacao: `docs/expansao_maxima_v6.md`, `docs/contratos_v6.md`; pacote: `artifacts/maxcoverage_v6`.

## Recursos e encerramento

A coleta anterior foi interrompida sem traceback conclusivo e nao permaneceu ativa. O ultimo checkpoint de preparo tinha 72 blocos; ha 74 particoes brutas completas. A fila original tem 531 candidatos e nao foi integralmente executada. No pedido posterior de continuar e finalizar, foi aberto um orcamento separado somente para encerramento, `budget_maxcoverage_v6_closing_20261112.json`, sem novas consultas de coleta. Nao se apresenta essa retomada como ocorrida dentro das oito horas originais.

Consultas registradas: 2284 de mascaras ORNL, 496 POWER Daily e 4 POWER Zarr. O teto combinado POWER foi respeitado. Snapshot de armazenamento rastreado: 1.919 GB. Pico amostrado de RAM: 457.7 MB; nao e medicao continua do pico instantaneo. CPU, sem uso de servico pago.

## Limites e proximo passo

A expansao v6 usa MODIS MOD14A2; nao demonstra harmonizacao MODIS/VIIRS global. VIIRS foi excluido dos regressores novos devido a lacuna de 2023; as versoes anteriores permanecem preservadas. A coleta nao foi global completa, e o treinamento usa seis anos, nao todo o historico positivo disponivel. Nao ha validacao prospectiva nem arquivo de previsoes meteorologicas emitidas para sustentar operacao completa em 2026. A projecao futura e sazonal/estatica, com hipotese explicita de estabilidade; nao usa meteorologia futura observada nem ajustes arbitrarios.

O proximo passo de maior impacto e ampliar e validar a deteccao de eventos positivos, sobretudo nos recortes com recall baixo, com cobertura VIIRS compativel e previsoes meteorologicas historicamente emitidas. Nova expansao precisa de outro run_id e teste futuro reservado; este teste fica congelado.

## Comandos

```powershell
.\.venv\Scripts\python.exe scripts/testar_expansao_real.py
.\.venv\Scripts\python.exe scripts/testar_expansao_real.py --historical
.\.venv\Scripts\python.exe scripts/verificar_expansao_v6.py --run-dir artifacts/maxcoverage_v6 --skip-originals
.\.venv\Scripts\python.exe -m pytest -q
```

O pacote permite consulta offline, sem retreinamento. O teste e os artefatos finalizados nao devem receber novas particoes. Os caches brutos e logs locais foram preservados em `runs/maxcoverage_v6_20261111` e `runs/maxcoverage_v6b_20261111`.
