# Encerramento AppEEARS — 12/11/2026

Integração, exemplo visual real, dois treinamentos e avaliação concluídos. Veja [REPORT_APPEEARS.md](REPORT_APPEEARS.md) para os dados, métricas, limitações e comandos. Teste offline: `.venv/Scripts/python.exe scripts/testar_geografia_real.py`. Suíte: 97 testes aprovados. O modelo novo é experimental e não substitui v6, pois piorou a qualidade probabilística e não detectou positivos no teste espacial no limiar 0,5. Coleta remota maior pendente; retomada explícita documentada, sem processo local deixado em execução. Abaixo permanece o histórico anterior.

# Encerramento v6

Veja REPORT_MAXCOVERAGE_V6.md para dados, limites, resultados medidos e comandos. A coleta anterior foi interrompida; a avaliacao foi concluida na retomada solicitada pelo usuario.

# Relatório da execução

## Continuação concluída — expansão multicontinental v5

Relatório completo em [REPORT_GLOBAL_V5.md](REPORT_GLOBAL_V5.md). Foram executados coleta NASA, seleção dos ZIPs locais, preparação, dois treinamentos Pyro, avaliação sem seleção pelo teste e consulta retrospectiva com incerteza. O piloto estudou 12 áreas de seis continentes; cinco recortes passaram nos critérios de validação. A referência selecionada foi a sazonal (86,25% de acurácia no teste completo, recall 47,10%); o resultado não demonstra cobertura operacional mundial. O pacote `artifacts/global_v5` permite teste offline com `scripts/testar_global_real.py`.

Foram aprovados 63 testes locais e a verificação com dados reais, incluindo hashes dos 31 originais. Os bloqueios e tarefas não concluídas estão no relatório e no model card. Os processos de coleta/treinamento foram encerrados. Os comandos de retomada estão em `docs/expansao_global.md`; a janela compartilhada usa `EARTHFIRE_BUDGET_ID=global_v5_20261111` e não é reiniciada silenciosamente.


> **Atualização de 11/11/2026, 16h26:** o acesso Earthdata permitiu obter 1.287 grânulos VIIRS e treinar um modelo diário com clima. No teste de 131 exemplos, seu Brier foi 0,09088, contra 0,08628 da referência sazonal. Foram obtidos outros 22 grânulos de outubro de 2026 para uma área; 11 dias tinham classificação incompleta. Passaram 52 testes; os 31 originais e os grânulos da amostra foram conferidos por hash, e a retomada não iniciou treinamento adicional. Detalhes, comandos, teste real e limitações estão em [REPORT_VIIRS_V4.md](REPORT_VIIRS_V4.md). O histórico anterior segue abaixo.

> **Atualização de 11/11/2026, 12h10:** foram adicionados dados públicos VIIRS de 2023 e retratos MODIS/VIIRS recentes, coletadas máscaras MODIS em oito áreas e treinadas duas variantes probabilísticas. O teste geral favoreceu a variante sem clima, mas a referência sazonal venceu nas regiões de treino. O acesso a séries GEOS-FP falhou; o reforecast obtido não prova disponibilidade histórica. Dois grânulos VIIRS foram localizados, mas o download requer token Earthdata. A suíte passou com 40 testes, os 31 arquivos originais foram revalidados por hash e nenhum processo permaneceu em execução. A nova execução, comandos, métricas, limitações e pendências estão em [REPORT_NASA_V3.md](REPORT_NASA_V3.md). O relatório abaixo preserva o histórico anterior.

> **Atualização de 11/11/2026, 08h40:** os oito arquivos MODIS de 2018–2025 e o NASA POWER Zarr foram processados em um novo experimento de 16 células; os resultados, testes, limites e comandos atuais estão em [REPORT_ZARR_V2.md](REPORT_ZARR_V2.md). A auditoria original e o relatório abaixo são preservados como histórico da execução v1; as frases abaixo sobre falta de sobreposição referem-se à cópia de dados anterior à adição dos novos arquivos.

Atualizado em 11 de novembro de 2026, aproximadamente 00h58, horário de São Paulo.

## Entregue

- Auditoria em fluxo dos CSVs dentro dos ZIPs, inventário por arquivo e país, validação, SHA-256 e identificação de duplicatas. Os originais foram lidos sem extração ou alteração. Os 22 hashes dos ZIPs registrados na preparação correspondem aos da auditoria posterior.
- Preparação do recorte, preservação de proveniência, Parquet de registros e séries semanais, calendário com ausências desconhecidas e anomalias retrospectivas.
- Referência descritiva e modelo Pyro para diferenças entre sensores pareados, com verificação de elegibilidade. Nesta base retornam `insufficient_data`, pois não existem anos comuns MODIS/VIIRS.
- Modelos probabilísticos de contagem condicional, com e sem meteorologia, baselines, avaliação cronológica, checkpoints, métricas e gráficos. A escolha da variante ocorreu pela validação antes do teste.
- Cliente NASA POWER com UTC, cache bruto, orçamento de rede, serialização de requisições, timeout e retentativas limitadas; simulador independente, fonte hipotética ou previsão externa validada, verificações de entrada, extrapolação, JSON e figura.
- Documentação, configuração, CLI, testes, recursos, estado e contratos JSON/GeoJSON. Foram produzidas execuções curtas de integração; as duas primeiras foram marcadas como superadas após a introdução de hashes de código. A execução curta atual é `runs/quick_20261111_v3` e não constitui evidência científica.

## Dados e operações realizadas

A pasta tinha 22 ZIPs e um JSON climático, sem repositório Git ou código prévio. Foram lidas integralmente 179.555.461 linhas dos 18 ZIPs únicos: 44.911.569 MODIS e 134.643.892 VIIRS/NOAA-20. Quatro arquivos eram cópias SHA-256 idênticas; havia outras três linhas exatamente duplicadas dentro ou entre CSVs de ZIPs únicos. As regras de auditoria apontaram 120 valores de FRP inválidos; os demais campos testados não apresentaram ausências textuais ou invalidez segundo as regras implementadas. Isso não comprova completude, cobertura ou qualidade física de todos os registros.

O arquivo `global_countries.json` contém cenários SSP e percentis para 246 códigos de país entre 2000-07 e 2100-07, sem variável física, unidade ou fonte identificáveis no conteúdo. Não foi usado como meteorologia. Três consultas NASA POWER forneceram históricos para as três células e ficaram em `runs/power_cache/` com respostas, unidades e metadados; nenhuma credencial é necessária e nenhuma linha local foi enviada ao serviço.

O recorte leu 14.313.853 linhas dos arquivos Brazil e selecionou 61.041 registros em três células. Não foram removidas duplicatas exatas ou datas inválidas entre os selecionados. Há 16.079 registros MODIS e 44.962 VIIRS; o modelo usa apenas VIIRS/NOAA-20, pois os sensores locais não se sobrepõem no tempo. O treino contém 235 semanas elegíveis, a validação 45 e o teste 75; o calendário inclui `null` para ausências não explicadas.

Os dois treinos completos tiveram 900 passos cada, em CPU; os tempos medidos da otimização foram cerca de 9,0 s e 7,3 s, respectivamente. Foram usados PyTorch 2.10.0 e `pyro-ppl` 1.9.2. A máquina tinha 12 processadores lógicos, aproximadamente 5,15 GB de RAM inicialmente disponível e 65,1 GB livres; CUDA não estava disponível. A última verificação encontrou cerca de 4,92 GB de RAM disponíveis e 64,89 GB livres. Os arquivos em `runs/` ocupavam aproximadamente 23 MB e o ambiente virtual 30 MB; a diferença no espaço livre inclui outras alterações do sistema e não é uma medida exata do tamanho dos artefatos. O orçamento compartilhado e o contador de três requisições estão registrados em `runs/`.

## Testes e resultados

`python -m pytest -q`: **15 testes aprovados**. Cobrem parsing UTC, coordenadas, confiança específica por produto, duplicação e preservação de originais sintéticos, agregação, ausência desconhecida, anomalias retrospectivas, separação temporal, features, cliente POWER, cenários, limites, checkpoints e exportação. Também foi verificado sobre os artefatos reais que os 8.148 registros de calendário incluem 7.156 ausências `null`, que os 75 exemplos retrospectivos têm intervalos coerentes, que as três geometrias são fechadas e que os três hashes das respostas POWER correspondem ao cache. A retomada dos modelos completos foi executada sem retreinar.

Na validação, o CRPS foi 43,49 com meteorologia e 45,95 sem meteorologia, selecionando a primeira variante. No teste condicional, a mediana do modelo selecionado teve MAE 112,17 e RMSE 247,15 detecções/semana. Persistência teve MAE 120,93; sazonal 191,24; a regressão log-linear regularizada teve o menor MAE, 105,32. O modelo sem meteorologia teve MAE 108,69. O modelo selecionado teve CRPS 87,11, cobertura do intervalo de 80% de 74,7% e cobertura do intervalo de 95% de 89,3%. No subconjunto de oito semanas acima do p90 do treino, seu MAE foi 578,63 e a cobertura de 80% foi 62,5%. Esses resultados não validam previsões de semanas sem registros.

O exemplo real retrospectivo da célula (-18,5; -57,5), emitido na simulação em 7/1/2024, deu média condicional de 17,12 detecções/semana (intervalo 80%: 2–43) na climatologia e 22,49 (2–57) no cenário modificado. Os resultados são associativos, não causais. A chance de pelo menos uma detecção permanece `null` e o estado principal é `insufficient_data`.

Houve uma falha transitória de substituição de arquivo no OneDrive durante o primeiro teste curto; foi corrigida com até três tentativas limitadas, e a retomada funcionou. Não há teste final falho conhecido. A página inicial do desafio retornou erro e a página alternativa encontrada retornou página não encontrada; as fontes oficiais FIRMS e POWER acessíveis foram registradas em `docs/fontes.md`.

## Limitações e trabalho ainda não concluído

- Não há dados MODIS e VIIRS coincidentes nesta cópia: a harmonização e seus intervalos não foram estimados em dados reais. O código para pares positivos é exploratório e não identifica atividade verdadeira ou eficiência de detecção sem informação adicional.
- Cobertura, nuvens, oportunidades de observação e completude não foram fornecidas. Não foram criados zeros nem uma chance de ocorrência artificial; Brier e calibração de ocorrência não podem ser avaliados.
- A avaliação é retrospectiva, com meteorologia histórica que pode ter sido revisada e sem histórico de versões disponíveis no instante de emissão dos focos. Não é validação operacional completa. A incerteza meteorológica futura não foi propagada porque o cenário usa uma trajetória fixa.
- Somente três células selecionadas pelo treino foram modeladas. Não houve teste espacial independente nem treino global. Os registros selecionados de cada execução são reunidos em memória; a ampliação deve usar lotes de células e países, não uma seleção global única.
- O JSON climático local não tem metadados suficientes para alinhamento; POWER foi usado de forma consistente no treino e na consulta. A comparação entre fontes meteorológicas não pôde ser feita.
- Um calendário baseado em semanas positivas não representa a distribuição de todas as semanas. A contagem inclui os tipos de anomalias disponíveis, sem confirmar que cada uma seja um incêndio de vegetação.

## Como executar e retomar

No PowerShell, na raiz do projeto, para reutilizar a execução existente com uma nova janela de orçamento:

```powershell
$env:PYTHONPATH = 'src'
$env:EARTHFIRE_BUDGET_ID = 'retomada_01'
.\.venv\Scripts\python.exe -m earthfire.cli --run-id full_20261111 audit --mode full
.\.venv\Scripts\python.exe -m earthfire.cli --run-id brasil_20261111 prepare
.\.venv\Scripts\python.exe -m earthfire.cli --run-id brasil_20261111 train --steps 900
.\.venv\Scripts\python.exe -m earthfire.cli --run-id brasil_20261111 evaluate --samples 400
.\.venv\Scripts\python.exe -m earthfire.cli --run-id brasil_20261111 export
.\.venv\Scripts\python.exe -m pytest -q
```

Os checkpoints existentes são reutilizados somente se dados, configuração e código forem compatíveis. Para construir um novo recorte, siga a instalação e as etapas `prepare`, `features`, `train`, `evaluate`, `export` do [README](README.md), com outro `run_id` e orçamento. Para reproduzir o cenário salvo:

```powershell
.\.venv\Scripts\python.exe -m earthfire.cli --run-id brasil_20261111 scenario `
  --latitude -18.5 --longitude -57.5 --reference-date 2024-01-07 `
  --horizon-days 7 --overrides-file config/scenario_example.json `
  --output runs/brasil_20261111/exports/scenario_pantanal_20240107.json
```

O próximo passo de maior impacto é obter, para as mesmas regiões e períodos, MODIS e VIIRS sobrepostos e metadados verificáveis de cobertura/qualidade ou máscaras de observação. Com eles será possível investigar efeitos de sensor e definir exemplos negativos sustentáveis; em seguida, obter versões históricas e previsões meteorológicas emitidas à época para uma avaliação operacional.

Encerramento verificado às 00h59 de 11/11/2026: não havia processo registrado desta execução ainda ativo; os checkpoints, logs e resultados foram preservados.
