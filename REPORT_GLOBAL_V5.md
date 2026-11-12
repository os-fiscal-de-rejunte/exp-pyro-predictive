# Relatório da expansão multicontinental v5

Concluído em 2026-11-12T00:52:00+00:00. O núcleo da expansão foi implementado e executado num piloto de 12 recortes, em seis continentes. Não foi concluído treinamento ou validação para todas as terras do planeta.

## O que foi executado

- Inventário incremental: 31 originais, quatro grupos de cópias idênticas; hashes dos 31 originais conferidos novamente ao final, sem alterações.
- Seleção em blocos: 122.536 registros MODIS/VIIRS de 13 ZIPs anuais, 2019–2025, com país, origem, campos e sensores preservados. VIIRS 2023 não está nos ZIPs disponíveis e continua sendo lacuna desse histórico multicontinental.
- Coleta MOD14A2: 360 partições planejadas, 3.301 recortes por composto, 2.443 com máscara integralmente válida, 858 com classificação insuficiente, 11 datas esperadas sem imagem. A rodada final fez 311 consultas e reaproveitou caches; não terminou com consultas em falha persistente.
- NASA POWER: 12 consultas, cinco variáveis, 12 áreas, 2019-12-01 a 2025-12-31; unidades verificadas, nenhum erro de clima. O piloto usa Daily para esse pequeno número de células; o cliente Zarr existente continua disponível para recortes maiores, sem alegar que esta execução processou o acervo mundial.
- Treino: 892 exemplos e 220 positivos, 2020–2022, oito áreas. Validação para escolha global: 303 exemplos nas áreas de treino, 2023; áreas reservadas usadas separadamente nos critérios de liberação. Teste cronológico final: **807 exemplos, 138 positivos**, 2024–2025. Quatro áreas inteiras ficaram fora do ajuste.
- Dois treinamentos Pyro/PyTorch, com e sem meteorologia, 1.500 passos cada, CPU; aproximadamente 98,5 s e 43,0 s de SVI, além de importação/preparação. Sem GPU e sem ajuste posterior pelo teste.
- Foram executados 63 testes automatizados e a verificação real em `scripts/verificar_global_v5.py`. As falhas encontradas em desenvolvimento (serialização de inteiros NumPy, ordem concorrente não determinística e diretório de exportação) foram corrigidas. A contagem antecipada de 862 exemplos em uma mensagem de progresso foi corrigida para os 807 efetivamente avaliados.

## Resultados medidos

| Candidato | Brier validação | Brier teste | Log loss teste | Acurácia teste (0,5) |
|---|---:|---:|---:|---:|
| global_frequency | 0.17122 | 0.14752 | 0.47427 | 82.90% |
| seasonal | 0.09970 | 0.11695 | 0.37809 | 86.25% |
| persistence | 0.13416 | 0.12814 | 0.41777 | 81.91% |
| pyro_no_weather | 0.17578 | 0.10381 | 0.34091 | 85.63% |
| pyro_weather | 0.17390 | 0.10258 | 0.33717 | 85.75% |
| logistic_no_weather | 0.17897 | 0.10393 | 0.33949 | 85.87% |
| logistic_weather | 0.17480 | 0.10250 | 0.33698 | 85.87% |

A seleção foi congelada na validação: **seasonal**, Brier 0,09971 contra 0,17390 do Pyro com clima. No teste, o Pyro com clima obteve Brier menor, mas trocar o campeão após observar esse resultado contaminaria a seleção. A consulta mantém a referência sazonal. A melhora pontual de Brier do Pyro no teste não comprova superioridade operacional.

O campeão teve **86,25% de acurácia** no teste completo, contra 82,90% de sempre prever negativo. Acurácia não é suficiente para avaliar ocorrência rara. No limiar 0,5: TP=65, FP=38, FN=73, TN=631; recall=47.10%, precisão positiva=63.11%. O Brier do campeão foi 0,11695; intervalo por bootstrap de blocos regionais: 0,07860–0,15637. São resultados destes recortes e desse evento condicional, não percentuais de acerto mundial.

| Trimestre UTC | n | Positivos | Brier sazonal | Brier Pyro com clima |
|---|---:|---:|---:|---:|
| 1 | 196 | 43 | 0.12188 | 0.10497 |
| 2 | 204 | 27 | 0.10571 | 0.08288 |
| 3 | 215 | 31 | 0.10290 | 0.08668 |
| 4 | 192 | 37 | 0.13958 | 0.13889 |

As métricas por região e continente e a curva de calibração estão nos JSONs e nas figuras. Foram preservadas as previsões individuais de todos os candidatos.

## Cobertura realmente sustentada

| Região | Continente | Grupo | Validação n / positivos | Liberação |
|---|---|---|---:|---|
| brasil_cerrado | South America | train | 46 / 2 | insufficient_data |
| argentina_chaco | South America | holdout | 46 / 0 | insufficient_data |
| california | North America | train | 20 / 8 | insufficient_data |
| canada_boreal | North America | holdout | 7 / 0 | insufficient_data |
| portugal | Europe | train | 35 / 0 | insufficient_data |
| grecia | Europe | holdout | 18 / 5 | insufficient_data |
| angola | Africa | train | 41 / 12 | elegível |
| sudao | Africa | train | 46 / 20 | elegível |
| india | Asia | train | 34 / 6 | elegível |
| tailandia | Asia | holdout | 43 / 14 | elegível |
| australia_norte | Oceania | train | 36 / 15 | elegível |
| australia_leste | Oceania | train | 45 / 3 | insufficient_data |

Cinco recortes passaram nos critérios de validação: Angola, Sudão, Índia, Tailândia e norte da Austrália. Somam aproximadamente **7,216.8 km²** de superfície dos recortes, cerca de **0.001415% da superfície do globo**. Essa fração não é cobertura operacional: o evento é condicional, as consultas são retrospectivas e datas com clima fora do treino continuam bloqueadas. Os 12 recortes estudados somam 17,320.4 km²; não representam países inteiros.

No teste, 376 de 807 linhas passaram na política espacial e nos limites climáticos. Nesse subconjunto, o campeão teve acurácia de 79,26% e Brier de 0,15771; 110 eventos positivos. A amostra é diferente e mais concentrada em atividade do que a amostra completa. O desempenho fraco ou insuficiente nas outras regiões permanece explícito.

## Demonstração reproduzível

Os artefatos compactos, com exemplos preparados e transformações, estão em `artifacts/global_v5` (1.69 MiB). A demonstração usa o primeiro exemplo de teste elegível, sem seleção pelo acerto:

```powershell
.\.venv\Scripts\python.exe scripts/testar_global_real.py
```

Local: 10.5, 30.0; emissão histórica 2024-01-01T00:00:00Z; horizonte 8 dias. Probabilidade condicional: 82.10%; intervalo sobre essa probabilidade: 59.04%–96.64%. O evento é ao menos um pixel MOD14A2 com foco, supondo a futura máscara integralmente válida. Não é probabilidade de ignição ou de todo fogo existente. Intervalos preditivos do resultado binário são retornados separadamente.

Para verificar os artefatos e a preservação dos originais:

```powershell
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe scripts/verificar_global_v5.py
.\.venv\Scripts\python.exe -m pytest -q
```

Os comandos exatos de preparação, treino e retomada estão em [Expansão global](docs/expansao_global.md). As versões executadas estão em `requirements-global-v5.txt` e `environment.json`. Os dados brutos e caches permanecem locais; não é possível reproduzir a coleta bruta exclusivamente com o pacote compacto se os serviços revisarem os dados.

## Limitações e trabalho pendente

1. A referência VIIRS diária v4 foi preservada. A expansão v5 usa MODIS de até oito dias, com escala e alvo próprios; não é uma melhoria diretamente comparável da previsão VIIRS diária.
2. Não há máscara VIIRS 375 m global integrada, harmonização identificável global ou comprovação da probabilidade incondicional de fogo. A contagem por sensor é diagnóstico, sem soma entre plataformas ou fator universal.
3. Os cortes e localidades são intencionais; poucos eventos e nuvens impedem liberação em várias áreas. Não houve teste deixando um continente inteiro fora do ajuste.
4. O histórico VIIRS usa atraso hipotético de dois dias; a data real de publicação dos arquivos científicos não está demonstrada. Meteorologia termina três dias antes da emissão, com mesma ressalva. A execução é retrospectiva; não demonstrou operação em tempo real.
5. O endpoint POWER Daily não fornece previsão futura. Arquivo de previsões meteorológicas emitidas, atualização regular, horizontes diários conjuntos de até sete dias, generalização mundial e estados administrativos permanecem pendentes.
6. Os intervalos são aproximações condicionais; a dependência residual e a seleção por observabilidade podem reduzir a calibração. Não se atribui causa ambiental aos resultados.

Próxima ação de maior impacto: completar o histórico VIIRS 2023 e obter máscaras/geolocalização compatíveis com VIIRS 375 m em um conjunto novo de regiões e datas, mantendo uma nova avaliação prospectiva intocada. O teste 2024–2025 agora é histórico de avaliação e não deverá ser reutilizado para afirmar ganho independente após mudanças guiadas por seus resultados.

## Verificação do pacote versionado

Uma cópia limpa do commit `6b51046`, reutilizando o ambiente Python local, executou `scripts/testar_global_real.py` sem rede e retornou `ok`, com a mesma probabilidade de 0,8209886418173092. Isso verifica o checkout, a integridade dos artefatos, os finais de linha e a independência dos caches locais; não é um teste de instalação do ambiente do zero. Evidência em `artifacts/global_v5/clone_verification.json`.
