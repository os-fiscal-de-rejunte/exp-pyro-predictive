# Expansão máxima v6 — protocolo da execução

## Objetivo e limites

Ampliar a área com estimativas sustentadas pelo histórico, preservando o piloto v5. A execução compartilha uma janela de oito horas, reserva final de 30 minutos, no máximo três ajustes probabilísticos completos, até metade das CPUs lógicas, 60% da RAM inicialmente disponível e o menor valor entre 20 GB e 20% do disco livre inicial. Os valores estão em `config/maxcoverage_v6_runtime.json`; o orçamento compartilhado é `runs/budget_maxcoverage_v6_20261111.json`. A fila não é uma meta garantida de cobertura: a execução termina quando um limite impede continuar ou os candidatos terminam.

## Ganho de coleta sem inflar a área

O serviço NASA ORNL permite recortes de até 100 km acima/abaixo e à esquerda/direita, com até dez datas por consulta. Um recorte real testado retornou 201×201 pixels. Ele foi dividido em 16 células completas de **41×41 pixels**, alinhadas a uma grade sinusoidal canônica. A dimensão de cada célula permanece a do piloto; não se atribui uma previsão de uma célula a todo o bloco.

A grade usa os parâmetros MODIS, origem global `(-20015109.354, 10007554.677)` m e pixel de aproximadamente 926,625 m no produto projetado. Cada célula tem cerca de 1.443,365 km² na projeção equivalente. Os identificadores nativos são únicos e as células não se sobrepõem. Recortes que reaparecem em consultas distintas são deduplicados pelo identificador, não por proximidade de focos. Dados incompletos permanecem ausentes.

Uma consulta pequena independente foi comparada com a célula extraída do bloco: **1.681 de 1.681 pixels idênticos**, com diferenças de coordenadas inferiores a 0,05 m, compatíveis com arredondamento do cabeçalho. Evidência em `runs/maxcoverage_v6_20261111/alignment_verification.json`. Isso verifica a orientação das linhas e o alinhamento, não a precisão de localização dos focos.

## Seleção anterior ao teste

Foram lidos 12.614.130 registros dos ZIPs MODIS de 2020–2022. A seleção considera Terra, coordenadas/datas/confiança válidas e `type=0`, definido pela NASA como fogo de vegetação presumido. Essa classificação é inferida; não comprova a causa. O alvo continua sendo FireMask, sem identificação causal de incêndios.

O planejamento resultou em 531 blocos candidatos, escolhidos com diversidade continental e separação geodésica de pelo menos 350 km entre centros. A seleção é intencional e privilegia atividade conhecida no treino; não representa proporcionalmente todo o planeta. As regiões vão aproximadamente de 55°S a 65°N. Não houve seleção pelas métricas do teste.

Um em cada cinco blocos é reservado espacialmente. Células de treino adjacentes a células reservadas na grade nativa são retiradas do ajuste como buffer. As localidades próximas dos recortes v5 recebem `legacy_context`; seus testes ficam separados dos contextos geográficos novos.

A seleção do campeão global não usa a validação das regiões reservadas. A liberação local, porém, consulta os rótulos de validação 2023 de cada célula: seu teste é uma avaliação com adaptação local, não transferência sem rótulos. Por isso, `spatial_holdout_unfiltered` também avalia todos os casos de teste das regiões novas reservadas, sem filtrar pela política local; essa é a comparação apropriada de generalização geográfica dos modelos globais.

## Observabilidade, clima e exemplos

O alvo é binário: ao menos um pixel com foco no composto MOD14A2 de até oito dias, **condicionado a todos os pixels da célula terem classificação válida**. Uma máscara incompleta não produz um negativo. Os últimos compostos do ano podem ter menos de oito dias. Não se estima ignição, propagação, área queimada ou todo fogo não observado.

Uma classificação válida no composto não comprova que houve oportunidade de observação válida em todos os dias. O composto e suas classes são a referência observacional; não se reconstrói uma cobertura diária inexistente. Condicionar a avaliação à máscara válida também seleciona condições de observabilidade e pode limitar a transferência para períodos nublados. A probabilidade não é a chance incondicional de fogo físico.

Primeiro são coletados os anos de treino. Para avançar ao restante do histórico, uma célula precisa de pelo menos 60 compostos válidos no treino, três de cada classe e fração média de água ≤20%. Os outros recortes podem fornecer somente contagens/frequências históricas observadas; isso não será chamado de previsão.

O acesso Zarr foi testado, mas blocos de 16 anos e 15×15 pontos climáticos transferiam muito mais informação que o necessário para esta amostragem esparsa. A execução passou à API NASA POWER Daily UTC, deduplicada pela célula MERRA-2 de 0,5°×0,625°. As tentativas Zarr entram no teto total de 500 novas requisições POWER. Não há consulta por hotspot. As variáveis e unidades são verificadas: T2M (°C), RH2M (%), PRECTOTCORR (mm/dia), WS10M (m/s) e GWETROOT (adimensional).

As features meteorológicas usam oito dias terminados três dias antes da emissão. Temperatura, umidade, vento e umidade do solo são médias; precipitação é soma em mm por janela. Ausências e valores fisicamente inválidos são preservados como ausentes. O histórico MODIS usa somente imagens anteriores cuja produção precede a emissão, com janela máxima de 40 dias. A latência histórica do POWER não está integralmente comprovada; a avaliação é retrospectiva.

VIIRS permanece fora dos regressores v6 devido à lacuna 2023, que mudava a disponibilidade do histórico entre treino e validação. O modelo VIIRS v4 e os dados originais ficam preservados. A expansão não é apresentada como harmonização MODIS/VIIRS global validada.

## Modelos e qualidade

Treino: 2020–2022. Validação: 2023. Teste: 2024–2025, acessado para avaliação de desempenho somente após persistir a política de seleção. Compostos que atravessam os cortes são excluídos. Normalizadores, imputadores, priors empíricos e parâmetros aprendidos usam somente o treino.

São comparados frequência global, sazonalidade, persistência, regressão logística e regressão Bernoulli Pyro/PyTorch, com e sem clima. A amostra computacional dos modelos globais fica limitada a 12.000 exemplos de treino, por amostragem uniforme determinística. As referências agregadas usam o histórico de treino disponível. Essa diferença de custo/amostragem será declarada. Comparações de contribuição meteorológica usam a mesma amostra de avaliação com clima completo.

Antes de abrir o teste, acrescentou-se uma quarta referência, `regional_seasonal`: frequência regularizada por continente e mês, ajustada exclusivamente nas regiões de treino. Ela evita impor a mesma sazonalidade global a continentes distintos e permite avaliar transferência para células reservadas. A referência sazonal original por célula permanece como comparador e limite de Brier. A opção regional passa pelos mesmos critérios de liberação e não pressupõe que um continente inteiro tenha cobertura.

Quando o clima faltar ou atingir seu orçamento, os rótulos observacionais não são descartados. Uma referência sem clima pode fornecer uma estimativa **marginal histórica**, se ela passar nos mesmos critérios de validação. A saída identifica que não usa meteorologia atual ou futura. Não há imputação de clima fictício nem ajuste arbitrário da probabilidade.

A liberação por célula exige na validação pelo menos 30 casos e cinco de cada classe, erro médio de calibração ≤0,10 e Brier não superior à referência sazonal. O modelo global é selecionado na validação das regiões de treino com clima; a seleção local compara referências elegíveis e o campeão global quando seus inputs são disponíveis. As escolhas ficam congeladas antes do teste. Esses critérios são demonstrativos e não garantem desempenho prospectivo.

Serão reportados Brier, log loss, acurácia, recall, precisão positiva, ROC AUC, average precision e desempenho por continente/grupo. A acurácia não é comparável diretamente aos 86,25% do piloto v5, pois o recorte e a prevalência mudaram. O relatório separará os testes com clima, a cobertura estatística sem clima e os contextos geográficos novos.

## Inferência

`earthfire.expansion_predict.predict` reconstrói uma emissão histórica usando o modelo e as transformações prontos. `forecast` oferece somente uma projeção estática/sazonal que tenha sido validada na célula, para o próximo composto MODIS. Ela usa o calendário conhecido e a distribuição histórica; não trata o POWER Daily como fornecedor de previsão futura. A estabilidade sazonal é uma hipótese explícita, e a observabilidade futura permanece desconhecida.

Intervalos sobre a probabilidade vêm da posterior variacional Pyro, aproximação de Laplace ou posterior Beta da referência. Os intervalos do resultado binário são separados. Dependência espacial/temporal residual e seleção por observabilidade podem tornar os intervalos otimistas. Nenhuma probabilidade é somada entre células ou dias para representar um país inteiro.

A projeção estática também exige suporte no mês consultado: pelo menos três compostos válidos no treino e um na validação naquela célula. Isso impede que a validação de uma estação seca seja usada para liberar automaticamente uma estação sem observações. O catálogo expõe `static_projection_months`. O relatório distingue a área validada em algum momento da área consultável na emissão atual. `data_assembled_at` registra quando o modelo ficou disponível; uma consulta anterior a essa data é uma reconstrução retrospectiva.

## Execução e rastreabilidade

```powershell
$env:PYTHONPATH='src'
$env:EARTHFIRE_BUDGET_ID='maxcoverage_v6_20261111'
.\.venv\Scripts\python.exe -m earthfire.expansion
```

O armazenamento compartilhado de máscaras fica em `runs/maxcoverage_v6_20261111`; features, modelos e avaliação em `runs/maxcoverage_v6b_20261111`. A seleção e seus hashes ficam em `runs/maxcoverage_v6b_20261111_selection/selection.json`. O protocolo de cache é independente do nome do lote e retoma escritas interrompidas somente quando o hash pendente confere. Até três tentativas por consulta; HTTP 429 interrompe a coleta. Tempo, disco e RAM são verificados antes de novas operações.

O controle de disco mede também o tamanho dos diretórios associados ao orçamento, a cada minuto. Uma liberação de espaço externa não reinicia a cota. Os snapshots em `resources.json` registram RAM dos processos conhecidos, pico amostrado e volume rastreado. Bibliotecas numéricas são carregadas após definir limites de threads. Esses monitores não são medições contínuas do pico instantâneo.

O calendário exportado em `evaluation/calendar_parts/` preserva os compostos observados e as ausências. Sua anomalia exige ao menos três compostos anteriores do mesmo mês cuja data de produção anteceda a emissão. Reprocessamentos recentes podem impedir reconstruir essa referência: nesse caso a anomalia fica ausente. Os caminhos dos registros e manifestos v6 usam barras portáveis entre Windows e Linux.

Progresso: `EXPANSION_STATUS.md`, `progress.json` e logs em `runs/maxcoverage_v6b_sparse_pipeline.log`. Modelos finalizados ficam congelados; acrescentar partições requer novo identificador, mantendo o orçamento explícito. Os resultados medidos e as causas reais de encerramento serão registrados em `REPORT_MAXCOVERAGE_V6.md` ao concluir.

Fontes oficiais consultadas: [serviço ORNL](https://modis.ornl.gov/data/modis_webservice.html), [campos MODIS FIRMS](https://firms.modaps.eosdis.nasa.gov/content/descriptions/FIRMS_MODIS_Firehotspots.html), [NASA POWER](https://power.larc.nasa.gov/docs/services/api/temporal/daily/). Respostas e hashes são preservados, sem credenciais nos arquivos versionados.
