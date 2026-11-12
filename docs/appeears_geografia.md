# Geografia NASA AppEEARS no PYRO

## O que foi implementado

O cliente AppEEARS consulta o catálogo real, autentica pela conta NASA Earthdata, envia tarefas de pontos, acompanha seu processamento e baixa os CSVs com limite de volume e verificação SHA-256. As solicitações têm identificador derivado dos parâmetros e são retomadas sem reenviar trabalhos confirmados. Submissões POST sem confirmação não são repetidas automaticamente, para evitar duplicações. Token e senha não são gravados nos manifestos, logs ou Git. No Windows, o token pode ficar no Gerenciador de Credenciais; a senha usada para gerá-lo não é salva.

O serviço é assíncrono: uma consulta pode retornar `pending`, sem dados inventados. Os orçamentos estão em `config/geography.json` e `config/geography_budget.json`: 200 requisições por cache, até 200 MB de downloads, timeout de 45 segundos, até três tentativas de GET, tarefas em lotes de até 100 pontos, dois treinamentos, CPU com duas threads e uma janela de 90 minutos para o piloto. O cliente foi implementado para uso serial por cache. Não é um servidor mundial pronto para requisições concorrentes.

## Produtos consultados

O catálogo oficial foi consultado em 12/11/2026 e confirmou a disponibilidade das camadas configuradas.

| Produto | Informação | Escala nominal |
|---|---|---|
| MCD12Q1.061 | Classes IGBP: florestas, savanas, agricultura, urbano, água e outras | 500 m, anual |
| MOD13Q1.061 | NDVI e EVI, com qualidade do índice | 250 m, composto de 16 dias |
| MOD44B.061 | Percentuais de cobertura arbórea, vegetação não arbórea e superfície não vegetada | 250 m, anual |
| NASADEM_NC.001 | Altitude | Aproximadamente 30 m, aquisição SRTM de 2000; faixa 56°S a 60°N |

Altitude média e amplitude entre amostras são descritores de relevo. Amplitude não é declividade. NDVI e EVI não medem diretamente combustível seco ou umidade do solo. Cobertura arbórea não identifica, isoladamente, uma floresta nativa ou o tipo de combustível. A classificação tem incerteza própria.

## Autenticação

```powershell
.\.venv\Scripts\python.exe scripts/configurar_appeears.py
```

O comando pede usuário e senha Earthdata no terminal; a senha fica oculta. Somente o token AppEEARS é guardado no cofre Windows, sem exibição. Ele precisa ser renovado quando o serviço indicar expiração. Como alternativa, configure `APPEEARS_TOKEN`, ou `EARTHDATA_USERNAME` e `EARTHDATA_PASSWORD` em variáveis de ambiente. Não coloque credenciais no código, em argumentos de terminal ou em commits. O token Earthdata de outros serviços não é automaticamente equivalente ao token AppEEARS.

## Visualização local

```powershell
.\.venv\Scripts\python.exe scripts/geografia_local.py --latitude -21.229167 --longitude -54.341825 --reference-date 2026-11-12
```

Abra `runs/geography_view/index.html`. O relatório é autossuficiente e oferece cartões, gráficos por coordenada, tabela dos valores, datas e produtos, além de JSON e GeoJSON. As nove posições ficam em uma grade 3×3 dentro da célula MODIS de aproximadamente 1.443 km². São amostras, não um mosaico contínuo de imagens nem todos os pixels do recorte. Não há interpolação que crie detalhes inexistentes. Em `pending` ou `authentication_required`, o relatório mostra a indisponibilidade explicitamente.

Reexecutar o mesmo comando retoma os mesmos IDs. `--no-fetch` usa somente os arquivos já existentes. Os caches brutos permanecem em `runs/geography_cache`; não alteram `DATA`.

Na consulta preditiva existente:

```powershell
.\.venv\Scripts\python.exe scripts/testar_expansao_real.py --latitude -21.229167 --longitude -54.341825 --geography --geography-view runs/minha_consulta
```

O JSON acrescenta `geographic_context`. A simples presença desse contexto não altera a probabilidade v6. A resposta expõe `model_uses_geography: false` quando as camadas estão apenas sendo exibidas.

## Qualidade, unidades e tempo

Os CSVs do serviço de pontos AppEEARS já têm as escalas aplicadas. Não se multiplica NDVI/EVI novamente por 0,0001. Preenchimentos e valores fora dos limites são ausentes. No MOD13Q1, somente valores com os dois bits MODLAND de qualidade iguais a zero entram nas features; não se exige que todo o campo de bits seja zero. MCD12Q1 usa o filtro conservador de QC 0 ou 2. MOD44B exige Quality e Cloud iguais a zero. Dados recusados continuam nos registros com seu estado de qualidade, sem virar zero observado. NASADEM usa preenchimento e limites do catálogo; não se afirma ter toda a qualidade do relevo reconstruída.

O alinhamento retrospectivo usa o fim de cada composto e atrasos assumidos: 30 dias após o composto de vegetação e 365 dias após o intervalo anual. O DEM só entra a partir de 2021, uma hipótese conservadora de disponibilidade. Esses atrasos **não comprovam a data original de publicação** nem o acesso histórico à versão atualmente reprocessada. O JSON informa `availability_verified: false`. No modo operacional estrito, a data de obtenção do arquivo também deve anteceder a emissão. Índices de vegetação com mais de 90 dias são considerados insuficientes. A média das amostras não é apresentada como média exata de toda a área.

## Treinamento e avaliação

```powershell
.\.venv\Scripts\python.exe scripts/treinar_geografia.py
```

O piloto usa 24 células, com dois recortes por continente e grupo espacial, escolhidos por identificador e suporte de treino, sem escolher pelas métricas do teste. Usa nove amostras por célula: 216 coordenadas, em seis tarefas. Datas de extração: 2018–2025; DEM em tarefa própria com sua data nominal de 2000. Os rótulos continuam sendo os compostos MOD14A2 de 2020–2025 já preparados, sem converter ausência em negativo.

São treinadas duas regressões Bernoulli em Pyro/PyTorch: coordenadas+sazonalidade e as mesmas entradas acrescidas das features geográficas. Indicadores de dados ausentes e imputadores são ajustados no treino. Não há histórico local obrigatório na inferência. As categorias de solo são frações amostrais de classes fixas, sem codificação ordinal artificial. Este experimento geográfico não usa clima atual; os modelos climáticos anteriores permanecem preservados.

A ablação usa exatamente a mesma coorte com geografia suficiente. Treino 2020–2022, validação 2023 e teste 2024–2025; regiões `holdout` não entram no ajuste nem na seleção global. A seleção pelo Brier usa somente validação e é gravada antes da avaliação final. Porém esse teste histórico já foi conhecido durante o desenvolvimento v6: a comparação é retrospectiva, não uma nova confirmação independente. O experimento não autoriza anunciar acurácia mundial.

Se a coleta estiver pendente, retome sem espera longa:

```powershell
.\.venv\Scripts\python.exe scripts/treinar_geografia.py --no-wait
```

O mesmo orçamento é preservado. Ao expirar, uma nova janela exige explicitamente outro `--budget-id`; os artefatos científicos concluídos ficam congelados e não recebem dados novos.

Depois de treinar:

```powershell
.\.venv\Scripts\python.exe scripts/testar_expansao_real.py --latitude -21.229167 --longitude -54.341825 --geography --geography-model runs/geography_pilot_v1/models --geography-view runs/minha_consulta
```

O resultado mantém a previsão v6 e apresenta `geography_candidate` para comparação. O candidato tem probabilidade e intervalo próprios, método selecionado, flags de extrapolação e estado exploratório. Não se troca silenciosamente uma saída validada por um experimento. Se o modelo sem geografia vencer, isso será informado, sem inventar ganho de acurácia. Faltando geografia ou checkpoint, o candidato não é calculado.

## Fontes

Consultadas em 12/11/2026: [API oficial AppEEARS](https://appeears.earthdatacloud.nasa.gov/api/), [catálogo público de produtos](https://appeears.earthdatacloud.nasa.gov/api/product), [MOD13Q1](https://www.earthdata.nasa.gov/data/catalog/lpcloud-mod13q1-061), [MCD12Q1](https://www.earthdata.nasa.gov/data/catalog/lpcloud-mcd12q1-061), [NASADEM](https://www.earthdata.nasa.gov/data/catalog/lpcloud-nasadem-hgt-001), [orientação NASA sobre escala no CSV](https://forum.earthdata.nasa.gov/viewtopic.php?t=5682) e [descrição oficial da máscara MOD13Q1 no Earth Engine](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD13Q1).
