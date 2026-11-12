# PYRO — Earth’s Fire Intelligence

Este repositório audita e agrega registros de focos de calor e contém um experimento reproduzível de previsão probabilística **condicional**. PYRO é o nome da plataforma; [Pyro](https://pyro.ai/) é a biblioteca `pyro-ppl` usada com PyTorch. O núcleo funciona sem aplicação web e sem GPU.

**Expansão multicontinental v5:** piloto executado em 12 áreas de seis continentes, com referência MODIS de até oito dias. Cinco recortes passaram nos critérios de validação; a consulta mantém o baseline sazonal selecionado antes do teste. No teste completo de 807 exemplos, acurácia de 86,25% e recall de 47,10% no limiar 0,5. Não é previsão operacional mundial. Resultados e limitações em [REPORT_GLOBAL_V5.md](REPORT_GLOBAL_V5.md), [metodologia](docs/expansao_global.md) e [model card](docs/model_card_global_v5.md).

Teste offline com o pacote de dados e modelos incluído no repositório:

```powershell
.\.venv\Scripts\python.exe scripts/testar_global_real.py
```

Para reproduzir as versões das dependências: `.venv/Scripts/python.exe -m pip install -r requirements-global-v5.txt`, usando Python 3.11.2 e ambiente isolado. Depois, instale o projeto com `pip install -e .`. O exemplo usa uma emissão histórica; outras datas/áreas podem retornar `insufficient_data`.

**Versão VIIRS diária v4:** foram obtidas máscaras diárias VIIRS/NOAA-20 de 1 km e treinado um modelo probabilístico com histórico de satélites e meteorologia NASA POWER. Ele estima detecção **condicionada à máscara futura estar integralmente válida**. No teste, o modelo não superou a referência sazonal. A validação usa apenas oito áreas e não sustenta previsão global; detalhes em [REPORT_VIIRS_V4.md](REPORT_VIIRS_V4.md) e no [model card v4](docs/model_card_viirs_v4.md).

Para testar uma previsão retrospectiva com dados reais:

```powershell
.\.venv\Scripts\python.exe scripts\testar_viirs_real.py
```

O exemplo usa uma área reservada em 22/10/2025. A saída diferencia uma probabilidade condicional de detecção da chance incondicional de incêndio, que permanece indisponível. Também há 22 máscaras VIIRS de outubro de 2026 para uma área, com classificação completa em 11 dias; esses arquivos não fazem parte do treino. Veja a [metodologia](docs/metodologia_viirs_v4.md) e o [dicionário](docs/dicionario_viirs_v4.md).

**Experimento anterior v3:** foram obtidos o VIIRS/NOAA-20 do Brasil em 2023, um retrato recente MODIS/VIIRS e máscaras MODIS de oito áreas brasileiras. Ele estima a probabilidade de detecção MODIS **condicionada a uma máscara futura integralmente válida**. A versão com meteorologia não melhorou a validação; a selecionada não usa clima. Os resultados são distintos dos do experimento VIIRS diário. Detalhes em [REPORT_NASA_V3.md](REPORT_NASA_V3.md) e no [model card v3](docs/model_card_nasa_v3.md).

### Teste com dados reais, sem internet

```powershell
.\.venv\Scripts\python.exe scripts\testar_mascara_real.py
```

O exemplo compara uma previsão simulada em 13/09/2025 com o composto MODIS observado de 14–21/09/2025. É uma estimativa condicional retrospectiva, não uma chance de incêndio presente ou de ignição. Para consultar outras linhas já avaliadas, passe `--latitude`, `--longitude` e `--data`. O JSON completo desse exemplo está em `runs/nasa_complement_v3b_20261111/evaluation/exemplo_real_20250913.json`.

### Consultar uma coordenada e os dados recentes

```powershell
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe -m earthfire.recent_firms --run-id recent_nasa_v3b_20261111 lookup --latitude -15.79389 --longitude -47.88278
```

Coordenadas válidas são reconhecidas mesmo fora do treino. O retrato salvo contém os arquivos FIRMS dos sete dias mais recentes disponíveis em 11/11/2026 para a América do Sul. A ausência de registros resulta em `missing_unknown`, não em zero. Acrescente `--weather` para consultar o histórico NASA POWER; isso requer acesso à rede e não fornece previsão meteorológica futura.

Uma máscara diária VIIRS/NOAA-20 de 1 km foi obtida com acesso Earthdata e incorporada ao experimento v4. Para ampliar a coleta, consulte os comandos e limites no [relatório v4](REPORT_VIIRS_V4.md).

Para atualizar os sete dias, use um novo identificador e execute `refresh` antes de `lookup`:

```powershell
$nova_execucao = 'recent_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
.\.venv\Scripts\python.exe -m earthfire.recent_firms --run-id $nova_execucao refresh
.\.venv\Scripts\python.exe -m earthfire.recent_firms --run-id $nova_execucao lookup --latitude -15.79389 --longitude -47.88278
```

Para inferir com o modelo v3, use `.\.venv\Scripts\python.exe -m earthfire.mask_pipeline predict --latitude -11.5 --longitude -46.875 --date 2025-09-13`. A emissão é simulada às 23:59:59 UTC e ocorre apenas no dia anterior a um composto MODIS; outras datas retornam a próxima data elegível. A consulta a áreas novas pode precisar da NASA; retornos `insufficient_data`, `out_of_distribution` e `weather_unavailable` não devem ser interpretados como chance zero. Mesmo com resultado experimental, a disponibilidade futura da máscara é desconhecida, e a chance incondicional permanece `null`.

## Experimento anterior: NASA POWER Zarr e dois sensores

Os dados e modelos v2 estão em `runs/zarr_brasil_v2_20261111/`. Foram treinados em 16 células do Brasil usando NOAA-20, MODIS e NASA POWER até 2025, estimando contagens semanais condicionadas a semanas com detecções. Consulte a [auditoria](docs/auditoria_zarr_v2.md), [metodologia](docs/metodologia_zarr_v2.md), [avaliação](docs/avaliacao_zarr_v2.md), [model card](docs/model_card_zarr_v2.md) e o [relatório da execução](REPORT_ZARR_V2.md). Seus resultados são distintos dos resultados v3.

### Testar uma previsão com dados reais

Na raiz do projeto, execute:

```powershell
.\.venv\Scripts\python.exe scripts\testar_previsao_real.py
```

O comando simula uma emissão em 02/11/2025 para a célula (-11,5°, -46,875°) e compara a previsão para 03–09/11/2025 com os registros VIIRS observados. A mediana prevista foi 19 detecções, o intervalo preditivo de 80% foi aproximadamente 4–55 e o satélite registrou 3; nesse exemplo, o valor observado ficou fora do intervalo. É uma simulação retrospectiva condicionada a uma semana com detecção, não uma previsão emitida naquela data nem uma probabilidade de ocorrência.

Os dados e as linhas de origem desse exemplo estão em `runs/zarr_brasil_v2_20261111/exports_v2/teste_previsao_com_observado.json`. Use `--latitude`, `--longitude` e `--data` para testar outra célula configurada e outro domingo com histórico disponível; `--saida caminho.json` salva a comparação. Sem registros para a semana, a contagem observada é desconhecida, não zero.

Para retomar as etapas com os checkpoints existentes, no PowerShell:

```powershell
$env:PYTHONPATH = 'src'
$env:EARTHFIRE_BUDGET_ID = 'zarr_v2_nova_janela'
.\.venv\Scripts\python.exe -m earthfire.power_zarr --config config/zarr_brazil_v2.json --run-id zarr_brasil_v2_20261111
.\.venv\Scripts\python.exe -m earthfire.dual_pipeline --config config/zarr_brazil_v2.json --run-id zarr_brasil_v2_20261111 prepare
.\.venv\Scripts\python.exe -m earthfire.dual_pipeline --config config/zarr_brazil_v2.json --run-id zarr_brasil_v2_20261111 harmonize
.\.venv\Scripts\python.exe -m earthfire.dual_pipeline --config config/zarr_brazil_v2.json --run-id zarr_brasil_v2_20261111 train --steps 900
.\.venv\Scripts\python.exe -m earthfire.dual_pipeline --config config/zarr_brazil_v2.json --run-id zarr_brasil_v2_20261111 evaluate --samples 400
.\.venv\Scripts\python.exe -m earthfire.dual_pipeline --config config/zarr_brazil_v2.json --run-id zarr_brasil_v2_20261111 evaluate-harmonization
.\.venv\Scripts\python.exe -m earthfire.dual_pipeline --config config/zarr_brazil_v2.json --run-id zarr_brasil_v2_20261111 export
```

Os checkpoints são reaproveitados quando os hashes de dados, código e configuração conferem. A avaliação pode levar tempo. Para reconstruir as etapas que antecedem essa sequência, consulte os comandos completos no relatório; uma alteração de insumo ou configuração exige novos identificadores de execução.

Exemplo retrospectivo com dois cenários meteorológicos, para a semana iniciada em 29/12/2025:

```powershell
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe -m earthfire.dual_pipeline --config config/zarr_brazil_v2.json --run-id zarr_brasil_v2_20261111 scenario `
  --latitude -11.5 --longitude -46.875 --reference-date 2025-12-28 `
  --overrides-file config/scenario_example.json `
  --output runs/zarr_brasil_v2_20261111/exports_v2/scenario_20251228.json
```

A mesma inferência está disponível em Python por `earthfire.dual_scenario.predict_week(latitude, longitude, reference_date, run_dir, scenario_overrides=...)`. Ela carrega o modelo treinado, aceita emissão em domingo UTC e tem horizonte fixo de sete dias. A meteorologia futura padrão é uma climatologia de treino, não uma previsão POWER. Alterações no clima são cenários condicionais, sem interpretação causal.

## Instalação

Python 3.10 ou superior. No PowerShell, dentro do projeto:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e '.[test]'
```

Na execução documentada foi utilizado um ambiente `.venv` com acesso de leitura aos pacotes científicos previamente instalados (`--system-site-packages`); `pyro-ppl` foi instalado apenas no ambiente do projeto. `runs/*/versions.json` registra as versões efetivamente usadas. Uma instalação nova sem `--system-site-packages` oferece isolamento de dependências mais forte e pode exigir download maior do PyTorch.

## Execução anterior v1

Os comandos abaixo reproduzem o primeiro experimento, realizado **antes** da inclusão dos arquivos MODIS de 2018–2025. Ele usa outra seleção de células e deve ser interpretado separadamente do v2.

### Executar e retomar

Os ZIPs são lidos diretamente em `DATA/`; o projeto não os extrai nem modifica. A auditoria completa percorre todos os CSVs e pode ser demorada. Cada ZIP concluído fica em checkpoint; ao repetir o comando, seu SHA-256 é verificado antes do reaproveitamento. Para fazer etapas de uma mesma execução compartilharem o orçamento de oito horas, escolha um identificador de orçamento:

```powershell
$env:EARTHFIRE_BUDGET_ID = 'minha_execucao_01'
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe -m earthfire.cli --run-id auditoria_01 audit --mode full
.\.venv\Scripts\python.exe -m earthfire.cli --run-id brasil_01 prepare
.\.venv\Scripts\python.exe -m earthfire.cli --run-id brasil_01 features
.\.venv\Scripts\python.exe -m earthfire.cli --run-id brasil_01 train
.\.venv\Scripts\python.exe -m earthfire.cli --run-id brasil_01 evaluate
.\.venv\Scripts\python.exe -m earthfire.cli --run-id brasil_01 export
.\.venv\Scripts\python.exe -m earthfire.cli --run-id brasil_01 predict --latitude -18.5 --longitude -57.5 --reference-date 2024-01-07
```

`features` consulta a NASA POWER para as células configuradas, usa cache persistente e requer acesso à internet na primeira execução. Auditoria e preparação são independentes da rede. Para trocar dados, células, cortes ou limites, crie uma configuração e passe `--config caminho.json`; uma configuração alterada requer outro `run_id`. É possível fornecer `source_countries` com os nomes usados nos CSVs dentro dos ZIPs e `weather_period` com `start` e `end` no formato ISO. Amplie em lotes de células e execuções independentes: os registros selecionados de cada lote são reunidos em memória para deduplicação, portanto o código não deve ser configurado para carregar todas as células globais de uma só vez. A preparação verifica SHA-256 e retoma somente se os insumos forem idênticos. Cada modelo salva estado do Pyro, otimizador, gerador aleatório e hashes dos exemplos e do código; checkpoints completos compatíveis não são retreinados.

Depois de preparar e construir as features, um teste rápido usa aproximadamente um quarto das semanas de treino, amostradas dentro de cada região e ano, e 20 passos de otimização. Seus resultados são **demonstrativos**, identificados pelo prefixo e pela versão `DEMO`:

```powershell
.\.venv\Scripts\python.exe -m earthfire.cli --run-id quick_01 quick --source-run brasil_01
.\.venv\Scripts\python.exe -m pytest -q
```

O orçamento limita prazo, reserva, tempo de treino, número de treinos completos, memória dos processos registrados, espaço e requisições. Veja `config/default.json`, `runs/budget_*.json` e [metodologia](docs/metodologia.md). `EARTHFIRE_BUDGET_ID` deve ser o mesmo em processos executados em paralelo; não execute treinos simultâneos com o mesmo orçamento, pois a reserva de vagas de treino é sequencial. Use um novo identificador para uma nova janela de trabalho.

## Cenário local da execução v1

O modelo de cenário recebe uma célula treinada, um domingo UTC e a semana seguinte. O exemplo a seguir é uma **simulação retrospectiva**, não uma previsão emitida em 2024:

```powershell
.\.venv\Scripts\python.exe -m earthfire.cli --run-id brasil_01 scenario `
  --latitude -18.5 --longitude -57.5 --reference-date 2024-01-07 `
  --horizon-days 7 --overrides-file config/scenario_example.json `
  --output runs/brasil_01/exports/scenario.json
```

O JSON inclui a área nominal, período e fonte das entradas, modelo, qualidade, intervalos preditivos condicionais e estados. O PNG ao lado compara os cenários se ambos estiverem dentro da distribuição de treino. Para permitir alterações climáticas, o simulador carrega sempre a variante com meteorologia, mesmo quando a validação selecionar a variante sem meteorologia para os exemplos gerais. A alteração de umidade é em pontos percentuais; precipitação e vento usam multiplicadores. Sem arquivo de previsão meteorológica, a trajetória futura é a climatologia do treino, um cenário fixo e não uma previsão do endpoint POWER Daily.

Para uso por Python:

```python
from earthfire.scenario import simulate_fire_scenario

result = simulate_fire_scenario(
    latitude=-18.5, longitude=-57.5, reference_date="2024-01-07", horizon_days=7,
    spatial_config={"run_dir": "runs/brasil_01", "power_cache": "runs/power_cache"},
    scenario_overrides={"temperature_delta_c": 1.0, "relative_humidity_delta_percentage_points": -3.0},
)
```

O contrato opcional de `forecast_path` e as razões para `insufficient_data`, `weather_unavailable` e `out_of_distribution` estão no [dicionário](docs/dicionario.md). O simulador nunca retreina o modelo. Só carrega checkpoints locais confiáveis: arquivos PyTorch usam serialização que não deve ser aberta de fonte desconhecida.

## Resultados e documentação da execução v1

- [Relatório de auditoria](docs/auditoria.md), incluindo progresso e inventário detalhado por arquivo em `runs/full_20261111/audit_full/`.
- [Metodologia](docs/metodologia.md), [dicionário de dados](docs/dicionario.md), [fontes](docs/fontes.md), [decisões](docs/decisions.md) e [model card](docs/model_card.md).
- [Avaliação](docs/avaliacao.md), [estado da execução](STATUS.md) e [relatório de encerramento](OVERNIGHT_REPORT.md).
- Os contratos de integração estão em `runs/<run_id>/exports/`: GeoJSON de retângulos, calendário JSON, previsões retrospectivas JSON e Parquet processado. Os retângulos são limites nominais; a entrada usa somente o arquivo de registros denominado Brazil, sem polígono de fronteira para recorte preciso.

Uma detecção não equivale a um incêndio independente, área queimada, ignição, propagação ou dano. Ausência de registro e anomalia não demonstram causa ambiental. Não use este experimento como alerta operacional ou apoio a decisões de emergência.
# Chave NASA FIRMS no Windows

O piloto de expansão multicontinental está documentado em [Expansão global](docs/expansao_global.md). Ele mantém a versão VIIRS diária separada e implementa coleta MODIS/POWER em partições, seleção na validação, teste espacial e consulta com bloqueio por falta de suporte. Os resultados medidos estão em [REPORT_GLOBAL_V5.md](REPORT_GLOBAL_V5.md).

O cliente utiliza `FIRMS_MAP_KEY` quando definida. No Windows, caso não haja variável de ambiente, ele procura a credencial `PYRO/FIRMS_MAP_KEY` no Gerenciador de Credenciais do usuário. A credencial fica vinculada ao perfil local e não é incluída no projeto.

Para cadastrá-la novamente em outra máquina, instale a opção Windows no ambiente isolado (`.\.venv\Scripts\python.exe -m pip install -e ".[windows]"`) e execute `.\.venv\Scripts\python.exe -m earthfire.credentials store-firms`. A entrada não é exibida no terminal. Uma chave diferente existente não é substituída automaticamente. Em outros sistemas, configure `FIRMS_MAP_KEY` pelo mecanismo seguro do ambiente.
