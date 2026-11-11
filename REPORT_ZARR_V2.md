# Relatório da atualização — NASA POWER Zarr e novos satélites

Execução em 11/11/2026. O modelo foi treinado de fato em CPU com PyTorch/Pyro e dados reais, mas **não é uma previsão operacional de probabilidade de fogo**. A saída científica sustentada é uma distribuição condicional de contagens semanais em áreas selecionadas, restrita a semanas que tiveram registros. O teste revelou desempenho fraco em regiões reservadas e picos.

## Trabalho realizado

- Auditoria integral dos oito ZIPs MODIS adicionados, com 34.582.684 registros globais; preservação dos dados originais, inventários, hashes e resultados anteriores. Os 30 hashes registrados nas auditorias e na preparação conferem; a última verificação compara manifestos, sem reler os ZIPs novamente.
- Seleção reprodutível de 16 células de 0,5° × 0,625° na partição Brazil, incluindo quatro reservadas para teste espacial. Processamento em lotes selecionou 155.580 registros: 22.297 MODIS/Terra, 40.651 MODIS/Aqua e 92.632 VIIRS/NOAA-20. A seleção favorece locais com registros coincidentes e não é amostra global ou nacional representativa.
- Cliente Zarr público da NASA, com cache persistente e verificável, leitura de blocos sob orçamento, retentativas, checkpoint por bloco e Parquet. Obtidos 49.674 dias em 17 células, inclusive uma de controle, 2018–2025, para temperatura, extremos diários, umidade relativa, precipitação, vento e dois indicadores de umidade do solo. Foram 143 requisições e 169.313.956 bytes de cache. Na célula de controle, seis variáveis concordaram com a API preservada em 2.922 dias pareados; diferença máxima de aproximadamente 0,0000293 por variável em sua unidade.
- Harmonização exploratória entre MODIS/Terra e VIIRS/NOAA-20 em 741 semanas pareadas de treino, com referência simples e Pyro hierárquico. A relação é condicional e não identifica atividade real ou detectabilidade separadamente.
- Duas regressões probabilísticas Pyro/PyTorch de contagem positiva, com e sem clima, treinadas em 683 exemplos das 12 células de ajuste, por 900 passos. Contagens históricas dos sensores entram em separado. Treino até 2021; validação 2022; teste 2024–2025. A ausência de arquivo VIIRS de 2023 impede usá-lo nesse recorte. Meteorologia futura real não foi usada na validação/teste: entrou climatologia ajustada no treino. A incerteza do contraste de sensores não foi propagada ao preditor.
- Comparação com persistência, sazonalidade e regressão regularizada, avaliação por região/intensidade, simulação de cenários, GeoJSON, JSON, Parquet, gráficos e estados explícitos. A simulação de 2025 foi marcada como retrospectiva e sinaliza a versão MODIS não vista no treino.

## Resultados e limites

No teste de 403 semanas elegíveis, a variante com clima, selecionada por CRPS na validação, teve MAE de **39,09 detecções/semana** e CRPS 29,36. A variante sem clima teve MAE 36,81 e CRPS 27,03; a regressão log-linear com MODIS, MAE 36,93; persistência, 41,06; referência sazonal, 45,50. Não foi demonstrado ganho do clima nesse teste.

Nas quatro células reservadas, o modelo com clima teve MAE 76,51 e cobertura de 63,5% para o intervalo nominal de 80%, contra MAE 63,58 da regressão simples. Acima do p90 de intensidade do treino, o MAE foi 178,27 e a cobertura de 80% foi 43,1% em 51 exemplos. Para o contraste dos sensores, o MAE logarítmico do Pyro foi 0,587, contra 0,608 da mediana, nos 445 pares de teste; nas células reservadas foi 0,779 contra 0,725.

No treino, o código de versão MODIS presente é `6.03`; nas 403 previsões de teste é `61.03`. O efeito dessa diferença não foi identificado e os intervalos não o incorporam. Os registros mantêm todos os tipos de anomalia e podem incluir fontes térmicas não correspondentes a incêndios de vegetação.

O cenário retrospectivo em `exports_v2/scenario_20251228.json` retornou média de 8,61 registros condicionais e intervalo de 80% [1, 19] na referência; o cenário modificado retornou média de 10,36 e intervalo [2, 21]. Esses valores são condicionais a uma semana-alvo positiva e a histórico positivo dos dois sensores; a modificação não demonstra efeito causal. `event_probability` permanece `null` porque arquivos de detecções não fornecem negativos nem cobertura verificáveis. A previsão para a semana de 09/11/2026 retorna `insufficient_data` por falta de arquivos de 2026 na semana anterior. As análises são retrospectivas e as versões dos dados disponíveis originalmente no momento simulado são desconhecidas.

Detalhes e tabelas: [auditoria](docs/auditoria_zarr_v2.md), [metodologia](docs/metodologia_zarr_v2.md), [avaliação](docs/avaliacao_zarr_v2.md), [model card](docs/model_card_zarr_v2.md), [fontes](docs/fontes.md) e [decisões](docs/decisions.md).

## Testes e recursos

- `python -m pytest -q`: **22 testes aprovados**, incluindo testes sintéticos identificados de alinhamento, ausência, clima futuro, produto, cliente Zarr, modelo, cenários e versões.
- `python scripts/verify_zarr_v2.py`: passou sobre os artefatos reais. Comparou 30 hashes registrados; releu e conferiu 140 objetos do cache; verificou JSON estrito, 16 geometrias, 403 previsões, 43.456 linhas de calendário das quais 33.706 com ausência desconhecida, e proteção do clima futuro em 811 linhas. Resultado: `runs/zarr_brasil_v2_20261111/verification_v2.json`.
- A chamada de treino foi repetida para testar retomada: os dois checkpoints completos foram reaproveitados e o orçamento continuou com três tarefas de treinamento: harmonização, sem clima e com clima. A otimização dos dois modelos preditivos levou aproximadamente 26,5 s e 21,8 s, respectivamente.
- Orçamento `runs/budget_zarr_v2_20261111.json`: início com 12 CPUs lógicas, cerca de 2,25 GB disponíveis, sem CUDA e 65,16 GB livres; limite de novos arquivos de 13,03 GB e de três treinos. As pastas específicas desta atualização ocupam cerca de 202 MB no momento do registro, sem contar artefatos anteriores e ambiente virtual. Logs e checkpoints permanecem em `runs/`.

## Comandos para reproduzir e retomar

No PowerShell, a partir da raiz do projeto. O ambiente isolado e as dependências estão documentados no README. Auditoria e seleção já executadas podem ser revistas com:

```powershell
$env:PYTHONPATH = 'src'
$env:EARTHFIRE_BUDGET_ID = 'reproducao_v2_01'
.\.venv\Scripts\python.exe -m earthfire.audit_incremental --run-id audit_zarr_v2_20261111
.\.venv\Scripts\python.exe -m earthfire.select_regions --run-id selection_zarr_v2b_20261111 --limit 16 --minimum-pairs 25
.\.venv\Scripts\python.exe -m earthfire.cli --config config/zarr_brazil_v2.json --run-id sat_zarr_v2_20261111 prepare
.\.venv\Scripts\python.exe -m earthfire.power_zarr --config config/zarr_brazil_v2.json --run-id zarr_brasil_v2_20261111
.\.venv\Scripts\python.exe -m earthfire.dual_pipeline --config config/zarr_brazil_v2.json --run-id zarr_brasil_v2_20261111 prepare
.\.venv\Scripts\python.exe -m earthfire.dual_pipeline --config config/zarr_brazil_v2.json --run-id zarr_brasil_v2_20261111 harmonize
.\.venv\Scripts\python.exe -m earthfire.dual_pipeline --config config/zarr_brazil_v2.json --run-id zarr_brasil_v2_20261111 train --steps 900
.\.venv\Scripts\python.exe -m earthfire.dual_pipeline --config config/zarr_brazil_v2.json --run-id zarr_brasil_v2_20261111 evaluate --samples 400
.\.venv\Scripts\python.exe -m earthfire.dual_pipeline --config config/zarr_brazil_v2.json --run-id zarr_brasil_v2_20261111 evaluate-harmonization
.\.venv\Scripts\python.exe -m earthfire.dual_pipeline --config config/zarr_brazil_v2.json --run-id zarr_brasil_v2_20261111 export
.\.venv\Scripts\python.exe scripts/verify_zarr_v2.py
.\.venv\Scripts\python.exe -m pytest -q
```

Os comandos acima apontam para os artefatos existentes, não devem ser tratados como treinamento global; a seleção pode percorrer novamente os arquivos de entrada. Os checkpoints são reutilizados se hashes de entrada, configuração e código forem compatíveis. Ao alterar dados, seleção, período ou configuração, crie novos `run_id` e ajuste `selection_manifest`, `--satellite-run`, caminhos e limites de orçamento de maneira consistente. O Zarr é atualizado pela NASA; preservar o cache é necessário para reproduzir esta versão dos dados. As execuções exigem espaço e tempo, mas não modificam `DATA/`.

Exemplo de cenário:

```powershell
.\.venv\Scripts\python.exe -m earthfire.dual_pipeline --config config/zarr_brazil_v2.json --run-id zarr_brasil_v2_20261111 scenario `
  --latitude -11.5 --longitude -46.875 --reference-date 2025-12-28 `
  --overrides-file config/scenario_example.json `
  --output runs/zarr_brasil_v2_20261111/exports_v2/scenario_20251228.json
```

## Próximo passo de maior impacto

Obter registros recentes dos dois sensores **junto com metadados verificáveis de cobertura e oportunidade de observação**, nuvens e processamento histórico por data de emissão. Só então construir rótulos negativos defensáveis, treinar e calibrar uma probabilidade de pelo menos uma detecção, validar previsões meteorológicas históricas e avaliar espacialmente mais regiões. Avaliar explicitamente as versões dos produtos e o efeito de fontes térmicas estáticas antes de qualquer uso operacional.
