# Relatório — complementação NASA e previsão condicional v3

> Este relatório preserva a situação da etapa v3. Posteriormente, grânulos VIIRS protegidos foram obtidos e um modelo diário foi treinado; resultados atualizados em [REPORT_VIIRS_V4.md](REPORT_VIIRS_V4.md).

Execução em 11/11/2026. O projeto preserva o experimento semanal v2. Este relatório descreve **outro alvo**: o aparecimento de pelo menos um pixel MODIS/Terra classificado como foco em um composto de oito dias, condicionado à futura máscara estar integralmente válida no recorte. Isso não é chance de ignição, incêndio presente, dano ou fogo não observado. Uma máscara válida no composto não assegura ausência de nuvens em todos os dias ou observação contínua.

## Dados acrescentados e auditoria

| Fonte efetivamente obtida | Escopo | Registros ou composições | Observação |
|---|---|---:|---|
| FIRMS VIIRS NOAA-20, arquivo por país | Brasil, 2023 | 1.312.648 | Datas de 01/01 a 31/12; não comprova cobertura completa |
| FIRMS VIIRS NOAA-20 NRT | América do Sul, 04–11/11/2026 | 63.159 | Retrato de sete dias; sujeito a revisão |
| FIRMS MODIS/Terra e Aqua NRT | América do Sul, 04–11/11/2026 | 19.215 | Retrato de sete dias, não o ano todo |
| MOD14A2/ORNL FireMask | Oito áreas, principalmente 2018–2025 e parte de 2026 | 3.000 | 2.809 recortes com todos os pixels em classes válidas; 191 insuficientes; oito períodos solicitados de 2018–2025 ausentes |

Nos três CSVs auditados não foram encontradas duplicatas exatas nem coordenadas, datas, horários ou campos examinados inválidos. A auditoria de confiança trata o histórico VIIRS `l/n/h` e as categorias NRT `low/nominal/high` por produto, sem convertê-las em probabilidade. A primeira auditoria interpretou incorretamente as categorias longas e foi preservada; a corrigida está em `runs/audit_nasa_v3b_20261111/audit/`.

Arquivos originais NASA, datas de obtenção e SHA-256 estão em `runs/nasa_complement_v3_20261111/firms_raw/manifest.json`. Resumo, geometria, consultas, lacunas e hashes das máscaras estão em `runs/nasa_complement_v3b_20261111/coverage/`. `DATA/` permanece somente para leitura. As fontes e os limites estão em [fontes](docs/fontes.md) e [decisões](docs/decisions.md).

Os arquivos anuais públicos de 2026 tentados retornaram 404; isso não prova que todos os arquivos anuais sejam inexistentes. Não obtivemos 2026 inteiro nem dados futuros. Para consultas FIRMS adicionais, o cliente `earthfire.firms_api` usa `FIRMS_MAP_KEY` no ambiente, caso o usuário obtenha uma chave NASA. Exemplos:

```powershell
$env:PYTHONPATH = 'src'
# Configure FIRMS_MAP_KEY no ambiente sem registrar a chave em arquivos ou na conversa.
.\.venv\Scripts\python.exe -m earthfire.firms_api --run-id consulta_firms --source VIIRS_NOAA20_SP --bbox -74 -34 -32 6 --start 2026-01-01 --days 5
```

O cliente limita cada consulta a cinco dias e preserva resposta e hash; dados baixados por ele precisam ser auditados e integrados numa nova preparação antes de treinar. A chave não é registrada nos logs. O endpoint tem limites próprios, e uma chave não prova cobertura nem completude das detecções. A [documentação NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/api/area/) explica o acesso.

## Cobertura, unidade e exemplo de uso

O produto obtido foi MOD14A2/Terra via ORNL: classes de fogo, não fogo, nuvem e ausência são avaliadas separadamente. A resposta de recorte não informou a coleção, por isso ela está marcada como desconhecida. Os oito recortes correspondem a matrizes de 41 × 41 pixels de aproximadamente 927 m em projeção sinusoidal (recorte solicitado com raio de 20 km). A máscara é do MODIS, não prova cobertura do VIIRS de 375 m. Classes desconhecidas e nuvens não viram negativos. Mesmo um composto válido resume passagens e não prova observação em todos os instantes.

Para a etapa VIIRS, o catálogo público NASA CMR identificou dois grânulos [VJ114A1.002](https://doi.org/10.5067/VIIRS/VJ114A1.002) para 14–15/09/2025 no recorte de exemplo. Seus metadados e links estão em `runs/viirs_coverage_nasa_v3_20261111/viirs_coverage/discovery.json`; o download retornou `authentication_required` porque não havia token Earthdata. O produto contém camada `FireMask` e é um composto diário NOAA-20 em grade de 1 km a partir de canais de resolução moderada. Ele pode ajudar a estudar observabilidade, mas não é a máscara do produto FIRMS de 375 m e não foi integrado ou usado no treino.

O coletor está preparado para receber `EARTHDATA_TOKEN` somente por variável de ambiente, sem imprimir ou salvar o token. Após o usuário obter e configurar um [token Earthdata](https://urs.earthdata.nasa.gov/documentation/for_users/user_token), a coleta limitada pode ser tentada assim:

```powershell
$env:PYTHONPATH = 'src'
# Configure EARTHDATA_TOKEN no ambiente, sem colar o token no projeto ou na conversa.
.\.venv\Scripts\python.exe -m earthfire.viirs_coverage --run-id viirs_coverage_minha_consulta --bbox -47.2 -11.8 -46.5 -11.2 --start 2025-09-14 --end 2025-09-15 --max-granules 3 --download
```

O comando salva os grânulos protegidos e seus hashes em `runs/`; não os adiciona automaticamente ao treino. É necessário auditar suas classes, projeção, disponibilidade e alinhamento com as detecções de 375 m antes de integrá-los. Sem `--download`, a descoberta pública funciona sem token. Consultas abrangentes exigem particionamento e novos identificadores de execução.

```powershell
.\.venv\Scripts\python.exe scripts\testar_mascara_real.py
```

Na célula (-11,5°, -46,875°), para 14–21/09/2025, a previsão retrospectiva salva foi **89,3% condicionados à máscara válida**; o intervalo de 80% sobre parâmetros foi 81,8%–95,3%. Na máscara houve 21 pixels com foco em 1.681 pixels válidos. A inferência foi realizada em 2026 com dados históricos que podem ter sido revisados. Esse caso é demonstrativo, não prova de calibração; o intervalo de parâmetros não é um intervalo preditivo de contagens.

`earthfire.recent_firms lookup` reconhece coordenadas geográficas válidas; o histórico NRT usado aqui cobre apenas América do Sul. O modelo `earthfire.mask_pipeline predict` identifica a área, verifica suporte na validação, limites das features e disponibilidade das fontes. Sua saída distingue `probability_given_full_mask` de `event_probability: null`; esta última continuou indisponível porque a futura oportunidade de observação não é conhecida. O status `experimental_conditional` não autoriza generalização global. A consulta dinâmica usa cache e compara hashes; a avaliação retrospectiva pode ser reproduzida offline nas regiões já coletadas.

## Treinamento, separação e avaliação

Selecionamos oito áreas usando a disponibilidade do recorte anterior; não são uma amostra aleatória do Brasil. Seis participaram do treino e duas foram reservadas espacialmente. As janelas completas até 2022 foram usadas no treino, 2023 na validação e 2024–2025 no teste. Janelas que cruzam cortes ou têm máscara desconhecida não foram usadas como rótulo. Ajuste de normalização/imputação foi feito no treino; a escolha da variante foi feita pelo Brier da validação antes do teste. Arquivos finais podem ter revisão posterior, de modo que o experimento não prova disponibilidade histórica operacional.

Treinamos regressões logísticas bayesianas em Pyro/PyTorch em 1.234 exemplos de treino, com 156 eventos, para as variantes com e sem meteorologia. Ambas usam sazonalidade, posição e histórico disponível MODIS/VIIRS; a variante meteorológica adiciona histórico POWER, nunca clima futuro observado como se fosse previsão. Períodos VIIRS sem registros permanecem desconhecidos, com indicador explícito. Os registros de 2023 foram incorporados à validação, não ao ajuste do treino que termina em 2022. O experimento não trata a harmonização v2 como rótulo exato e não é uma validação de contagem VIIRS harmonizada.

O Brier é o erro quadrático médio da probabilidade para o evento definido; valores menores são melhores. Na validação temporal das regiões de treino, o Brier foi 0,09292 sem meteorologia, 0,09679 com meteorologia e 0,09646 para a referência sazonal. Portanto, foi selecionada a versão **sem meteorologia** antes de consultar o teste.

| Teste 2024–2025 | Exemplos | Modelo sem clima | Modelo com clima | Referência sazonal | Persistência |
|---|---:|---:|---:|---:|---:|
| Todas as oito áreas | 697 | 0,08289 | 0,09013 | 0,09237 | 0,09861 |
| Seis áreas de treino | 518 | 0,08049 | 0,08062 | 0,07552 | consultar métricas completas |
| Duas áreas reservadas | 179 | 0,08982 | 0,11764 | 0,14110 | consultar métricas completas |

O agregado favoreceu o modelo selecionado, mas nas seis áreas de treino a referência sazonal foi melhor no teste. As áreas reservadas são apenas duas e sua seleção não representa todas as regiões. Os resultados não demonstram previsão atual, calibração operacional ou recuperação da atividade real. As métricas por área e mês, Brier, log loss, ROC-AUC, precisão média e calibração estão em `runs/nasa_complement_v3b_20261111/evaluation/metrics.json`; probabilidades e observações em Parquet e um gráfico de calibração estão na mesma pasta.

Um critério fixado antes do teste libera a estimativa condicional apenas se a validação do grupo tiver ao menos 30 exemplos, cinco positivos e Brier não pior que a referência sazonal. Isso é uma barreira mínima, não garantia de calibração local. Entradas muito fora do intervalo aproximado do treino recebem `out_of_distribution`.

## Meteorologia com emissão e validade

Baixamos oito arquivos diários de um membro da previsão retrospectiva NASA GEOS-S2S-3, inicializado em 03/09/2021, e extraímos 64 pares espaço-temporais de temperatura e precipitação para as oito áreas. O histórico de metadados informa produção em 2026: é um **reforecast**, não comprova uma previsão disponível em 2021. Após conversão de unidades e associação à grade mais próxima, a diferença média S2S menos POWER foi 0,76 °C para temperatura e 0,88 mm/dia para precipitação; os erros absolutos médios foram 1,19 °C e 3,56 mm/dia, respectivamente. São somente uma inicialização e um membro, com grades e produtos diferentes; não foram ajustadas correções nem usados como previsão operacional. Arquivos e hashes: `runs/s2s_nasa_v3_20261111/s2s_hindcast/`.

O cliente experimental NASA GEOS-FP consulta ciclos com data de inicialização e validade, mas na primeira tentativa as variáveis responderam HTTP 502 após as retentativas; um ciclo mais recente excedeu o tempo de resposta. A descrição da coleção estava acessível, mas não obtivemos séries completas. As falhas e o cache estão em `runs/geos_nasa_v3_20261111/`. Uma amostra de reforecast não substitui um arquivo extenso de previsões emitidas historicamente; esse requisito continua pendente.

## Reprodução

```powershell
$env:PYTHONPATH = 'src'
$env:EARTHFIRE_BUDGET_ID = 'minha_janela_de_execucao'
.\.venv\Scripts\python.exe -m earthfire.mask_pipeline prepare
.\.venv\Scripts\python.exe -m earthfire.mask_pipeline baselines
.\.venv\Scripts\python.exe -m earthfire.mask_pipeline train --steps 1000
.\.venv\Scripts\python.exe -m earthfire.mask_pipeline evaluate --samples 400
.\.venv\Scripts\python.exe -m earthfire.mask_pipeline predict --latitude -11.5 --longitude -46.875 --date 2025-09-13
.\.venv\Scripts\python.exe scripts\verificar_nasa_v3.py
.\.venv\Scripts\python.exe -m pytest -q
```

Os checkpoints completos são reaproveitados se configuração, código e hashes dos insumos forem compatíveis; não sobrescreva esses resultados para testar configurações diferentes. A verificação v3 relê os 31 arquivos de `DATA/` e compara seus hashes com os manifestos de auditoria; nesta execução, todos conferiram. A suíte final teve 40 testes aprovados. Para reproduzir o recorte de reforecast já baixado: `.\.venv\Scripts\python.exe -m earthfire.s2s_forecast --initial-date 2021-09-03 --days 8 --ensemble 1 --extract` na raiz do projeto. A extração salva uma nova tabela sob o identificador de execução, sem modificar `DATA`.

Pendências de maior impacto: cobertura/nuvens própria do VIIRS e observações de passagem, histórico recente completo por sensor com versões conhecidas, arquivo de previsões meteorológicas realmente emitidas, mais áreas e anos para validar espacialmente, compatibilização meteorológica ajustada só no treino e calibração separada por região e estação. A integração com o simulador climático do v2 permanece experimental; o novo classificador selecionado não utiliza meteorologia e não deve ser apresentado como simulador causal.
