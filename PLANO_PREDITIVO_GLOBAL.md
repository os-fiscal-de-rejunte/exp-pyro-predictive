# Plano do preditivo global — PYRO

**Situação:** plano para execução posterior. Nenhuma das ampliações abaixo é apresentada como concluída. O escopo é dados, modelos, avaliação e uma interface de consulta; não inclui construir o globo 3D ou outras telas.

## 1. Objetivo e definição do resultado

Criar uma base que aceite coordenadas e período e, quando os dados e a validação permitirem, retorne uma estimativa probabilística para a área correspondente. O usuário poderá selecionar um país, estado ou ponto no futuro; os limites administrativos serão usados para localizar e agrupar resultados, enquanto a previsão continuará vinculada a uma área espacial definida.

**Evento prioritário:** ao menos uma detecção de foco de calor pelo produto e pela referência observacional definidos, na célula selecionada e no período informado. A primeira versão deve demonstrar **um dia**; o horizonte de **até sete dias** será incluído após validação específica e com amostras conjuntas para representar a dependência entre dias. A probabilidade de pelo menos uma detecção em vários dias não será obtida por soma de probabilidades.

Uma detecção não é necessariamente um incêndio independente. Essa saída não representa probabilidade de ignição, área queimada, propagação, prejuízo ou fogo que existe mas não foi observado. Para chamar uma saída de probabilidade, é necessário comprovar que o evento e a observabilidade são identificáveis e que a estimativa é suficientemente calibrada para o uso declarado. Quando isso não for possível, retornar `insufficient_data` com valor `null`; eventualmente oferecer um **índice**, explicitamente identificado como tal, após validação de sua utilidade.

### Duas saídas que não devem ser confundidas

1. **Probabilidade condicional:** chance de detecção supondo que a condição de observação especificada seja satisfeita. A previsão v4 atual pertence a esta categoria.
2. **Probabilidade de detecção futura:** chance para uma referência observacional e janela definidas, considerando a incerteza da própria observação. Só será liberada se houver informação e validação suficientes sobre oportunidades de observação e nuvens. Ela ainda não é a probabilidade de todo fogo presente.

## 2. Ponto de partida verificável

O projeto já contém ingestão, auditoria, cliente NASA POWER, recortes do acervo POWER em Zarr, componentes MODIS/VIIRS, modelos em PyTorch/Pyro, avaliação, exemplos de inferência, testes e checkpoints. A pasta `DATA` deve continuar somente para leitura. Relatórios existentes: [v2](REPORT_ZARR_V2.md), [v3](REPORT_NASA_V3.md) e [v4](REPORT_VIIRS_V4.md).

- O experimento v4 usou seis áreas brasileiras para treino e duas reservadas; não constitui validação global. O teste teve 131 exemplos elegíveis e 14 positivos.
- No teste v4, o Brier foi 0,09088 para o modelo com clima e 0,08628 para a referência sazonal: o modelo **não venceu essa referência no agregado**.
- A máscara `VJ114A1.002` obtida é um produto VIIRS/NOAA-20 diário de 1 km. Ela não deve ser tratada como a máscara das detecções FIRMS de 375 m. Dos 1.144 registros área-dia preparados, 570 tinham classificação insuficiente para o rótulo usado.
- Os dados de 2026 disponíveis são recortes limitados; não há série global completa e operacional. A amostra meteorológica GEOS-S2S obtida é um **reforecast**; não prova disponibilidade como previsão na data histórica. O cliente GEOS-FP teve falhas de rede nas tentativas anteriores.
- A consulta pode reconhecer coordenadas fora do treino, mas reconhecimento de localização não equivale a previsão validada. Nenhuma taxa de sucesso global foi demonstrada.

## 3. Decisões iniciais para execução

As seguintes escolhas são pontos de partida, não resultados garantidos:

| Tema | Escolha inicial | O que pode mudar |
|---|---|---|
| Região | Terras em todos os continentes, progressivamente; oceanos e regiões sem dados recebem estado próprio | Ampliação após avaliação por região |
| Grade | Células globais comuns, inicialmente da ordem de dezenas de quilômetros; avaliar H3 ou outra grade e documentar resolução e variação de área | Resolução final após medir cobertura, esparsidade e custo; o ponto escolhido não terá precisão de hotspot |
| Tempo | Diário em UTC; primeiro horizonte de um dia | Até sete dias depois de validar trajetórias conjuntas |
| Sensor de referência | Propor VIIRS/NOAA-20 375 m para o evento observacional inicial | Confirmar cobertura, versão e disponibilidade antes de fixar o alvo; MODIS e outras plataformas permanecem separadas |
| Clima | NASA POWER histórico e, se disponíveis e compatíveis, previsões meteorológicas NASA com emissão e validade | Não usar reanálise futura como se fosse previsão operacional |
| Saída | JSON versionado por célula, data e horizonte; exportação em Parquet | Formato ampliável para o futuro site |
| Computação | Particionada, em CPU; GPU opcional | Capacidade escolhida depois de medir um piloto |

A resolução nominal do sensor ou da grade não é precisão espacial uniforme. Uma consulta de ponto indicará a célula efetivamente utilizada e seus limites; locais na mesma célula poderão receber a mesma saída.

## 4. Ordem de execução

### Etapa 0 — Preparar uma execução controlada

**Fazer**

1. Ler as instruções locais, `README.md`, `STATUS.md`, relatórios e configurações antes de modificar qualquer coisa.
2. Conferir novamente arquivos disponíveis, versões, espaço em disco, RAM, CPU e eventual GPU. Fazer inventário incremental de novos dados, sem repetir indevidamente auditorias já concluídas.
3. Criar `run_id`, orçamento operacional, manifesto de origem e registro de decisões. Preservar arquivos e alterações existentes; `DATA/` permanece somente leitura.
4. Definir treino, validação, teste e áreas espaciais reservadas **antes** de ajustar transformações, escolher variáveis ou usar o teste. Registrar a configuração e sua versão.
5. Trabalhar com limite de tempo, memória, disco, paralelismo, requisições, retentativas e número de treinamentos definido pelo usuário. Registrar progresso e permitir retomada por checkpoint.

**Entrega:** inventário atualizado, `STATUS.md` e configuração de uma execução reproduzível. A execução futura não deve contratar serviços nem gerar custo pago sem autorização.

### Etapa 1 — Inventariar as fontes globais

Para cada produto, registrar plataforma, instrumento, versão, período real, abrangência, resolução, origem, formato, fuso, tempo de disponibilização, revisões, acesso e restrições. Guardar URL, data de obtenção, tamanho e SHA-256; não registrar credenciais.

| Informação | Fonte inicial | Verificações obrigatórias |
|---|---|---|
| Detecções históricas e recentes | [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/) e [arquivo histórico](https://firms2.modaps.eosdis.nasa.gov/download/) | Datas por produto e continente; distinguir histórico e NRT; auditar duplicatas, datas, confiança específica e versões |
| Oportunidade de observação e nuvens | Produtos originais NASA por sensor e passagem, por exemplo [VIIRS/NOAA-20 375 m](https://doi.org/10.5067/VIIRS/VJ114IMG.002) e sua geolocalização/qualidade | Confirmar quais camadas identificam observação válida, nuvem e ausência; não substituir a máscara de 375 m pelo produto diário `VJ114A1` de 1 km |
| Clima histórico | [NASA POWER Zarr](https://power.larc.nasa.gov/docs/tutorials/service-data-request/aws/) | Confirmar variáveis, unidades, resolução, período e revisões; temperatura, umidade, chuva, vento e, se sustentado, umidade do solo |
| Previsão meteorológica | Investigar produtos [NASA GEOS-FP](https://www.nccs.nasa.gov/geos5-forecast/) e arquivos de ciclos históricos acessíveis | Guardar data de emissão, validade, horizonte e versão; distinguir previsão emitida, reanálise e reforecast |
| Limites administrativos | Fonte global aberta apropriada a países e subdivisões | Conferir licença, versão, topologia, nomes e lacunas; os limites não definem a resolução do modelo |

Fontes e preços consultados em 11/11/2026. O [tutorial POWER](https://power.larc.nasa.gov/docs/tutorials/service-data-request/aws/) informa que os valores do Zarr podem mudar; datas presentes no eixo não provam que existam observações para elas. A [API de área FIRMS](https://firms.modaps.eosdis.nasa.gov/api/area/) descreve sua chave e seus limites; a disponibilidade dos produtos deve ser consultada, sem presumir um arquivo anual completo. O DOI do produto VIIRS 375 m retornou HTTP 403 nesta consulta; sua identificação também consta no [anúncio oficial NASA/LP DAAC](https://forum.earthdata.nasa.gov/viewtopic.php?t=5486), mas as camadas efetivamente acessíveis ainda precisarão ser verificadas.

**Entrega:** catálogo global e matriz por sensor × região × período, com `disponível`, `ausente`, `acesso bloqueado` e `completude desconhecida`. Registrar separadamente informações observadas e hipóteses. Se um serviço exigir autenticação, usar apenas credenciais fornecidas pelo usuário em ambiente seguro; não copiá-las para código, relatórios ou logs.

### Etapa 2 — Escolher e testar a grade global

1. Comparar a grade atual com uma grade global, por exemplo H3, avaliando tamanho real das células, áreas polares, antimeridiano, países pequenos e fronteiras.
2. Definir uma célula de modelagem compatível com a escala do clima, a quantidade de focos e a disponibilidade de máscaras. Testar mais de uma resolução no **treino/validação**, sem otimizar com o teste.
3. Criar identificador estável, centro, geometria e relações com país/estado. Uma célula pode cruzar fronteiras: definir regra de interseção e preservar a geometria original, sem inventar divisão exata dos focos.
4. Documentar associação do clima à célula: área representada, agregação, unidades e limitações. Interpolação não cria resolução meteorológica adicional.

**Entrega:** catálogo de células, geometria, contrato de consulta por coordenadas e relatório de sensibilidade da resolução.

### Etapa 3 — Coletar, normalizar e auditar em partições

1. Processar por **produto, ano/mês e bloco espacial**, sem carregar a base global na RAM. Aproveitar catálogos e checkpoints já existentes.
2. Preservar campos originais e criar campos padronizados para coordenadas, aquisição UTC, plataforma, produto, versão, FRP, dia/noite, confiança e qualidade quando existirem. Não equiparar automaticamente as escalas de confiança dos produtos.
3. Validar coordenadas, data e horário; registrar contagens antes/depois dos filtros, duplicatas exatas e sobreposição de arquivos. Não fundir observações apenas por estarem próximas.
4. Conservar detecções por sensor/plataforma. Usar Parquet particionado para tabelas, Zarr quando adequado a grades e manifestos de arquivos/versões. Impossibilidade de leitura deve gerar erro explícito ou partição marcada como pendente.
5. Construir e auditar primeiro um piloto com regiões de treino de diferentes continentes e, em separado, regiões inteiras para teste espacial. A seleção deve cobrir estações, regimes climáticos, baixa e alta atividade e qualidade variável; registrar o viés de seleção.

**Entrega:** dados padronizados e auditados; relatório global progressivo de cobertura e lacunas. Não declarar o piloto como processamento global concluído.

### Etapa 4 — Reconstruir observabilidade e zeros válidos

1. Para cada produto de referência, analisar dados de passagem/geolocalização e suas máscaras de qualidade e nuvem, na resolução disponível. Verificar se as classes realmente correspondem ao algoritmo de detecção em uso.
2. Agregar à grade os indicadores de oportunidade de observação, nuvem, área/pixels válidos e disponibilidade. Registrar órbita, horário e intervalo a que cada indicador se refere.
3. Manter estados distintos: `positive_detection`, `observed_no_detection`, `cloud_obscured`, `not_observed`, `missing_unknown` e `invalid`. Definir regras e evidências antes de criar cada estado.
4. Só atribuir zero de detecções quando a extração estiver completa para a referência e a condição de observação definida for satisfeita. Mesmo então, zero não comprova ausência de fogo.
5. Medir fração utilizável por região/estação e o impacto das nuvens. Recortes sem informação suficiente não devem produzir rótulos negativos nem probabilidades globais artificiais.

**Critério para avançar:** demonstrar, com exemplos reais e testes, como o produto de **375 m** distingue uma oportunidade válida sem detecção de uma passagem obscurecida/ausente. Se não for possível, continuar com a saída condicional ou `insufficient_data` e documentar a limitação.

### Etapa 5 — Harmonizar MODIS e VIIRS

1. Calcular referências interpretáveis por região, estação, plataforma, dia/noite e qualidade durante a sobreposição. Não somar sensores como se fossem eventos independentes nem usar fator universal.
2. Modelar efeitos de instrumento/plataforma e sazonalidade com uma referência identificável e compartilhamento parcial em Pyro; comparar Poisson e binomial negativa quando o alvo forem contagens. Incluir exposição apenas quando sustentada pelos metadados.
3. Examinar alterações de versão e séries NRT versus científicas. Avaliar discordâncias sem apagar picos; informar incerteza da harmonização.
4. Ajustar parâmetros de harmonização **somente no treino**. Se a previsão usar estimativas harmonizadas, propagar suas amostras/incerteza; não transformar estimativa incerta em rótulo exato.
5. Verificar identificabilidade: focos positivos e sensores sem informação de cobertura podem não separar atividade de observabilidade. A concordância entre sensores não comprova verdade física.

**Entrega:** baseline e modelo probabilístico de harmonização, métricas separadas das métricas de previsão e documentação da escala do índice/estado estimado.

### Etapa 6 — Construir exemplos e clima sem vazamento

1. Definir, para cada emissão, dados disponíveis até aquele instante: histórico por sensor, qualidade/cobertura conhecida, sazonalidade e meteorologia anterior com atraso de publicação realista.
2. Recortar POWER Zarr por partição e alinhar UTC, unidades e geometria; comparar fontes e ajustar eventual correção apenas no treino. Salvar hashes, data de obtenção e período de validade.
3. Para meteorologia futura, usar somente ciclos de previsão com **emissão anterior à previsão de focos**. Se não houver arquivo histórico de previsões, fazer avaliação retrospectiva com clima observado apenas como experimento claramente identificado, sem chamá-la operacional.
4. Calcular defasagens e janelas móveis estritamente retrospectivas; preservar indicadores de ausência. Evitar imputações que usem dados de validação/teste.
5. Construir exemplos de horizonte de um dia e, posteriormente, sequências de até sete dias com emissão e vencimento definidos. Criar verificação automática de que cada fonte existia ou estaria disponível na emissão simulada.

**Entrega:** dataset de treino e avaliação com dicionário, cortes, fontes e regras de disponibilidade.

### Etapa 7 — Treinar e comparar modelos

1. Calcular antes do modelo referências de persistência, sazonalidade e regressão estatística simples no mesmo alvo, recortes e horizontes.
2. Treinar primeiro um modelo temporal probabilístico compacto em PyTorch/Pyro, com sazonalidade, efeitos regionais, histórico de sensores e clima. Complexidade adicional só se melhorar a validação e couber no orçamento.
3. Comparar versões com e sem meteorologia, mantendo os mesmos exemplos. Registrar sementes, parâmetros, versões, estado do otimizador, perdas, consumo e checkpoint.
4. Selecionar arquitetura e hiperparâmetros na validação; congelar a escolha antes do teste. Respeitar limites de treinamentos e tempo configurados.
5. Produzir amostras preditivas e intervalos adequados ao alvo. Para um evento em sete dias, representar a dependência temporal e a incerteza meteorológica quando disponível.

**Entrega:** modelo versionado, baselines, comparação climática e treinamento retomável. Se o modelo perder para uma referência, preservar e divulgar o resultado e usar a referência onde for mais adequada.

### Etapa 8 — Validar no tempo e no espaço

1. Usar treino, validação e teste cronológicos; avaliar origens sucessivas e incluir folgas onde as janelas ou eventos possam se sobrepor.
2. Reservar regiões espacialmente separadas, incluindo continentes, biomas e países não usados no ajuste. Aplicar distância de separação ou blocos espaciais para reduzir contaminação por células vizinhas; registrar exceções e amostras pequenas.
3. Medir Brier, log loss e curva de confiabilidade para ocorrência; para contagens, MAE/RMSE e uma regra probabilística como CRPS, quando aplicável. Medir também cobertura e largura de intervalos preditivos.
4. Comparar com frequência global, persistência, sazonalidade e modelo estatístico. Detalhar por horizonte, região, estação, intensidade, observabilidade e versão de produto. Acurácia com muitos negativos não pode ser a única métrica.
5. Calibrar e definir limites de liberação usando apenas treino/validação. O teste final deve permanecer intocado até o método estar congelado. Informar incerteza das métricas e quantidades de eventos, não somente percentuais.
6. Diferenciar teste retrospectivo de simulação operacional. A ausência de previsões meteorológicas históricas realmente disponíveis impede alegar validação operacional completa.

**Critério para avançar:** estabelecer previamente exigências mínimas de amostras, positivos, calibração e comparação com baseline por grupo. Grupos sem evidência recebem `insufficient_data`; não extrapolar automaticamente o desempenho do Brasil para outro continente.

### Etapa 9 — Inferência e atualização regular

1. Criar um serviço Python independente da interface visual: coordenadas + data de emissão + horizonte → célula, histórico disponível, versão de clima e previsão ou estado explícito.
2. Usar o modelo e as transformações já salvos; nunca retreinar a cada consulta. Verificar suporte espacial, temporal, versão do produto, frescor dos dados e extrapolação.
3. Atualizar por lotes, por exemplo diariamente após a chegada dos dados necessários. Pré-calcular e armazenar previsões de células elegíveis quando isso reduzir custo; horários e frequência finais dependem da latência real das fontes.
4. Implementar cache com validade e proveniência, execução idempotente, monitoramento, alertas de falha e reprocessamento controlado. Falha de uma fonte não deve produzir probabilidade zero.
5. Definir política de revisão de resultados NRT, troca de modelo e rollback. Não divulgar resultados vencidos como atuais.

**Contrato inicial proposto:**

```json
{
  "schema_version": "1.0",
  "status": "insufficient_data",
  "requested_location": {"latitude": -15.79, "longitude": -47.88},
  "cell": {"id": null, "geometry": null, "resolution": null},
  "issued_at_utc": null,
  "target_start_utc": null,
  "target_end_utc": null,
  "event_definition": "Ao menos uma detecção na célula e período, segundo a referência indicada",
  "observation_reference": null,
  "probability": null,
  "probability_condition": null,
  "parameter_interval_80": null,
  "predictive_interval": null,
  "weather_source": null,
  "data_quality": {"observation_status": "unknown", "missing": [], "out_of_distribution": null},
  "model_version": null,
  "preprocessing_version": null,
  "reason": "Exemplo de contrato; nenhuma previsão foi calculada."
}
```

Estados previstos: `ok`, `insufficient_data`, `out_of_distribution`, `weather_unavailable`, `observation_unavailable` e `stale_data`. `probability_condition` deve dizer se a máscara futura válida é uma condição; `parameter_interval_80` não é intervalo preditivo. Os campos e unidades serão testados e versionados antes da integração ao site.

### Etapa 10 — Demonstrar e ampliar

1. Primeiro concluir o fluxo ponta a ponta em um piloto multirregional com fontes reais, baselines e teste reservado.
2. Registrar quais áreas têm suporte e quais não têm. Demonstrar uma consulta válida e uma consulta com `insufficient_data`, ambas rastreáveis até insumos e modelo.
3. Ampliar para mais blocos e períodos somente após conferir custo, cobertura e qualidade. Repetir auditoria e validação por continente.
4. Preparar uma API ou arquivos para o futuro site consultar, sem depender de frontend. País/estado devem ser filtros de navegação; o resultado do ponto continuará referente à célula.
5. Considerar o sistema global pronto somente após atingir os critérios de qualidade definidos para cada grupo e manter rotina atualizada; sem isso, apresentar o alcance efetivamente comprovado.

## 5. Testes e critérios de aceite

- **Integridade:** hashes dos originais e dados NASA; deduplicação exata; retomada compatível com configuração e dados.
- **Geografia:** coordenadas inválidas, polos, antimeridiano, geometria e cruzamento de tiles/fronteiras; ponto associado à célula correta.
- **Observabilidade:** nuvem, não observado, ausência desconhecida e zero verificável tratados separadamente; máscara de 1 km nunca usada como prova automática para 375 m.
- **Tempo:** datas UTC, produção e publicação anteriores à emissão; separação cronológica e espacial; features sem futuro e transformações ajustadas no treino.
- **Modelo:** baselines calculados antes da avaliação final, salvamento/recarga, resultados reproduzíveis com sementes e verificação de recursos; testes sintéticos não contam como evidência científica.
- **Previsão:** probabilidades entre 0 e 1, horizontes e unidades coerentes, incerteza corretamente rotulada, dependência entre dias/células tratada; `null` para saídas não sustentadas.
- **Serviço:** dados vencidos, clima indisponível, local não suportado e falha NASA retornam estado explícito; JSON e Parquet compatíveis com o contrato.
- **Evidência:** relatório separado para harmonização, previsão e calibração; métricas por continente e grupos críticos; comparação com baseline e limitações.

## 6. Recursos, custos e prioridades

Os volumes globais, tempo de processamento e custo total **ainda não foram medidos**. Primeiro executar um piloto, medir arquivos, memória, tempo por partição e taxa de consulta, e extrapolar com margem; depois decidir se cabe em máquina local ou exige nuvem. Não tratar quantidade de horas de GPU como estimativa comprovada para o treinamento global.

| Item | Referência para planejamento | Condição |
|---|---|---|
| Dados NASA | Usar fontes públicas e autenticação quando necessária | Acesso gratuito não elimina custo de armazenamento, rede ou processamento; conferir termos e disponibilidade de cada produto |
| Protótipo local | Pode usar CPU e dados já disponíveis | Alcance restrito às regiões efetivamente validadas |
| GPU opcional | A página de [preços Runpod](https://www.runpod.io/pricing) anunciava RTX A5000 a US$ 0,27/h em 11/11/2026 | Exemplo aritmético: 20 h custariam US$ 5,40 pela máquina, não uma previsão de duração global; verificar preço/disponibilidade e custos adicionais |
| Armazenamento | [Cloudflare R2](https://developers.cloudflare.com/r2/pricing/) informa 10 GB-mês gratuitos e US$ 0,015/GB-mês no armazenamento padrão excedente | Exemplo: 1.000 GB por um mês, após franquia, aproximadamente US$ 14,85 apenas em armazenamento; requisições e computação são adicionais |
| Serviço de consulta | [Google Cloud Run](https://cloud.google.com/run/pricing) cobra conforme recursos usados e possui franquia | O custo depende de tráfego, memória, latência, região e configuração; não foi calculado para o projeto |
| Equipe e tempo | Dependem de escopo, qualidade das fontes e cobertura desejada | Prever aquisição/auditoria, engenharia de dados, ciência, validação e manutenção; não há prazo medido ainda |

O maior custo pode estar em baixar e processar máscaras por passagem, manter históricos meteorológicos com emissão e validar por região, não necessariamente em treinar o modelo. As referências de preço são exemplos, não compras efetuadas ou orçamento fechado.

## 7. Dependências, riscos e bloqueios

| Risco ou dependência | Tratamento planejado |
|---|---|
| Sem máscaras/oportunidades de observação para o produto-alvo | Preservar ausência desconhecida; restringir a saída a uma condição explícita ou `insufficient_data` |
| Clima histórico diferente da previsão futura | Comparar fontes sobrepostas e ajustar correção somente no treino; sem compatibilidade, não liberar previsão operacional |
| Falha, autenticação ou limite de serviço NASA | Cache, checkpoints, retentativas limitadas; registrar bloqueio e continuar tarefas independentes |
| Modelos piores que sazonalidade/persistência | Manter a referência vencedora quando cabível e divulgar os resultados |
| Local ou continente fora do suporte | Retornar estado explícito; não apresentar uma porcentagem extrapolada como calibrada |
| Eventos raros ou poucas observações válidas | Informar tamanho amostral e incerteza; não reduzir critérios apenas para liberar o resultado |
| Mudança de versão e NRT revisado | Versionar insumos, reavaliar diferenças e rastrear toda previsão à versão usada |
| Limites de orçamento | Reduzir recorte/lotes, encerrar com checkpoint e atualizar relatório; não iniciar trabalho que exceda a reserva |

## 8. Resultado esperado e próximo passo

O produto final deve permitir que um programa consulte uma coordenada e período e receba uma saída rastreável, com probabilidade calibrada quando sustentada, incerteza e qualidade — ou um motivo claro para indisponibilidade. A cobertura deve ser publicada como **áreas e períodos efetivamente avaliados**, não como promessa de 100% do planeta.

A primeira ação quando este plano for autorizado para execução é a **Etapa 0**, seguida de um inventário global de observabilidade e previsões meteorológicas. O piloto deve ser concluído e avaliado antes de qualquer treinamento ou processamento global de grande escala.
