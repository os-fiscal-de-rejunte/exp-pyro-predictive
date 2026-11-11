# Metodologia e limites

## Pergunta, unidade e observabilidade

A unidade desta primeira execução é o registro de detecção nos CSVs do arquivo Brazil, agregado em semanas de segunda-feira 00:00 UTC até a segunda seguinte, em três retângulos de 0,5° de latitude por 0,625° de longitude. Centros: (-18,5; -57,5), (-10; -64,375) e (-7,5; -61,875). A área efetivamente representada está restrita aos registros incluídos no arquivo Brazil; o projeto não dispõe de um polígono confiável para quantificá-la. A grade acompanha a resolução nominal meteorológica, não a precisão de um hotspot nem uma afirmação de identidade perfeita entre limites de grades. Todas as categorias de `type` são mantidas: as anomalias térmicas podem incluir fontes que não são fogo de vegetação.

Não há como inferir nuvens, passagem dos satélites, exposição ou completude da extração somente com registros positivos. `missing_unknown` cobre tanto ausência de cobertura quanto outros motivos possíveis. Não se preenchem zeros e nenhuma ausência significa ausência de fogo. Os CSVs não são prova independente de completude.

## Ingestão e rastreabilidade

Leitura em blocos diretamente no ZIP; campos originais, hora como texto, horário convertido para UTC, produto, arquivo, membro, linha e SHA-256 são preservados nos registros selecionados. Coordenadas inválidas e datas/horas inválidas são removidas, com contagens no manifesto. As demais validações são contadas, sem filtro arbitrário de confiança nem conversão de escalas. Duplicatas exatas são removidas somente quando todos os valores originais e o produto coincidem; observações próximas ou de sensores diferentes não são agrupadas.

A auditoria calcula SHA-256 dos ZIPs para reconhecer cópias. Em cada ZIP único, identifica candidatos a repetição por hash de todas as colunas, depois compara os valores originais para verificar duplicatas exatas. Isso cobre sobreposição exata entre arquivos de países do mesmo ZIP; diferentes anos só são comparáveis se houver datas fora do ano nominal, situação sinalizada na auditoria. Não identifica eventos comuns com medidas diferentes. Arquivos adicionais ou alterações de hash interrompem a retomada da preparação.

## Harmonização

No arquivo disponível não há período de sobreposição: MODIS/Terra e Aqua ocorrem até 2010 e VIIRS/NOAA-20 a partir de 2018. A saída efetiva é `insufficient_data`, sem multiplicador estimado. Para dados futuros que contenham sobreposição, o código pareia apenas semanas e células com detecções em MODIS/Terra e VIIRS/NOAA-20. A referência descritiva usa mediana da diferença `log1p(VIIRS) - log1p(MODIS)` por região e mês; o modelo Pyro compartilha informação entre regiões e inclui termos sazonais, com intervalos posteriores. Ambos são condicionados aos pares positivos e não representam toda a população de semanas.

O parâmetro de atividade latente e as eficiências de detecção não são identificáveis apenas por listas positivas. Escolher MODIS/Terra como referência é uma convenção, não uma declaração de verdade. Cobertura, nuvens, plataformas, qualidade, geografia e tempo podem ser confundidos. Menor discrepância não valida a recuperação de incêndios reais. A harmonização implementada é exploratória e não é aplicada ao treino real sem pares; consequentemente não há incerteza de harmonização para propagar na execução reportada.

## Previsão experimental

Alvo: número de registros VIIRS/NOAA-20 na célula e semana, **condicionado a haver ao menos um registro na semana-alvo e também na semana anterior**. Esta seleção permite estudar intensidade entre semanas observadas, mas não produz probabilidade incondicional nem identifica semanas que deveriam ser zero. Nenhuma janela com semana anterior ausente é convertida em zero; ela é excluída. O arquivo pode estar incompleto mesmo nas semanas positivas, então a resposta é condicional aos registros disponíveis.

Treino: semanas inteiras até 2021-12-31; validação: semanas inteiras de 2022; teste: semanas inteiras em 2024–2025. O ano de 2023 está ausente. Semanas que cruzam os limites entre conjuntos são excluídas. A célula/região é escolhida usando o histórico anterior ao teste. O horizonte é uma semana, sem alegação de previsão diária. As três células são insuficientes para uma conclusão de generalização espacial; não houve teste espacial independente.

As features incluem log da contagem da semana anterior positiva, senos/cossenos sazonais e região; o modelo meteorológico adiciona a semana terminada pelo menos sete dias antes da emissão e a trajetória meteorológica hipotética da semana futura. Temperatura, máxima e mínima são médias das respectivas variáveis diárias; precipitação é soma semanal em mm, umidade é média em %, vento médio em m/s. São exigidos sete dias válidos. A climatologia mensal/regional usada como trajetória futura em validação e teste foi calculada somente com semanas meteorológicas terminadas no treino. Os parâmetros e os normalizadores também são ajustados somente no treino.

O modelo é uma regressão bayesiana com intercepto e coeficientes regularizados em Pyro/PyTorch e binomial negativa truncada em zero. A truncagem corresponde à seleção do alvo, não a uma hipótese de zeros observados. A binomial negativa acomoda dispersão; o diagnóstico empírico e resultados ficam na avaliação. A posteriori variacional é aproximada; as amostras preditivas combinam incerteza aproximada dos parâmetros com variabilidade de contagem. Os intervalos são para contagens futuras **condicionais**, não para parâmetros ou chance de fogo. Persistência, média sazonal regional e regressão log-linear regularizada usam os mesmos exemplos e horizontes.

Na avaliação, a variante é selecionada pelo CRPS de validação e essa escolha é gravada antes do teste. MAE e RMSE são em detecções/semana; CRPS e log score medem qualidade distributiva; cobertura e largura dos intervalos medem calibração e precisão condicionais. Também há divisões por região, trimestre e intensidade definida no treino. Os dois modelos são apresentados no teste como ablação previamente definida. Uma amostra pequena, dependência temporal e caudas fortes limitam a interpretação.

As séries POWER históricas foram obtidas agora e podem ter sido revisadas depois da data simulada. A versão/latência dos focos disponível na data de emissão também não pôde ser reconstruída. Assim, as divisões obedecem à ordem dos eventos, mas **não comprovam um backtest operacional sem vazamento de versões**. Não há previsões meteorológicas arquivadas de emissão histórica. A climatologia é uma entrada hipotética disponível pelo histórico, não meteorologia observada futura disfarçada de previsão.

## Cenários e anomalias

O simulador carrega o modelo meteorológico, busca a semana anterior de atividade e meteorologia anterior com defasagem, e usa climatologia ou previsão externa explicitamente emitida antes da simulação. Alterações entram nas mesmas features usadas pelo treino. Verificações físicas e uma regra de extrapolação (além do mínimo/máximo de treino por mais de um desvio-padrão) são aplicadas antes da inferência; ela é um alarme simples, não garantia de suporte conjunto.

Uma trajetória meteorológica fixa não representa incerteza meteorológica. A comparação é uma associação do modelo, não uma inferência causal. O estado principal permanece `insufficient_data` porque o evento de ao menos uma detecção não foi calibrado; `conditional_count_status: ok` pode coexistir com esse estado. Brier, log loss e curvas de confiabilidade de ocorrência não são calculáveis sem rótulos verificáveis de não ocorrência.

O calendário usa a mediana de semanas positivas anteriores do mesmo mês, região e produto, após pelo menos três registros, e informa a diferença observada. Trata-se de anomalia entre semanas positivas observadas, não climatologia de todas as semanas e não previsão. A referência é atualizada somente depois de avaliar a semana corrente.

## Recursos e reprodução

O orçamento compartilhado acompanha relógio, espaço livre e memória RSS dos processos registrados e interrompe tarefas nos pontos de verificação; o processo de auditoria iniciado antes da implementação do orçamento foi acompanhado separadamente. O monitor não conta perfeitamente memória compartilhada ou processos externos, e a meta de 60% é conservadora, não um limite de sistema operacional. A cada 250 passos ou 15 minutos o treino salva parâmetros, otimizador e gerador; também salva ao detectar limite ou interrupção de teclado. A auditoria salva ao concluir cada ZIP. A contagem de requisições POWER é persistente, com até três tentativas; espera solicitada superior ao limite local encerra a consulta sem antecipar o serviço. Sementes, versões e hashes são registrados, mas bibliotecas, plataforma e operações numéricas podem produzir diferenças.
