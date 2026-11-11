# Model card

- **Nome:** `nb_positive_weather_v1` e `nb_positive_no_weather_v1`.
- **Uso pretendido:** pesquisa e demonstração de intensidade de registros positivos em três células brasileiras, em semanas após outra semana positiva.
- **Saída:** distribuição preditiva de contagem condicional, em detecções VIIRS/NOAA-20 por semana, para uma trajetória meteorológica fixa quando aplicável.
- **Arquitetura:** regressão bayesiana regularizada e binomial negativa truncada em zero; inferência variacional aproximada com Pyro sobre PyTorch; CPU.
- **Dados:** arquivos locais por país e meteorologia NASA POWER/MERRA-2; sem fontes locais para cobertura ou catálogo de incêndios independentes.
- **Divisões:** treino até 2021, validação em 2022, teste em 2024–2025; dados de 2023 ausentes. O teste espacial independente não foi realizado.
- **Limitações:** seleção de semanas positivas, completude desconhecida, sensibilidade a eventos persistentes e a versões de processamento, três células selecionadas, revisões de dados históricos, ausência de observações de oportunidades e impossibilidade de calibrar ocorrência.
- **Usos inadequados:** alerta de emergência, inferir ignições, estimar propagação, área queimada, causa ou dano, extrapolar globalmente ou apresentar a contagem condicional como chance de fogo.
- **Avaliação:** consultar [avaliação](avaliacao.md) para resultados realmente executados e comparações; `DEMO` identifica execução curta e não constitui evidência de desempenho.
- **Reprodução:** hashes e configuração nos manifestos de `runs/`; checkpoints de origem desconhecida não devem ser carregados.
