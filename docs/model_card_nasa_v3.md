# Model card — MODIS condicional v3

**Versão e implementação:** `mod14a2_8day_no_weather_v3`, regressão logística bayesiana em Pyro/PyTorch. Checkpoint e parâmetros em `runs/nasa_complement_v3b_20261111/models/mask_no_weather.pt`; hashes de dados, configuração e código estão no checkpoint. A versão com clima foi treinada e mantida separadamente, mas não foi selecionada na validação.

**Pergunta:** qual é a probabilidade de pelo menos um pixel ser classificado como foco no composto MOD14A2/Terra do próximo período de oito dias, para o recorte, **condicionada à futura máscara ser válida em todos os pixels**? A emissão é às 23:59:59 UTC do dia anterior ao início do composto. O recorte solicitado tem raio de 20 km e 41 × 41 pixels na resposta examinada. A classe de foco pode representar fontes térmicas distintas; um pixel não equivale a um incêndio independente.

**Entradas:** latitude, longitude, sazonalidade, evento MODIS anterior quando comprovadamente disponível, registros positivos VIIRS NOAA-20 anteriores e indicadores de ausência. A variante não selecionada adiciona histórico de temperatura, umidade, precipitação, vento e umidade modelada do solo NASA POWER. Informação ausente não é considerada ausência de detecção.

**Ajuste:** 1.234 exemplos elegíveis, 156 positivos, seis áreas brasileiras, composições completas até 2022. Validação em 2023 e teste em 2024–2025. Duas áreas não participaram do treino. A seleção por Brier foi feita na validação e os parâmetros de transformação foram ajustados somente no treino. Datas e dados históricos podem ser de produtos reprocessados que não estavam disponíveis na data simulada.

**Evidência:** no teste, Brier 0,08289 no agregado de 697 exemplos, contra 0,09237 da referência sazonal; nas seis áreas de treino, o Brier foi 0,08049 contra 0,07552 para a referência sazonal. As duas áreas reservadas tiveram Brier de 0,08982 contra 0,14110 da referência. Métricas e calibração completas: `runs/nasa_complement_v3b_20261111/evaluation/`.

**Uso adequado:** pesquisa, comparação de modelos e demonstrações retrospectivas identificadas. Áreas novas são consultadas por coordenada e verificadas contra critérios mínimos de validação e extrapolação; isso não substitui calibração naquela área. Dados de nuvens e oportunidade MODIS não estabelecem cobertura VIIRS. A futura validade da máscara é desconhecida, logo a chance incondicional continua indisponível.

**Não utilizar para:** alerta operacional, decisões de evacuação, probabilidade de ignição, fogo não observado, propagação, área queimada ou dano. Não interpretar intervalo de parâmetros como intervalo de contagem futura. A versão selecionada não usa meteorologia e não responde a cenários climáticos. Mais áreas, máscaras por passagem, previsões meteorológicas históricas realmente emitidas, versões e calibração local são necessários.
