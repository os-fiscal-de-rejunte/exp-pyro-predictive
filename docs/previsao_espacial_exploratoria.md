# Previsão espacial exploratória

A consulta pública `earthfire.spatial_predict.forecast` permite estimar o evento em células sem histórico local. O comando `scripts/testar_expansao_real.py` usa esse adaptador por padrão. Nenhum treinamento adicional ou acesso de rede ocorre na consulta.

Primeiro é executada a rota v6 validada. Se houver previsão disponível, seu número permanece igual. Fora desse suporte, aplica-se o checkpoint real `mask_no_weather.pt` de Pyro/PyTorch. O JSON retorna `status: exploratory`, `prediction_type: preditivo_exploratorio`, `is_predictive: true`, `probability_locally_calibrated: false`, disponibilidade de histórico e distância à célula mais próxima da amostra de treinamento. O campo `within_validated_coverage` preserva a distinção entre cálculo possível e cobertura validada.

As entradas são centro da célula nativa, latitude, longitude e sazonalidade do próximo composto. O histórico indisponível usa os indicadores de ausência e as transformações já salvas no treinamento. Não se fabricam observações negativas. Esta versão não recebe temperatura nem consulta meteorologia: `weather_context` informa explicitamente essa condição. O resultado continua sendo a probabilidade Bernoulli estimada de detecção MOD14A2 no composto de até oito dias, condicionada à classificação válida; não é probabilidade validada de ignição.

O intervalo de 95% representa a posterior variacional dos parâmetros. Não inclui toda a incerteza de transferência geográfica ou de observabilidade. Coordenadas fora do intervalo observado no treino são sinalizadas; estar dentro dos limites de latitude/longitude não demonstra estar dentro da distribuição ambiental. Não há verificação de vegetação, oceano ou uso do solo nesta rota. Pode haver estimativas inadequadas nesses locais. As emissões permanecem limitadas a 2023–2027 para não estender silenciosamente um modelo desatualizado.

## Uso

```powershell
.\.venv\Scripts\python.exe scripts/testar_expansao_real.py --latitude 40.7128 --longitude -74.006 --data "2026-11-12T12:00:00Z"
```

Para conservar o comportamento restrito à validação:

```powershell
.\.venv\Scripts\python.exe scripts/testar_expansao_real.py --latitude 40.7128 --longitude -74.006 --validated-only
```

Na integração Python:

```python
from earthfire.spatial_predict import forecast
result = forecast(40.7128, -74.006, "artifacts/maxcoverage_v6")
```

O núcleo `earthfire.expansion_predict` continua estrito e congelado. Os artefatos, hashes e métricas v6 não foram alterados. As instruções antigas do README dentro do pacote referem-se ao comportamento estrito; use `--validated-only` para reproduzi-lo no comando atualizado. A estimativa exploratória não aumenta a área de cobertura validada e não herda a acurácia das 498 células liberadas. Falhas de integridade geram erro; falta de checkpoint não vira probabilidade zero.
