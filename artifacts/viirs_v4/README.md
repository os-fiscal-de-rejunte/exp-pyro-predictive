# Artefatos da execução VIIRS v4

O checkpoint `mask_weather.pt` e os arquivos JSON são cópias da execução local `runs/viirs_daily_v4b_20261111`, descrita em [REPORT_VIIRS_V4.md](../../REPORT_VIIRS_V4.md). O checkpoint inclui as transformações e os parâmetros ajustados; as métricas e o resumo registram o recorte avaliado.

Esses arquivos preservam o modelo e sua avaliação, mas não constituem uma execução completa: os dados de entrada, caches, manifestos e demais produtos de `runs/` não são distribuídos neste repositório. Para reproduzir a execução ou consultar o histórico local, obtenha os dados e siga as instruções do relatório. O modelo não fornece cobertura global validada.

Carregue checkpoints PyTorch somente de origem confiável: a desserialização pode executar código.
