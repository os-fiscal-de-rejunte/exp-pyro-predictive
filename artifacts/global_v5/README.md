# Artefatos do piloto global v5

Pacote congelado para consulta retrospectiva offline, descrito em [REPORT_GLOBAL_V5.md](../../REPORT_GLOBAL_V5.md). O modelo escolhido na validação foi o baseline sazonal; os dois checkpoints Pyro são mantidos para comparação. Cinco áreas passaram no critério demonstrativo de liberação. Não há cobertura mundial operacional.

Execute `.venv/Scripts/python.exe scripts/testar_global_real.py` a partir da raiz do projeto. Para consultar outra área/data preparada, passe `--latitude`, `--longitude` e `--data`. O retorno pode ser `insufficient_data` ou `out_of_distribution`, com probabilidade nula.

Os Parquets deste pacote são produtos processados de dados NASA, com fontes e hashes nos manifestos. Os arquivos brutos, caches e o ambiente Python não estão incluídos. A geometria efetiva do alvo é o recorte MODIS, e não toda a célula meteorológica. Carregue checkpoints PyTorch apenas de fonte confiável.
