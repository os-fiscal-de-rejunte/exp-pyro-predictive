# Artefatos v6 — expansão em células nativas MODIS

Pacote offline de dados derivados e modelos reais. O resultado não é um modelo mundial validado. Consulte o relatório de avaliação e `coverage.geojson` para a área efetivamente sustentada.

O evento é ao menos um pixel MOD14A2 com foco no composto de até oito dias, condicionado a todos os pixels 41×41 terem máscara válida. Não é a probabilidade de ignição nem de todo fogo físico. A projeção futura disponível usa referência sazonal/estática validada; não incorpora uma previsão meteorológica atual.

Na raiz do repositório, após instalar as dependências em ambiente isolado:

```powershell
.\.venv\Scripts\python.exe scripts/testar_expansao_real.py
.\.venv\Scripts\python.exe scripts/testar_expansao_real.py --historical
.\.venv\Scripts\python.exe scripts/verificar_expansao_v6.py --run-dir artifacts/maxcoverage_v6 --skip-originals
```

Sem `DATA/`, a verificação usa `--skip-originals`. Isso não substitui a conferência dos originais, já registrada na execução de origem. As consultas não usam rede ou retreinamento. Coordenadas externas à cobertura retornam `insufficient_data` com probabilidade nula.

`model_registry.json` verifica modelos, features e código. `package_manifest.json` cobre os demais arquivos distribuídos. O calendário preserva ausências e não calcula anomalias com imagens cuja produção é posterior à emissão histórica.
