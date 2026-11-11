"""Resumo legível gerado dos relatórios completos, sem recontar nem inventar dados."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

from .io import read_json


def write_audit_report(audit_dir: Path, target: Path) -> None:
    index = read_json(audit_dir / "inventory.json")
    reports = {item["archive"]: read_json(Path(item["report"])) for item in index["archives"]}
    lines = ["# Auditoria dos dados", "", f"Estado: **{'concluída' if index['complete'] else 'parcial'}**; "
             f"{len(reports)} ZIPs inventariados. Cada CSV foi lido integralmente apenas nos ZIPs processados; "
             "os relatórios JSON preservam detalhes por país, coluna, categoria e SHA-256.", "",
             "| Arquivo | Tamanho (bytes) | Registros | Data mínima | Data máxima | Membros CSV | Duplicatas exatas |",
             "|---|---:|---:|---|---|---:|---:|"]
    total = 0
    invalid: Counter = Counter(); missing: Counter = Counter()
    versions: dict[str, Counter] = defaultdict(Counter)
    satellites: dict[str, Counter] = defaultdict(Counter)
    coordinates: dict[str, list[float]] = defaultdict(list)
    outside = 0; duplicates = 0
    for name, item in reports.items():
        duplicate = item.get("duplicate_of")
        source = reports.get(duplicate) if duplicate else item
        if not source:
            lines.append(f"| {name} | {item.get('bytes', 0)} | desconhecido | — | — | — | cópia ainda não resolvida |")
            continue
        members = source.get("members", [])
        dates_min = [x["date_min"] for x in members if x["date_min"]]
        dates_max = [x["date_max"] for x in members if x["date_max"]]
        count = source.get("row_count", 0)
        dup_text = f"cópia integral de {duplicate}" if duplicate else str(source.get("exact_duplicates", "não calculado"))
        lines.append(f"| {name} | {item.get('bytes', 0):,} | {count:,} | {min(dates_min) if dates_min else '—'} | "
                     f"{max(dates_max) if dates_max else '—'} | {source.get('total_csv_members', 0)} | {dup_text} |")
        if duplicate: continue
        total += count; duplicates += source.get("exact_duplicates") or 0
        product = source.get("product_in_filename", "unknown")
        for m in members:
            invalid.update(m["invalid"]); missing.update(m["missing"])
            versions[product].update(m["categories"].get("version", {}))
            satellites[product].update(m["categories"].get("satellite", {}))
            outside += m.get("outside_nominal_year", 0)
            for field in ("latitude_min", "latitude_max", "longitude_min", "longitude_max"):
                if m[field] is not None: coordinates[field].append(m[field])
    lines += ["", f"**Registros em ZIPs únicos processados:** {total:,}. Duplicatas exatas verificadas dentro e entre CSVs do mesmo ZIP único: {duplicates:,}. "
              f"Registros com ano de aquisição diferente do ano nominal do ZIP: {outside:,}.",
              "As quatro cópias integrais entram na tabela, mas não nos totais únicos. Duplicatas exatas são igualdade de todos os campos CSV, não associação de eventos.", ""]
    if coordinates:
        lines += [f"**Extremos geográficos observados**, antes de excluir coordenadas inválidas: latitude "
                  f"{min(coordinates['latitude_min'])} a {max(coordinates['latitude_max'])}; longitude "
                  f"{min(coordinates['longitude_min'])} a {max(coordinates['longitude_max'])} graus.", ""]
    lines += ["## Campos e qualidade", "", "Os CSVs são texto. Os campos comuns são latitude, longitude, scan, track, acq_date, acq_time, satellite, instrument, confidence, version, frp, daynight e type. "
              "MODIS usa brightness e bright_t31; VIIRS usa bright_ti4 e bright_ti5. Os tipos de análise e unidades constam no [dicionário](dicionario.md).", "",
              "Valores ausentes (texto vazio), totais por coluna nos ZIPs únicos processados:", "", "```json", json.dumps(dict(sorted(missing.items())), ensure_ascii=False), "```", "",
              "Valores inválidos por regra, contagens independentes que podem se sobrepor:", "", "```json", json.dumps(dict(sorted(invalid.items())), ensure_ascii=False), "```", ""]
    for product in sorted(versions):
        lines += [f"- `{product}` — satélites: `{dict(satellites[product])}`; valores literais de `version`: `{dict(versions[product])}`."]
    lines += ["", "Os valores de `version` são registrados literalmente: sem metadados do download não se deve inferir com segurança coleção exata, fonte de processamento ou mistura NRT/histórica. "
              "O relatório por arquivo contém categorias de confiança, dia/noite e tipo e os valores ausentes por coluna.", "",
              "## Cobertura, sobreposição e desconhecidos", "",
              "Os nomes dos ZIPs disponíveis têm MODIS em 2000–2010 e VIIRS/NOAA-20 em 2018–2022 e 2024–2025; não há sobreposição temporal local entre os produtos. "
              "O arquivo de 2023 e os anos 2011–2017 estão ausentes. A presença de arquivos anuais e de detecções em um dia não comprova completude, cobertura orbital ou ausência de nuvens. "
              "Não é possível reconstruir dias verdadeiramente sem detecção e não se inventam zeros. "
              "Se houver registros fora do ano nominal, duplicatas entre anos não foram verificadas globalmente; fora isso, anos distintos não podem conter linhas totalmente idênticas com a mesma data. "
              "Pares espaço-temporais de eventos não foram associados.", "",
              "O JSON `global_countries.json` tem 27.008.745 bytes; a inspeção encontrou 246 códigos de país, percentis, cenários SSP e chaves temporais de 2000-07 a 2100-07. "
              "A variável física, unidade e proveniência não são identificadas no conteúdo. Não é utilizado como meteorologia observada.", "",
              "A demonstração leu 14.313.853 linhas dos arquivos Brazil e reteve 61.041 registros em três células; "
              "não removeu linhas selecionadas por horário inválido ou duplicação exata. "
              "Ver [metodologia](metodologia.md), [fontes](fontes.md) e os manifestos da execução."]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
