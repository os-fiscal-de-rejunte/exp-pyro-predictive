"""Compara uma previsão retrospectiva com os registros reais de uma semana."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthfire.dual_scenario import predict_week  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Teste retrospectivo com dados reais de satélite e NASA POWER")
    parser.add_argument("--latitude", type=float, default=-11.5)
    parser.add_argument("--longitude", type=float, default=-46.875)
    parser.add_argument("--data", default="2025-11-02", help="Domingo UTC anterior à semana a prever")
    parser.add_argument("--saida", type=Path, help="Arquivo JSON opcional para salvar a comparação")
    args = parser.parse_args()
    run = ROOT / "runs/zarr_brasil_v2_20261111"

    # O resultado observado é consultado somente depois da chamada de inferência.
    result = predict_week(args.latitude, args.longitude, args.data, run)
    start = pd.Timestamp(result["target_week_start"], tz="UTC")
    end = start + pd.Timedelta(days=7)
    raw = pd.read_parquet(run / "processed_v2/records.parquet")
    observed = raw.loc[
        (raw.region_id == result["location"]["region_id"])
        & (raw["product"] == "viirs-jpss1")
        & (raw.satellite == "N20")
        & (raw.acquired_utc >= start)
        & (raw.acquired_utc < end)
    ]
    columns = ["latitude", "longitude", "acq_date", "acq_time", "confidence", "version",
               "frp", "type", "source_archive", "source_member", "source_row"]
    rows = json.loads(observed[columns].to_json(orient="records"))
    count = len(rows) if rows else None
    estimate = result.get("reference_scenario")
    comparable = estimate is not None and estimate.get("status") == "ok"
    interval = estimate["predictive_interval_80"] if comparable else None
    comparison = {
        "tipo": "simulação retrospectiva com dados reais",
        "previsao": result,
        "observado": {
            "deteccoes_viirs_noaa20": count,
            "estado": "positivo" if rows else "ausente_ou_desconhecido",
            "registros": rows,
        },
        "observado_dentro_do_intervalo_80": (
            interval[0] <= count <= interval[1] if interval is not None and count is not None else None
        ),
    }
    print(f"Local: {args.latitude}, {args.longitude} | emissão simulada: {args.data} UTC")
    print(f"Semana prevista: {start.date()} a {(end - pd.Timedelta(days=1)).date()}")
    previous = result.get("previous_week")
    if previous:
        print(f"Semana anterior: VIIRS {previous['viirs_n20_detections']}; MODIS/Terra {previous['modis_terra_detections']} detecções")
    if comparable:
        print(f"Previsão condicional: mediana {estimate['median_count_conditional']:.0f}; intervalo 80% {interval[0]:.1f} a {interval[1]:.1f}")
    else:
        print(f"Previsão condicional indisponível: {result.get('reason', 'dados insuficientes')}")
    print(f"Detecções VIIRS observadas: {count if count is not None else 'ausentes ou desconhecidas'}")
    if comparison["observado_dentro_do_intervalo_80"] is not None:
        print("Observado dentro do intervalo 80%:", "sim" if comparison["observado_dentro_do_intervalo_80"] else "não")
    print("Probabilidade de ocorrência: indisponível; cobertura dos satélites desconhecida.")
    print("Previsão condicionada a uma semana com detecção; simulação feita após a data histórica.")
    warning = result.get("data_quality", {}).get("version_warning")
    if warning:
        print("Aviso:", warning)
    if args.saida:
        args.saida.parent.mkdir(parents=True, exist_ok=True)
        args.saida.write_text(json.dumps(comparison, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print("Dados e resultado salvos em:", args.saida)


if __name__ == "__main__":
    main()
