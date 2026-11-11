"""Consulta uma previsão de teste salva e a compara com a máscara MODIS real."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "nasa_complement_v3b_20261111"


def main() -> None:
    parser = argparse.ArgumentParser(description="Exemplo retrospectivo real, sem acesso à internet")
    parser.add_argument("--latitude", type=float, default=-11.5)
    parser.add_argument("--longitude", type=float, default=-46.875)
    parser.add_argument("--data", default="2025-09-13", help="Emissão simulada, AAAA-MM-DD UTC")
    parser.add_argument("--saida", type=Path, help="Salvar a comparação em JSON")
    args = parser.parse_args()
    emission = pd.Timestamp(args.data).normalize()
    selection = json.loads((RUN / "evaluation/selection.json").read_text(encoding="utf-8"))
    variant = selection["variant"]
    rows = pd.read_parquet(RUN / "evaluation" / f"predictions_test_{variant}.parquet")
    # A emissão do modelo ocorre no dia anterior ao início do composto.
    chosen = rows.loc[(rows.latitude == args.latitude) & (rows.longitude == args.longitude)
                      & (rows.period_start == emission + pd.Timedelta(days=1))]
    if len(chosen) != 1:
        parser.error("Não há previsão de teste salva para essa coordenada e data. Consulte o README para outros locais.")
    row = chosen.iloc[0]
    result = {
        "tipo": "simulação retrospectiva com dados reais de teste",
        "local": {"latitude": float(row.latitude), "longitude": float(row.longitude)},
        "emissao_simulada_utc": emission.date().isoformat(),
        "periodo": {"inicio": row.period_start.date().isoformat(),
                    "fim_exclusivo": row.period_end_exclusive.date().isoformat()},
        "evento": "ao menos um pixel MODIS/Terra com foco no composto de oito dias, condicionado à máscara integralmente válida",
        "probabilidade_condicional": float(row.probability_given_full_mask),
        "intervalo_80_parametros": [float(row.parameter_lower_80), float(row.parameter_upper_80)],
        "referencia_sazonal_condicional": float(row.seasonal),
        "observado": {"evento": bool(row.event), "pixels_com_foco": int(row.fire_pixels),
                       "pixels_validos": int(row.valid_pixels), "pixels_totais": int(row.total_pixels)},
        "modelo": variant,
        "fonte_observada": "NASA MOD14A2 via ORNL, arquivo do composto identificado pelo hash abaixo",
        "mascara_sha256": row.cache_sha256,
        "limitacoes": ["A simulação foi feita depois da data histórica.",
                       "A máscara futura válida não é garantida; a probabilidade incondicional é indisponível.",
                       "Um exemplo não mede calibração; não é chance de ignição ou de incêndio não observado.",
                       "Os arquivos históricos podem ter sido revisados após a emissão simulada."],
    }
    print(f"Local: {args.latitude}, {args.longitude} | emissão simulada: {args.data} UTC")
    print(f"Período: {result['periodo']['inicio']} a {row.period_end_exclusive.date() - pd.Timedelta(days=1)}")
    print(f"Probabilidade condicional de detecção MODIS: {100 * result['probabilidade_condicional']:.1f}%")
    print("Intervalo de incerteza dos parâmetros: "
          f"{100 * result['intervalo_80_parametros'][0]:.1f}% a {100 * result['intervalo_80_parametros'][1]:.1f}%")
    print(f"Observado: {'detecção' if row.event else 'nenhuma detecção no composto'}; "
          f"{row.fire_pixels} de {row.total_pixels} pixels com foco")
    print("Simulação retrospectiva condicionada à máscara válida; não representa chance de incêndio real.")
    if args.saida:
        args.saida.parent.mkdir(parents=True, exist_ok=True)
        args.saida.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print("Resultado salvo em:", args.saida)


if __name__ == "__main__":
    main()
