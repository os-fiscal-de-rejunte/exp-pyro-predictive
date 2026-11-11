"""Reproduz uma previsão diária VIIRS retrospectiva e compara com a máscara real."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthfire.io import read_json, sha256_file, write_json  # noqa: E402
from earthfire.mask_model import posterior_probabilities  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Previsão diária VIIRS retrospectiva com dados reais")
    # Área reservada cujo grupo satisfez o critério mínimo na validação pré-teste.
    parser.add_argument("--latitude", type=float, default=-31.0)
    parser.add_argument("--longitude", type=float, default=-52.5)
    parser.add_argument("--data", help="Dia previsto em AAAA-MM-DD; por padrão, último exemplo de teste elegível")
    parser.add_argument("--saida", type=Path)
    args = parser.parse_args()
    run = ROOT / "runs/viirs_daily_v4b_20261111"
    model_path = run / "models/mask_weather.pt"
    examples_path = run / "processed/mask_examples.parquet"
    prediction_path = run / "evaluation/viirs_predictions_test.parquet"
    if not model_path.exists() or not prediction_path.exists():
        parser.error("O experimento diário ainda não foi treinado e avaliado.")
    saved = torch.load(model_path, map_location="cpu", weights_only=False)
    summary = read_json(run / "processed/viirs_daily_summary.json")
    if (saved.get("model_version") != "vj114a1_daily_weather_v4b" or
            saved.get("daily_model_code_sha256") != sha256_file(ROOT / "src/earthfire/viirs_daily_model.py") or
            saved.get("data_sha256") != sha256_file(examples_path) or
            saved.get("preparation_manifest_sha256") != sha256_file(run / "processed/viirs_daily_summary.json") or
            summary["input_sha256"]["code"] != sha256_file(ROOT / "src/earthfire/viirs_daily_data.py")):
        parser.error("O checkpoint é incompatível com os dados ou o pré-processamento.")
    predictions = pd.read_parquet(prediction_path)
    chosen = predictions.loc[(predictions.latitude == args.latitude) & (predictions.longitude == args.longitude)]
    if args.data:
        chosen = chosen.loc[chosen.period_start == pd.Timestamp(args.data)]
    if chosen.empty:
        parser.error("Não há exemplo elegível salvo para essa área e data; falta de dados não significa ausência de foco.")
    row = chosen.sort_values("period_start").iloc[-1]
    gate = read_json(run / "evaluation/viirs_validation_before_test.json")["support"][row.spatial_group]
    result = {"schema_version": "1.0", "status": "insufficient_data", "tipo": "simulação retrospectiva",
        "local": {"latitude": float(row.latitude), "longitude": float(row.longitude)},
        "emissao_simulada_utc": (row.period_start - pd.Timedelta(days=1)).date().isoformat(),
        "dia_previsto_utc": row.period_start.date().isoformat(),
        "evento": saved["target"], "event_probability": None, "probabilidade_condicional": None,
        "validacao_do_grupo": gate, "modelo": saved["model_version"],
        "modelo_sha256": sha256_file(model_path), "historico_meteorologico_ate": row.weather_last_date.date().isoformat(),
        "observado": {"evento": bool(row.event), "pixels_com_foco": int(row.fire_pixels),
                       "pixels_validos": int(row.valid_pixels), "pixels_totais": int(row.total_pixels)},
        "limitações": ["A inferência foi executada após a data histórica; arquivos podem ter sido revisados.",
                       "A probabilidade exige máscara futura válida e não é chance de incêndio ou ignição.",
                       "O produto diário de 1 km não equivale às detecções FIRMS de 375 m."]}
    if gate["eligible"]:
        draws = posterior_probabilities(model_path, pd.DataFrame([row]), n_samples=400, seed=20261111)[:, 0]
        result.update({"status": "experimental_conditional", "probabilidade_condicional": float(draws.mean()),
                       "intervalo_80_parametros": np.quantile(draws, [.1, .9]).tolist()})
    else:
        result["motivo"] = "O grupo não atingiu o critério mínimo de validação pré-definido."
    print(f"Local: {args.latitude}, {args.longitude} | dia previsto: {result['dia_previsto_utc']}")
    if result["probabilidade_condicional"] is not None:
        print(f"Probabilidade condicional de detecção VIIRS 1 km: {100 * result['probabilidade_condicional']:.1f}%")
        print("Intervalo de 80% sobre parâmetros: "
              f"{100 * result['intervalo_80_parametros'][0]:.1f}% a {100 * result['intervalo_80_parametros'][1]:.1f}%")
    else:
        print("Estimativa indisponível:", result["motivo"])
    print(f"Observado no composto: {result['observado']['pixels_com_foco']} pixels com foco em "
          f"{result['observado']['pixels_validos']} pixels classificados.")
    print("Resultado retrospectivo condicionado à máscara válida; não é chance de incêndio real.")
    if args.saida:
        write_json(args.saida, result)
        print("Resultado salvo em:", args.saida)


if __name__ == "__main__":
    main()
