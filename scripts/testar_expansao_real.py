"""Consulta v6 local, sem rede e sem retreinamento."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthfire.expansion_predict import predict
from earthfire.spatial_predict import forecast
from earthfire.io import read_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, default=ROOT / "artifacts/maxcoverage_v6"
    )
    parser.add_argument("--latitude", type=float)
    parser.add_argument("--longitude", type=float)
    parser.add_argument(
        "--geography",
        action="store_true",
        help="Consulta contexto geográfico AppEEARS; pode retornar pending",
    )
    parser.add_argument(
        "--geography-model",
        type=Path,
        help="Diretório opcional do experimento geográfico treinado",
    )
    parser.add_argument(
        "--geography-view", type=Path, help="Exporta relatório HTML, JSON e GeoJSON"
    )
    parser.add_argument(
        "--validated-only",
        action="store_true",
        help="Desabilita estimativas exploratórias fora do suporte local",
    )
    parser.add_argument(
        "--data",
        help="Data UTC; em modo histórico deve ser início de composto preparado",
    )
    parser.add_argument(
        "--historical",
        action="store_true",
        help="Reconstruir emissão histórica, em vez da próxima projeção sazonal",
    )
    args = parser.parse_args()
    if (args.latitude is None) != (args.longitude is None):
        parser.error("Forneça latitude e longitude juntas.")
    lat, lon, date = args.latitude, args.longitude, args.data
    if lat is None:
        name = "example_prediction.json" if args.historical else "example_forecast.json"
        path = args.run_dir / "evaluation" / name
        if not path.exists():
            parser.error("Sem exemplo validado; informe coordenadas e data.")
        sample = read_json(path)
        lat = sample["requested_location"]["latitude"]
        lon = sample["requested_location"]["longitude"]
        if args.historical and date is None:
            date = sample["issued_at"][:10]
    if args.historical and date is None:
        parser.error("Modo histórico exige --data.")
    if args.geography or args.geography_model or args.geography_view:
        if args.historical:
            parser.error(
                "Use geografia no modo de projeção; replay histórico v6 permanece congelado."
            )
        from earthfire.geographic_predict import forecast_with_geography

        value = forecast_with_geography(
            lat,
            lon,
            args.run_dir,
            date,
            view_dir=args.geography_view,
            geography_model_dir=args.geography_model,
            allow_exploratory=not args.validated_only,
        )
    else:
        value = (
            predict(lat, lon, date, args.run_dir)
            if args.historical
            else forecast(
                lat, lon, args.run_dir, date, allow_exploratory=not args.validated_only
            )
        )
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
