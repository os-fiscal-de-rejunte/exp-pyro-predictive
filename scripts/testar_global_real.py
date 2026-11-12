"""Consulta retrospectiva com artefatos reais, sem rede e sem retreino."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthfire.global_pipeline import predict  # noqa: E402
from earthfire.io import read_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconstrói uma previsão global piloto com dados reais."
    )
    parser.add_argument("--run-dir", type=Path, default=ROOT / "artifacts/global_v5")
    parser.add_argument("--latitude", type=float)
    parser.add_argument("--longitude", type=float)
    parser.add_argument("--data")
    args = parser.parse_args()
    root = args.run_dir
    if not (root / "model_registry.json").exists():
        parser.error(
            "Artefatos não encontrados. Informe --run-dir com a execução treinada."
        )
    sample = root / "evaluation/example_prediction.json"
    if args.latitude is None or args.longitude is None or args.data is None:
        if not sample.exists():
            parser.error("Forneça coordenadas e data; não há exemplo elegível salvo.")
        example = read_json(sample)
        latitude = (
            example["requested_location"]["latitude"]
            if args.latitude is None
            else args.latitude
        )
        longitude = (
            example["requested_location"]["longitude"]
            if args.longitude is None
            else args.longitude
        )
        day = example["issued_at"][:10] if args.data is None else args.data
    else:
        latitude, longitude, day = args.latitude, args.longitude, args.data
    print(
        json.dumps(
            predict(latitude, longitude, day, root), ensure_ascii=False, indent=2
        )
    )


if __name__ == "__main__":
    main()
