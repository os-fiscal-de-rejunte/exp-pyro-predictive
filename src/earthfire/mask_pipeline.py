"""Etapas do experimento de oito dias com cobertura MODIS."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .mask_evaluate import evaluate, evaluate_baselines
from .mask_features import prepare
from .mask_model import train
from .runtime import Run


def main() -> None:
    parser = argparse.ArgumentParser(description="Modelo MOD14A2 com cobertura identificada")
    parser.add_argument("--config", type=Path, default=Path("config/nasa_complement_v3b.json"))
    parser.add_argument("--run-id", default="nasa_complement_v3b_20261111")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    sub.add_parser("baselines")
    t = sub.add_parser("train")
    t.add_argument("--steps", type=int, default=1000)
    e = sub.add_parser("evaluate")
    e.add_argument("--samples", type=int, default=400)
    p = sub.add_parser("predict")
    p.add_argument("--latitude", type=float, required=True)
    p.add_argument("--longitude", type=float, required=True)
    p.add_argument("--date", required=True, help="Data de emissão em UTC")
    p.add_argument("--output", type=Path)
    args = parser.parse_args()
    run = Run(args.config, args.run_id)
    if args.command == "prepare":
        result = prepare(run)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.command == "predict":
        from .io import write_json
        from .mask_location import predict_location
        result = predict_location(run, args.latitude, args.longitude, args.date)
        if args.output:
            write_json(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    examples = pd.read_parquet(run.path / "processed/mask_examples.parquet")
    if args.command == "baselines":
        print(json.dumps(evaluate_baselines(run, examples), ensure_ascii=False, indent=2))
    elif args.command == "train":
        paths = {variant: str(train(run, examples, variant == "weather", args.steps))
                 for variant in ("no_weather", "weather")}
        print(json.dumps(paths, ensure_ascii=False, indent=2))
    else:
        models = {variant: run.path / "models" / f"mask_{variant}.pt" for variant in ("no_weather", "weather")}
        result = evaluate(run, examples, models, args.samples)
        print(json.dumps({"selection": result["selection"], "test": {v: x["overall"] for v,x in result["test"].items()}},
                         ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
