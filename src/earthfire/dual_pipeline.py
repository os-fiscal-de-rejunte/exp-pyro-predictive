"""Execução do experimento com MODIS, VIIRS e NASA POWER Zarr."""
from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .dual_evaluate import evaluate
from .dual_features import build_examples
from .dual_model import train
from .harmonize import harmonize, matched_training_weeks
from .io import read_json, sha256_file, write_json
from .runtime import Run


def prepare_features(run: Run, satellite_run: str = "sat_zarr_v2_20261111") -> dict[str, Any]:
    source = run.root / run.config["runs_dir"] / satellite_run
    satellite_manifest = read_json(source / "run.json")
    if satellite_manifest["config_sha256"] != run.config_hash or satellite_manifest["stages"].get("prepare", {}).get("state") != "complete":
        raise ValueError("Preparação dos satélites ausente ou incompatível com a configuração.")
    weather_path = run.path / "weather_zarr/weekly.parquet"
    provenance_path = run.path / "weather_zarr/provenance.json"
    if not weather_path.exists() or not provenance_path.exists():
        raise ValueError("Recorte Zarr não está concluído.")
    output = run.path / "processed_v2"; output.mkdir(parents=True, exist_ok=True)
    existing = output / "features_provenance.json"
    if existing.exists():
        previous = read_json(existing)
        checks = [(output / "records.parquet", previous["records_sha256"]),
                  (output / "weekly_positive.parquet", previous["satellite_weekly_sha256"]),
                  (weather_path, previous["weather_weekly_sha256"]),
                  (output / "examples.parquet", previous["examples_sha256"]),
                  (output / "climatology.parquet", previous["climatology_sha256"]),
                  (run.root / run.config["selection_manifest"], previous["selection_sha256"])]
        if any(not path.exists() or sha256_file(path) != digest for path, digest in checks):
            raise ValueError("Insumos ou artefatos das features mudaram; inicie outro run_id.")
        return previous
    names = ("records.parquet", "weekly_positive.parquet", "calendar.parquet", "preparation.json")
    for name in names:
        src, dst = source / "processed" / name, output / name
        if dst.exists():
            if sha256_file(src) != sha256_file(dst):
                raise ValueError(f"Artefato existente incompatível: {dst}")
        else:
            shutil.copyfile(src, dst)
    weekly = pd.read_parquet(output / "weekly_positive.parquet")
    weather = pd.read_parquet(weather_path)
    examples, climate = build_examples(weekly, weather, run.config)
    examples.to_parquet(output / "examples.parquet", index=False)
    climate.to_parquet(output / "climatology.parquet", index=False)
    labels = examples.groupby(["split", "spatial_group"]).size().to_dict()
    summary = {"satellite_run": satellite_run, "records_sha256": sha256_file(output / "records.parquet"),
               "satellite_weekly_sha256": sha256_file(output / "weekly_positive.parquet"),
               "weather_weekly_sha256": sha256_file(weather_path), "examples_sha256": sha256_file(output / "examples.parquet"),
               "climatology_sha256": sha256_file(output / "climatology.parquet"),
               "selection_manifest": run.config["selection_manifest"], "selection_sha256": sha256_file(run.root / run.config["selection_manifest"]),
               "eligibility": "alvo VIIRS positivo e semana anterior positiva em VIIRS/NOAA-20 e MODIS/Terra; ausência não é zero",
               "split_counts": {f"{a}/{b}": int(n) for (a,b), n in labels.items()},
               "label_definition": "registros VIIRS/NOAA-20 por semana, incluindo tipos de anomalia do arquivo",
               "future_weather_train": "meteorologia observada retrospectiva para aprender resposta condicional",
               "future_weather_evaluation": "climatologia ajustada apenas em regiões e período de treino"}
    write_json(output / "features_provenance.json", summary)
    run.stage("features_v2", "complete", examples=len(examples), split_counts=summary["split_counts"])
    return summary


def fit_harmonization(run: Run) -> dict[str, Any]:
    out = run.path / "harmonization_v2"; out.mkdir(parents=True, exist_ok=True)
    saved = out / "report.json"
    source = run.path / "processed_v2/weekly_positive.parquet"
    data_hash = sha256_file(source)
    records_hash = sha256_file(run.path / "processed_v2/records.parquet")
    code_hash = sha256_file(Path(inspect.getfile(harmonize)))
    if saved.exists():
        previous = read_json(saved)
        if (previous.get("input_sha256") != data_hash or previous.get("records_sha256") != records_hash or
            previous.get("harmonization_code_sha256") != code_hash or previous.get("config_sha256") != run.config_hash):
            raise ValueError("Harmonização salva é incompatível com o conjunto atual.")
        return previous
    weekly = pd.read_parquet(source)
    train_regions = weekly.loc[~weekly.region_id.isin(run.config["spatial_holdout_regions"])].copy()
    run.claim_training("v2_harmonization")
    run.manifest["training_runs"] += 1; run.save()
    run.stage("harmonization_v2", "running")
    torch.set_num_threads(min(2, run.config["operational"]["max_cpu_threads"]))
    model = harmonize(train_regions, run.config["split"]["train_end"], steps=500, seed=run.config["seed"])
    model.update(config_sha256=run.config_hash, input_sha256=data_hash, records_sha256=records_hash,
                 harmonization_code_sha256=code_hash)
    model["note"] = "Contraste condicionado a semanas positivas em ambos; não estima cobertura nem atividade real."
    raw = pd.read_parquet(run.path / "processed_v2/records.parquet")
    raw["week_start"] = raw.acquired_utc.dt.tz_localize(None).dt.to_period("W-SUN").dt.start_time
    raw = raw.loc[(raw.week_start + pd.Timedelta(days=6) <= pd.Timestamp(run.config["split"]["train_end"])) &
                  (~raw.region_id.isin(run.config["spatial_holdout_regions"])) &
                  (((raw["product"] == "modis") & (raw.satellite == "Terra")) |
                   ((raw["product"] == "viirs-jpss1") & (raw.satellite == "N20")))].copy()
    model["quality_categories_train"] = {}
    for name, group in raw.groupby("product"):
        model["quality_categories_train"][name] = {"rows": len(group),
            "confidence": {str(k): int(v) for k, v in group.confidence.value_counts().items()},
            "type": {str(k): int(v) for k, v in group["type"].value_counts().items()},
            "daynight": {str(k): int(v) for k, v in group.daynight.value_counts().items()},
            "version": {str(k): int(v) for k, v in group.version.value_counts().items()}}
    by_day = raw.groupby(["region_id", "week_start", "daynight", "product"]).size().unstack("product")
    if {"modis", "viirs-jpss1"}.issubset(by_day.columns):
        by_day = by_day.dropna(subset=["modis", "viirs-jpss1"]).reset_index()
        by_day["difference"] = np.log1p(by_day["viirs-jpss1"]) - np.log1p(by_day["modis"])
        model["paired_daynight_train"] = {str(k): {"n": len(g), "median_log1p_difference": float(g.difference.median())}
                                         for k, g in by_day.groupby("daynight")}
    if model.get("status") == "exploratory":
        paired = matched_training_weeks(train_regions, run.config["split"]["validation_end"])
        paired = paired.loc[paired.week_start > pd.Timestamp(run.config["split"]["train_end"])].copy()
        if len(paired):
            region_index = {r: i for i, r in enumerate(model["regions"])}
            posterior = model["posterior"]
            coef = posterior["season"]["mean"]
            regions = posterior["region"]["mean"]
            intercept = posterior["intercept"]["mean"][0]
            phase = 2 * np.pi * paired.week_start.dt.dayofyear / 365.2425
            pred = np.array([intercept + regions[region_index[r]] if r in region_index else intercept for r in paired.region_id])
            pred += coef[0] * np.sin(phase.to_numpy()) + coef[1] * np.cos(phase.to_numpy())
            actual = np.log1p(paired.detections_viirs_n20) - np.log1p(paired.detections_modis_terra)
            train_pairs = matched_training_weeks(train_regions, run.config["split"]["train_end"])
            train_diff = np.log1p(train_pairs.detections_viirs_n20) - np.log1p(train_pairs.detections_modis_terra)
            reference = float(train_diff.median())
            model["validation"] = {"n": len(paired), "pyro_mae_log1p_difference": float(np.abs(actual-pred).mean()),
                                   "global_median_mae_log1p_difference": float(np.abs(actual-reference).mean()),
                                   "scope": "somente células de treino e semanas com os dois produtos positivos"}
    write_json(saved, model)
    run.stage("harmonization_v2", "complete", status=model["status"], matched_weeks=model["matched_weeks"])
    return model


def train_models(run: Run, steps: int) -> dict[str, Path]:
    examples = pd.read_parquet(run.path / "processed_v2/examples.parquet")
    return {name: train(run, examples, name == "weather", steps) for name in ("no_weather", "weather")}


def evaluate_harmonization(run: Run) -> dict[str, Any]:
    report = read_json(run.path / "harmonization_v2/report.json")
    if report.get("status") != "exploratory":
        return {"status": "insufficient_data", "reason": "Harmonização não ajustada."}
    weekly = pd.read_parquet(run.path / "processed_v2/weekly_positive.parquet")
    train_regions = weekly.loc[~weekly.region_id.isin(run.config["spatial_holdout_regions"])]
    fit = matched_training_weeks(train_regions, run.config["split"]["train_end"])
    fit_diff = np.log1p(fit.detections_viirs_n20) - np.log1p(fit.detections_modis_terra)
    baseline = float(fit_diff.median())
    pairs = matched_training_weeks(weekly, run.config["split"]["test_end"])
    pairs = pairs.loc[pairs.week_start >= pd.Timestamp(run.config["split"]["test_start"])].copy()
    if pairs.empty:
        return {"status": "insufficient_data", "reason": "Sem pares positivos no teste."}
    idx = {region: i for i, region in enumerate(report["regions"])}
    p = report["posterior"]
    phase = 2 * np.pi * pairs.week_start.dt.dayofyear.to_numpy() / 365.2425
    mean = np.array([p["intercept"]["mean"][0] + (p["region"]["mean"][idx[r]] if r in idx else 0.)
                     for r in pairs.region_id])
    mean += p["season"]["mean"][0] * np.sin(phase) + p["season"]["mean"][1] * np.cos(phase)
    pairs["actual_difference"] = np.log1p(pairs.detections_viirs_n20) - np.log1p(pairs.detections_modis_terra)
    pairs["posterior_mean_difference"] = mean
    pairs["baseline_difference"] = baseline
    pairs["spatial_group"] = np.where(pairs.region_id.isin(run.config["spatial_holdout_regions"]), "holdout", "train_regions")
    def metrics(g: pd.DataFrame) -> dict[str, Any]:
        return {"n": len(g), "pyro_mae_log1p_difference": float((g.actual_difference-g.posterior_mean_difference).abs().mean()),
                "global_median_mae_log1p_difference": float((g.actual_difference-g.baseline_difference).abs().mean())}
    result = {"status": "evaluated", "reference": "MODIS/Terra", "comparison": "VIIRS/NOAA-20",
              "condition": "ambos os sensores positivos na mesma célula e semana",
              "training_baseline": baseline, "overall": metrics(pairs),
              "by_spatial_group": {str(k): metrics(g) for k,g in pairs.groupby("spatial_group")},
              "limitation": "Comparação entre observações, não valida atividade real; versões MODIS mudam no teste."}
    out = run.path / "harmonization_v2"
    pairs.to_parquet(out / "test_pairs.parquet", index=False)
    write_json(out / "evaluation.json", result)
    run.stage("harmonization_v2_evaluation", "complete", n=len(pairs))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="PYRO: experimento de previsão com Zarr e dois sensores")
    parser.add_argument("--config", type=Path, default=Path("config/zarr_brazil_v2.json"))
    parser.add_argument("--run-id", default="zarr_brasil_v2_20261111")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--satellite-run", default="sat_zarr_v2_20261111")
    sub.add_parser("harmonize")
    sub.add_parser("evaluate-harmonization")
    t = sub.add_parser("train"); t.add_argument("--steps", type=int, default=900)
    e = sub.add_parser("evaluate"); e.add_argument("--samples", type=int, default=400)
    sub.add_parser("export")
    for name in ("predict", "scenario"):
        s = sub.add_parser(name)
        s.add_argument("--latitude", type=float, required=True); s.add_argument("--longitude", type=float, required=True)
        s.add_argument("--reference-date", required=True); s.add_argument("--variant", choices=["selected", "weather", "no_weather"], default="selected")
        s.add_argument("--overrides-file", type=Path); s.add_argument("--forecast-path", type=Path); s.add_argument("--output", type=Path)
    args = parser.parse_args()
    run = Run(args.config, args.run_id)
    versions = {}
    for name in ("numpy", "pandas", "pyarrow", "torch", "pyro-ppl", "requests", "scipy", "matplotlib", "psutil", "zarr", "numcodecs"):
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = None
    write_json(run.path / "versions.json", versions)
    if args.command == "prepare": print(prepare_features(run, args.satellite_run))
    elif args.command == "harmonize":
        result = fit_harmonization(run); print("harmonização", result["status"], "pares", result["matched_weeks"])
    elif args.command == "evaluate-harmonization": print(evaluate_harmonization(run))
    elif args.command == "train": print(train_models(run, args.steps))
    elif args.command == "evaluate":
        models = {x: run.path / "models_v2" / f"{x}.pt" for x in ("no_weather", "weather")}
        result = evaluate(run, pd.read_parquet(run.path / "processed_v2/examples.parquet"),
                          pd.read_parquet(run.path / "processed_v2/climatology.parquet"), models, args.samples)
        print("avaliação", result["selection"])
    elif args.command == "export":
        from .dual_export import export
        print(export(run))
    elif args.command in ("predict", "scenario"):
        import json
        from .dual_scenario import predict_week
        from .export import plot_scenario
        result = predict_week(args.latitude, args.longitude, args.reference_date, run.path,
                              read_json(args.overrides_file) if args.overrides_file else None,
                              args.forecast_path, args.variant)
        if args.output:
            write_json(args.output, result); plot_scenario(result, args.output.with_suffix(".png"))
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
