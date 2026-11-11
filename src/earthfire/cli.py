from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
from pathlib import Path

import pandas as pd

from .audit import audit
from .evaluate import evaluate
from .export import export, plot_scenario
from .features import build_examples, prepare_weather
from .harmonize import harmonize
from .io import read_json, sha256_file, write_json
from .model import train
from .power import NasaPowerClient
from .prepare import prepare
from .runtime import ResourceLimit, Run
from .reports import write_audit_report
from .scenario import simulate_fire_scenario


def _features(run: Run) -> None:
    source = pd.read_parquet(run.path / "processed/weekly_positive.parquet")
    write_json(run.path / "processed/harmonization.json", harmonize(source, run.config["split"]["train_end"]))
    client = NasaPowerClient(run.root / "runs/power_cache", max_requests=run.config["operational"]["nasa_power_requests"],
                             timeout=run.config["operational"]["network_timeout_seconds"], retries=run.config["operational"]["network_retries"])
    weather = prepare_weather(run, client)
    examples, climate = build_examples(source, weather, run.config)
    examples.to_parquet(run.path / "processed/examples.parquet", index=False)
    climate.to_parquet(run.path / "processed/climatology.parquet", index=False)
    run.stage("features", "complete", splits=examples.groupby("split").size().to_dict(), weather_lag_weeks=2)


def _verify_prepared(run: Run) -> bool:
    report = run.path / "processed/preparation.json"
    if not report.exists(): return False
    if not all((run.path / "processed" / name).exists() for name in ("records.parquet", "weekly_positive.parquet", "calendar.parquet")):
        return False
    info = read_json(report)
    inventory = info.get("archive_inventory")
    if not inventory: return False
    current = sorted(p.name for p in (run.root / run.config["data_dir"]).glob("*.zip"))
    if current != sorted(x["archive"] for x in inventory):
        raise ResourceLimit("Inventário de ZIPs mudou; escolha outro run_id.")
    for item in inventory:
        if sha256_file(run.root / run.config["data_dir"] / item["archive"]) != item["sha256"]:
            raise ResourceLimit("Um arquivo original mudou; escolha outro run_id.")
    return True


def _train(run: Run, steps: int) -> dict[str, Path]:
    examples = pd.read_parquet(run.path / "processed/examples.parquet")
    return {name: train(run, examples, include_weather=(name == "weather"), steps=steps) for name in ("no_weather", "weather")}


def _evaluate(run: Run, paths: dict[str, Path], samples: int) -> None:
    evaluate(run, pd.read_parquet(run.path / "processed/examples.parquet"),
             pd.read_parquet(run.path / "processed/climatology.parquet"), paths, n_samples=samples)


def main() -> None:
    parser = argparse.ArgumentParser(description="PYRO — processamento científico de detecções")
    parser.add_argument("--config", type=Path, default=Path("config/default.json"))
    parser.add_argument("--run-id", default="brasil_20261111")
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("audit"); a.add_argument("--mode", choices=["full", "brazil"], default="full")
    sub.add_parser("prepare"); sub.add_parser("features")
    t = sub.add_parser("train"); t.add_argument("--steps", type=int, default=900)
    e = sub.add_parser("evaluate"); e.add_argument("--samples", type=int, default=400)
    sub.add_parser("export")
    rep = sub.add_parser("audit-report"); rep.add_argument("--audit-run", default="full_20261111")
    q = sub.add_parser("quick"); q.add_argument("--source-run", required=True); q.add_argument("--steps", type=int, default=20)
    for command in ("predict", "scenario"):
        s = sub.add_parser(command); s.add_argument("--latitude", type=float, required=True); s.add_argument("--longitude", type=float, required=True)
        s.add_argument("--reference-date", required=True); s.add_argument("--horizon-days", type=int, default=7)
        s.add_argument("--overrides", default="{}"); s.add_argument("--overrides-file", type=Path)
        s.add_argument("--forecast-path"); s.add_argument("--model-version"); s.add_argument("--output", type=Path)
    args = parser.parse_args()
    run = Run(args.config, args.run_id)
    versions = {}
    for name in ("numpy", "pandas", "pyarrow", "torch", "pyro-ppl", "requests", "scipy", "matplotlib", "psutil"):
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = None
    write_json(run.path / "versions.json", versions)
    if args.command == "audit": print(audit(run, full=args.mode == "full"))
    elif args.command == "prepare":
        if not _verify_prepared(run): prepare(run)
        print(run.path / "processed/preparation.json")
    elif args.command == "features": _features(run); print(run.path / "processed/examples.parquet")
    elif args.command == "train": print(_train(run, args.steps))
    elif args.command == "evaluate":
        paths = {name: run.path / "models" / f"{name}.pt" for name in ("no_weather", "weather")}
        _evaluate(run, paths, args.samples); print(run.path / "evaluation/metrics.json")
    elif args.command == "export": export(run); print(run.path / "exports")
    elif args.command == "audit-report":
        write_audit_report(run.root / run.config["runs_dir"] / args.audit_run / "audit_full", run.root / "docs/auditoria.md")
        print(run.root / "docs/auditoria.md")
    elif args.command == "quick":
        if not run.run_id.startswith("quick_"):
            raise ValueError("O modo rápido exige run_id iniciado por quick_.")
        src = run.root / run.config["runs_dir"] / args.source_run / "processed"
        out = run.path / "processed"; out.mkdir(parents=True, exist_ok=True)
        for name in ("weekly_positive.parquet", "weather_weekly.parquet", "climatology.parquet", "calendar.parquet"):
            shutil.copyfile(src / name, out / name)
        if not (out / "examples.parquet").exists():
            examples = pd.read_parquet(src / "examples.parquet")
            training = examples.loc[examples.split == "train"].copy()
            training["year"] = training.week_start.dt.year
            keep = training.groupby(["region_id", "year"]).cumcount().mod(4).eq(0)
            training = training.loc[keep].drop(columns="year")
            examples = pd.concat([training, examples.loc[examples.split != "train"]]).sort_values("week_start")
            examples.to_parquet(out / "examples.parquet", index=False)
        run.manifest["mode"] = "demonstration_only"; run.save()
        _evaluate(run, _train(run, args.steps), samples=30)
        export(run)
        print("DEMONSTRATIVO:", run.path)
    elif args.command in ("predict", "scenario"):
        overrides = read_json(args.overrides_file) if args.overrides_file else json.loads(args.overrides)
        result = simulate_fire_scenario(args.latitude, args.longitude, args.reference_date, args.horizon_days,
            {"run_dir": str(run.path), "power_cache": str(run.root / "runs/power_cache")}, overrides, args.forecast_path, args.model_version)
        if args.output:
            write_json(args.output, result)
            plot_scenario(result, args.output.with_suffix(".png"))
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
