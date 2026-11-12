"""Coleta um piloto continental AppEEARS e compara Pyro com/sem geografia."""
# ruff: noqa: E402

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import pandas as pd
from earthfire.appeears import AppEEARSClient
from earthfire.geography import collect, parse_csv, sampling_grid, snapshot
from earthfire.io import read_json, sha256_file, utc_now, write_json
from earthfire.runtime import Run, ResourceLimit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, default=ROOT / "runs/geography_pilot_v1"
    )
    parser.add_argument(
        "--source", type=Path, default=ROOT / "artifacts/maxcoverage_v6"
    )
    parser.add_argument("--config", type=Path, default=ROOT / "config/geography.json")
    parser.add_argument(
        "--per-group",
        type=int,
        default=2,
        help="Células por continente e grupo espacial, escolhidas sem métricas do teste",
    )
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Consulta estados uma vez e devolve pending sem espera",
    )
    parser.add_argument(
        "--budget-id",
        default="geography_app_v1_20261112",
        help="Identificador explícito do orçamento; mantenha-o para retomar a mesma janela",
    )
    args = parser.parse_args()
    output, source = args.run_dir, args.source
    if output.resolve().parent != (ROOT / "runs").resolve():
        raise ValueError("Use um diretório de execução diretamente dentro de runs/.")
    os.environ["EARTHFIRE_BUDGET_ID"] = args.budget_id
    run = Run(ROOT / "config/geography_budget.json", output.name, ROOT)
    output.mkdir(parents=True, exist_ok=True)
    config = read_json(args.config)
    if not 1 <= args.per_group <= 10:
        parser.error("--per-group deve estar entre 1 e 10")
    catalog = read_json(source / "cell_catalog.json")
    plan_path = output / "plan.json"
    if plan_path.exists():
        plan = read_json(plan_path)
        if plan["config"] != config or plan["per_group"] != args.per_group:
            raise ValueError("Plano mudou; use outro diretório.")
    else:
        selected = []
        for continent in sorted({v["continent"] for v in catalog.values()}):
            for group in ("train", "holdout"):
                candidates = [
                    (k, v)
                    for k, v in sorted(catalog.items())
                    if v.get("feature_file")
                    and v["continent"] == continent
                    and v["spatial_group"] == group
                    and v["training_n"] >= 60
                    and 10 <= v["training_positive"] <= v["training_n"] - 10
                ]
                selected.extend(k for k, _ in candidates[: args.per_group])
        plan = {
            "created_utc": utc_now(),
            "selection": "ordem de identificador nativo por continente/grupo; suporte apenas do treino; nenhuma métrica do teste",
            "cells": selected,
            "config": config,
            "per_group": args.per_group,
            "source_registry_sha256": sha256_file(source / "model_registry.json"),
        }
        write_json(plan_path, plan)
    if plan["source_registry_sha256"] != sha256_file(source / "model_registry.json"):
        raise ValueError("Dados de origem mudaram.")
    metrics_path = output / "models/metrics.json"
    if metrics_path.exists():
        registry = read_json(output / "models/registry.json")
        for filename, digest in registry["artifacts"].items():
            if sha256_file(output / "models" / filename) != digest:
                raise ValueError("Modelo geográfico concluído foi alterado.")
        if (
            sha256_file(output / "models/examples.parquet")
            != registry["examples_sha256"]
        ):
            raise ValueError("Exemplos do modelo concluído foram alterados.")
        for module, digest in registry["code"].items():
            if sha256_file(ROOT / "src/earthfire" / (module + ".py")) != digest:
                raise ValueError("Código incompatível com o modelo concluído.")
        print(
            {"stage": "complete", "cached": True, "metrics": read_json(metrics_path)},
            flush=True,
        )
        return
    points = [
        p
        for rid in plan["cells"]
        for p in sampling_grid(catalog[rid]["latitude"], catalog[rid]["longitude"])
    ]
    write_json(output / "points.json", points)
    client = AppEEARSClient(
        output / "cache",
        max_requests=config["max_requests"],
        max_download_bytes=config["max_download_bytes"],
        guard=run.guard,
    )
    collect_config = {**config, "wait_seconds": 0} if args.no_wait else config
    result = collect(client, points, "2018-01-01", "2025-12-31", collect_config)
    print(
        {
            "stage": "collection",
            "status": result["status"],
            "cells": len(plan["cells"]),
            "points": len(points),
        },
        flush=True,
    )
    write_json(output / "status.json", result)
    if result["status"] in (
        "authentication_required",
        "source_unavailable",
        "resource_limit",
    ):
        raise SystemExit(2)
    if args.collect_only or result["status"] != "ok":
        return
    metadata = {p: client.metadata(p) for p in config["products"]}
    paths = [Path(p) for p in result["files"]]
    observations = parse_csv(paths, config, metadata, retrieved_at=utc_now())
    if observations.empty:
        raise ValueError(
            "Não houve observações AppEEARS utilizáveis; treinamento não iniciado."
        )
    observations.to_parquet(output / "observations.parquet", index=False)
    import importlib.metadata

    write_json(
        output / "environment.json",
        {
            name: importlib.metadata.version(name)
            for name in (
                "numpy",
                "pandas",
                "pyarrow",
                "torch",
                "pyro-ppl",
                "requests",
                "scipy",
            )
        },
    )
    write_json(
        output / "input_manifest.json",
        {
            "created_utc": utc_now(),
            "source_registry_sha256": plan["source_registry_sha256"],
            "observations_sha256": sha256_file(output / "observations.parquet"),
            "metadata": {
                product: sha256_file(client.cache / ("layers_" + product + ".json"))
                for product in config["products"]
            },
            "raw_csv": {
                p.relative_to(client.cache).as_posix(): sha256_file(p) for p in paths
            },
        },
    )
    registry = read_json(source / "model_registry.json")
    tables = []
    for filename in sorted({catalog[rid]["feature_file"] for rid in plan["cells"]}):
        path = source / filename
        if sha256_file(path) != registry["features"][filename]:
            raise ValueError("Features de origem alteradas.")
        frame = pd.read_parquet(path)
        tables.append(frame.loc[frame.region_id.isin(plan["cells"])])
    frame = pd.concat(tables, ignore_index=True)
    enriched = []
    for rid, part in frame.groupby("region_id"):
        group_points = [p for p in points if p["region_id"] == rid]
        data = observations.loc[
            observations.point_id.isin([p["id"] for p in group_points])
        ]
        for row in part.to_dict("records"):
            context = snapshot(data, group_points, row["period_start"].isoformat())
            row.update(context["features"])
            enriched.append(row)
    dataset = pd.DataFrame(enriched)
    from earthfire.geo_model import GEO_FEATURES, experiment

    for column in GEO_FEATURES + ["geo_landcover_sample_n", "geo_ndvi_sample_n"]:
        if column not in dataset:
            dataset[column] = float("nan")
    dataset.to_parquet(output / "joined_examples.parquet", index=False)
    metrics = experiment(dataset, output / "models", config, run=run)
    write_json(
        output / "status.json",
        {"status": "complete", "finished_utc": utc_now(), "metrics": metrics},
    )
    print(metrics)


if __name__ == "__main__":
    try:
        main()
    except ResourceLimit as exc:
        print({"status": "resource_limit", "reason": str(exc)})
        raise SystemExit(2) from None
