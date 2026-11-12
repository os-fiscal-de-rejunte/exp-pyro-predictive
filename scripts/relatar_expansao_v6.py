"""Métricas e figuras reais após o congelamento da política; não seleciona modelos."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch

from earthfire.expansion_evaluate import summarize
from earthfire.io import read_json, sha256_file, utc_now, write_json


def past_month_reference(group: pd.DataFrame) -> tuple[list[int], list[float]]:
    """Varredura cronológica: só adiciona rótulos já publicados e encerrados."""
    usable = group.loc[group.event.notna() & group.produced_at.notna()].copy()
    release = np.maximum(
        usable.period_end_exclusive.to_numpy(dtype="datetime64[ns]").astype(np.int64),
        usable.produced_at.to_numpy(dtype="datetime64[ns]").astype(np.int64) + 1,
    )
    order = np.argsort(release, kind="stable")
    release = release[order]
    months = usable.period_start.dt.month.to_numpy()[order]
    outcomes = usable.event.to_numpy(float)[order]
    counts, positives = np.zeros(13, int), np.zeros(13, float)
    cursor = 0
    reference_n, reference_rate = [], []
    for origin, month in zip(
        group.period_start.to_numpy(dtype="datetime64[ns]").astype(np.int64),
        group.period_start.dt.month,
    ):
        while cursor < len(release) and release[cursor] <= origin:
            counts[months[cursor]] += 1
            positives[months[cursor]] += outcomes[cursor]
            cursor += 1
        n = int(counts[month])
        reference_n.append(n)
        reference_rate.append(
            float((positives[month] + 0.5) / (n + 1)) if n >= 3 else np.nan
        )
    return reference_n, reference_rate


def calendar(root: Path) -> dict:
    """Calendário observado; a referência da anomalia exige publicação anterior."""
    registry = read_json(root / "model_registry.json")
    folder = root / "evaluation/calendar_parts"
    folder.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "native41-calendar-1.0",
        "time_standard": "UTC",
        "product": "MOD14A2",
        "event_unit": "evento binário por célula-composto; não incêndios independentes",
        "anomaly_unit": "evento observado menos frequência anterior do mesmo mês",
        "reference": "mesma célula e mês; fim do composto e produção anteriores à emissão; mínimo três alvos válidos",
        "parts": {},
    }
    for name, digest in registry["features"].items():
        source = root / name
        if sha256_file(source) != digest:
            raise ValueError("Features alteradas na exportação do calendário.")
        frame = pd.read_parquet(source)
        pieces = []
        for rid, group in frame.groupby("region_id"):
            group = group.sort_values("period_start").reset_index(drop=True)
            reference_n, reference_rate = past_month_reference(group)
            columns = [
                "region_id",
                "parent_id",
                "period_start",
                "period_end_exclusive",
                "produced_at",
                "event",
                "fire_pixels",
                "valid_pixels",
                "cloud_pixels",
                "total_pixels",
                "observation_state",
                "weather_complete",
            ]
            one = group[columns].copy()
            one["reference_n"] = reference_n
            one["reference_probability"] = reference_rate
            one["observed_anomaly"] = (
                one.event.astype(float) - one.reference_probability
            )
            one["anomaly_status"] = np.where(
                one.observed_anomaly.notna(), "ok", "insufficient_data"
            )
            one["valid_fraction"] = one.valid_pixels / one.total_pixels
            pieces.append(one)
        output = folder / source.name
        result = pd.concat(pieces, ignore_index=True)
        result.to_parquet(output, index=False, row_group_size=276)
        manifest["parts"][output.relative_to(root).as_posix()] = {
            "rows": len(result),
            "sha256": sha256_file(output),
            "source_sha256": digest,
            "anomalies_supported": int(result.observed_anomaly.notna().sum()),
        }
    write_json(root / "evaluation/calendar_manifest.json", manifest)
    return manifest


def calibration(frame: pd.DataFrame, column: str) -> list[dict]:
    bins = np.minimum((frame[column].to_numpy() * 10).astype(int), 9)
    result = []
    for index in range(10):
        one = frame.loc[bins == index]
        if len(one):
            result.append(
                {
                    "bin": index,
                    "n": len(one),
                    "mean_probability": float(one[column].mean()),
                    "observed_rate": float(one.event.mean()),
                }
            )
    return result


def cluster_intervals(frame: pd.DataFrame, draws: int = 1000) -> dict:
    """Reamostra blocos geográficos inteiros; não supõe linhas independentes."""
    rows = []
    for name, part in frame.groupby("parent_id"):
        p, y = part.routed_probability.to_numpy(), part.event.to_numpy()
        rows.append(
            [
                len(part),
                ((p - y) ** 2).sum(),
                ((p >= 0.5) == y).sum(),
                ((p >= 0.5) & (y == 1)).sum(),
                y.sum(),
            ]
        )
    if len(rows) < 2:
        return {"status": "insufficient_clusters", "clusters": len(rows)}
    values = np.asarray(rows, float)
    rng = np.random.default_rng(20261112)
    trials = []
    for _ in range(draws):
        n, loss, correct, tp, positives = values[
            rng.integers(len(values), size=len(values))
        ].sum(axis=0)
        trials.append([loss / n, correct / n, tp / positives if positives else np.nan])
    result = {
        "method": "bootstrap por bloco geográfico; não é intervalo da probabilidade local",
        "clusters": len(rows),
        "draws": draws,
    }
    for i, name in enumerate(["brier", "accuracy", "recall"]):
        result[name + "_95"] = np.nanquantile(
            np.asarray(trials)[:, i], [0.025, 0.975]
        ).tolist()
    return result


def world_map(root: Path, metrics: dict) -> None:
    countries = read_json(ROOT / "artifacts/global_v5/catalog/countries.geojson")
    shapes = []
    for feature in countries["features"]:
        geo = feature["geometry"]
        for polygon in (
            geo["coordinates"]
            if geo["type"] == "MultiPolygon"
            else [geo["coordinates"]]
        ):
            shapes.append(polygon[0])
    coverage = read_json(root / "coverage.geojson")
    current_path = root / "verification.json"
    current = (
        set(read_json(current_path).get("currently_projectable_cell_ids", []))
        if current_path.exists()
        else set()
    )
    grouped = {"history": [], "validated": [], "current": []}
    for feature in coverage["features"]:
        prop = feature["properties"]
        if not prop.get(
            "historical_information_supported", prop["historical_observed_periods"] > 0
        ):
            continue
        group = (
            "current"
            if prop["cell_id"] in current
            else "validated"
            if prop["forecast_supported"]
            else "history"
        )
        grouped[group].append(feature["geometry"]["coordinates"][0])
    colors = {"history": "#8d9ca7", "validated": "#f29457", "current": "#ce2137"}
    fig = plt.figure(figsize=(17, 10.5), facecolor="white")
    grid = fig.add_gridspec(2, 3, height_ratios=[3.6, 1.6], hspace=0.28, wspace=0.19)
    ax = fig.add_subplot(grid[0, :], facecolor="#edf2f5")

    def draw(axis, limits):
        axis.add_collection(
            PolyCollection(
                shapes, facecolors="#d4dce1", edgecolors="white", linewidths=0.25
            )
        )
        for group in ["history", "validated", "current"]:
            axis.add_collection(
                PolyCollection(
                    grouped[group],
                    facecolors=colors[group],
                    edgecolors=colors[group],
                    linewidths=0.15,
                )
            )
        axis.set(xlim=limits[:2], ylim=limits[2:])
        axis.tick_params(labelsize=8, length=0, colors="#64727d")
        for spine in axis.spines.values():
            spine.set_visible(False)

    draw(ax, (-180, 180, -60, 80))
    ax.set_aspect("equal")
    for i, (label, limits) in enumerate(
        [
            ("Américas", (-125, -30, -45, 55)),
            ("África e Europa", (-20, 55, -40, 55)),
            ("Ásia e Oceania", (65, 165, -45, 60)),
        ]
    ):
        local = fig.add_subplot(grid[1, i], facecolor="#edf2f5")
        draw(local, limits)
        local.set_title(label, fontsize=11, loc="left")
    fig.suptitle(
        "PYRO · cobertura medida da expansão v6",
        x=0.07,
        ha="left",
        fontsize=22,
        weight="bold",
    )
    fig.text(
        0.07,
        0.922,
        f"{metrics['released_cells']:,} células validadas · {metrics['released_area_km2']:,.0f} km² · {metrics['global_surface_percent']:.4f}% da superfície global",
        fontsize=12,
    )
    fig.legend(
        handles=[
            Patch(
                color=colors["current"],
                label="Projeção sazonal disponível nesta emissão",
            ),
            Patch(
                color=colors["validated"], label="Validada; sujeita a dados e estação"
            ),
            Patch(color=colors["history"], label="Somente histórico observado"),
        ],
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.045),
    )
    fig.text(
        0.07,
        0.025,
        "Polígonos na área real; sem preencher países. Alvo: detecção MOD14A2 em composto de até 8 dias, condicional à máscara válida.",
        fontsize=9,
        color="#64727d",
    )
    fig.subplots_adjust(left=0.07, right=0.96, top=0.89, bottom=0.13)
    for ext in ("png", "svg"):
        path = root / "evaluation" / ("coverage_map." + ext)
        fig.savefig(path, dpi=180)
        if ext == "svg":
            path.write_text(
                "\n".join(
                    line.rstrip()
                    for line in path.read_text(encoding="utf-8").splitlines()
                )
                + "\n",
                encoding="utf-8",
                newline="\n",
            )
    plt.close(fig)


def report(root: Path) -> dict:
    metrics = read_json(root / "evaluation/metrics.json")
    calendar_manifest = calendar(root)
    paths = sorted((root / "evaluation/test_parts").glob("*.parquet"))
    frame = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    eligible = frame.loc[frame.release_eligible].copy()
    if eligible.empty:
        result = {"status": "insufficient_data", "created_utc": utc_now()}
        write_json(root / "evaluation/diagnostics.json", result)
        world_map(root, metrics)
        return result
    result = {
        "created_utc": utc_now(),
        "status": "ok",
        "cohort": "test; release policy selected on validation only",
        "same_released_cohort": {
            name: summarize(eligible.event, eligible[name])
            for name in [
                "global_frequency",
                "seasonal",
                "regional_seasonal",
                "persistence",
                "routed_probability",
            ]
        },
        "calibration": calibration(eligible, "routed_probability"),
        "cluster_bootstrap": cluster_intervals(eligible),
        "by_month": {},
        "by_horizon_days": {},
        "by_parent": {},
        "coverage_by_continent": {},
        "full_test_baselines_by_continent": {},
        "calendar": {
            "rows": sum(v["rows"] for v in calendar_manifest["parts"].values()),
            "anomalies_supported": sum(
                v["anomalies_supported"] for v in calendar_manifest["parts"].values()
            ),
        },
    }
    eligible["month"] = eligible.period_start.dt.month
    following_year = pd.to_datetime(
        (eligible.period_start.dt.year + 1).astype(str) + "-01-01"
    )
    eligible["horizon_days"] = np.minimum(
        8, (following_year - eligible.period_start).dt.days
    )
    for column, target in [
        ("month", "by_month"),
        ("horizon_days", "by_horizon_days"),
        ("parent_id", "by_parent"),
    ]:
        for name, part in eligible.groupby(column):
            result[target][str(name)] = summarize(part.event, part.routed_probability)
    # Intervalos preditivos do evento binário; o intervalo de parâmetros não é testável contra um único evento.
    p, y = eligible.routed_probability.to_numpy(), eligible.event.to_numpy()
    from scipy.stats import bernoulli

    result["predictive_event_intervals"] = {}
    for level in (0.8, 0.95):
        low = bernoulli.ppf((1 - level) / 2, p)
        high = bernoulli.ppf((1 + level) / 2, p)
        result["predictive_event_intervals"][str(level)] = {
            "coverage": float(((y >= low) & (y <= high)).mean()),
            "mean_width": float((high - low).mean()),
            "unit": "evento binário; não contagens",
        }
    write_json(root / "evaluation/diagnostics.json", result)
    # JSON de tamanho previsível para integração; não exporta todos os exemplos em uma resposta.
    catalog = read_json(root / "cell_catalog.json")
    for continent in sorted({v["continent"] for v in catalog.values()}):
        cells = [v for v in catalog.values() if v["continent"] == continent]
        released = [v for v in cells if v["release"].get("eligible", False)]
        result["coverage_by_continent"][continent] = {
            "catalog_cells": len(cells),
            "cells_with_observed_training": sum(v["training_n"] > 0 for v in cells),
            "released_cells": len(released),
            "released_area_km2": sum(v["area_km2"] for v in released),
        }
    for continent, part in frame.groupby("continent"):
        result["full_test_baselines_by_continent"][str(continent)] = {
            name: summarize(part.event, part[name])
            for name in [
                "global_frequency",
                "seasonal",
                "regional_seasonal",
                "persistence",
            ]
        }
    write_json(root / "evaluation/diagnostics.json", result)
    export = [
        {
            "cell_id": rid,
            "latitude": cell["latitude"],
            "longitude": cell["longitude"],
            "area_km2": cell["area_km2"],
            "continent": cell["continent"],
            "forecast_supported": cell["release"].get("eligible", False),
            "static_projection_months": cell["release"].get(
                "static_projection_months", []
            ),
            "historical_observed_periods": cell["training_n"],
            "historical_positive_periods": cell["training_positive"],
        }
        for rid, cell in sorted(catalog.items())
    ]
    pd.DataFrame(export).to_parquet(root / "evaluation/cells.parquet", index=False)
    write_json(
        root / "evaluation/cells.json",
        {"schema_version": "native41-catalog-1.0", "cells": export},
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    scores = result["same_released_cohort"]
    labels = list(scores)
    axes[0].barh(
        labels,
        [scores[k]["brier"] for k in labels],
        color=["#81939c"] * (len(labels) - 1) + ["#ce2137"],
    )
    axes[0].set(
        xlabel="Brier score · menor é melhor",
        title="Mesmas células e períodos do teste",
    )
    bins = result["calibration"]
    axes[1].plot([0, 1], [0, 1], "--", color="#81939c")
    axes[1].scatter(
        [b["mean_probability"] for b in bins],
        [b["observed_rate"] for b in bins],
        s=[20 + np.sqrt(b["n"]) * 3 for b in bins],
        color="#ce2137",
    )
    axes[1].set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Probabilidade prevista média",
        ylabel="Frequência observada",
        title="Calibração no teste · rota validada",
    )
    fig.tight_layout()
    fig.savefig(root / "evaluation/comparison.png", dpi=160)
    plt.close(fig)
    world_map(root, metrics)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir", type=Path, default=ROOT / "runs/maxcoverage_v6b_20261111"
    )
    args = parser.parse_args()
    result = report(args.run_dir)
    print({"status": result["status"], "created_utc": result["created_utc"]})
