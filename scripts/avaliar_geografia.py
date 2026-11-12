"""Compara o experimento geográfico congelado com referências no mesmo recorte."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from earthfire.expansion_evaluate import summarize
from earthfire.io import read_json, sha256_file, write_json


def evaluate(root: Path) -> dict:
    registry = read_json(root / "registry.json")
    if sha256_file(root / "examples.parquet") != registry["examples_sha256"]:
        raise ValueError("Coorte alterada.")
    for filename, digest in registry["artifacts"].items():
        if sha256_file(root / filename) != digest:
            raise ValueError("Artefato alterado.")
    examples = pd.read_parquet(root / "examples.parquet")
    fit = examples.loc[
        (examples.split == "train") & (examples.spatial_group == "train")
    ]
    test = examples.loc[examples.split == "test"].copy()
    predictions = pd.read_parquet(root / "predictions.parquet")
    test = test.merge(
        predictions[
            ["region_id", "period_start", "with_geography", "without_geography"]
        ],
        on=["region_id", "period_start"],
        validate="one_to_one",
    )
    rate = (fit.event.sum() + 0.5) / (len(fit) + 1)
    monthly = fit.groupby(fit.period_start.dt.month).event.agg(["sum", "count"])
    rates = (monthly["sum"] + 0.5) / (monthly["count"] + 1)
    test["frequency"] = rate
    test["seasonal"] = test.period_start.dt.month.map(rates).fillna(rate)
    test["persistence"] = test.past_modis_event.fillna(rate)
    names = [
        "frequency",
        "seasonal",
        "persistence",
        "without_geography",
        "with_geography",
    ]
    report = {
        "interpretation": "Diagnóstico posterior à seleção já congelada; não altera a seleção. Referências ajustadas apenas no treino das regiões de ajuste.",
        "test": {name: summarize(test.event, test[name]) for name in names},
        "spatial_test": {
            name: summarize(
                test.loc[test.spatial_group == "holdout"].event,
                test.loc[test.spatial_group == "holdout", name],
            )
            for name in names
        },
    }
    write_json(root / "baseline_diagnostics.json", report)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    labels = ["Frequência", "Sazonal", "Persistência", "Pyro sem geo", "Pyro com geo"]
    for ax, group, title in zip(
        axes,
        ("test", "spatial_test"),
        ("Teste temporal completo", "Teste em regiões reservadas"),
    ):
        ax.bar(
            labels,
            [report[group][name]["brier"] for name in names],
            color=["#aaa"] * 3 + ["#4477aa", "#cc5533"],
        )
        ax.set_title(title)
        ax.set_ylabel("Brier (menor é melhor)")
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Recorte geográfico retrospectivo — 11 células, não global")
    fig.tight_layout()
    fig.savefig(root / "evaluation.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Referência ideal")
    for name, label in zip(names[-2:], labels[-2:]):
        points = []
        for interval in range(10):
            mask = np.minimum((test[name] * 10).astype(int), 9) == interval
            if mask.any():
                points.append(
                    (test.loc[mask, name].mean(), test.loc[mask, "event"].mean())
                )
        ax.plot(*np.array(points).T, marker="o", label=label)
    ax.set(
        xlabel="Probabilidade média prevista",
        ylabel="Frequência observada",
        xlim=(0, 1),
        ylim=(0, 1),
        title="Confiabilidade no teste retrospectivo",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "reliability.png", dpi=150)
    plt.close(fig)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args()
    print(evaluate(args.model_dir))
