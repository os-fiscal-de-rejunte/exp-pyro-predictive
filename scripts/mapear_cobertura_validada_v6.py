"""Mapa científico da área validada: polígonos reais em vermelho, sem ampliar células."""

from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    data = ROOT / "artifacts/maxcoverage_v6"
    metrics = json.loads((data / "evaluation/metrics.json").read_text(encoding="utf-8"))
    coverage = json.loads((data / "coverage.geojson").read_text(encoding="utf-8"))
    countries = json.loads(
        (ROOT / "artifacts/global_v5/catalog/countries.geojson").read_text(
            encoding="utf-8"
        )
    )
    released = [
        f for f in coverage["features"] if f["properties"]["forecast_supported"]
    ]
    assert len(released) == metrics["released_cells"]
    area = sum(f["properties"]["area_km2"] for f in released)
    assert abs(area - metrics["released_area_km2"]) < 0.01
    land = []
    for feature in countries["features"]:
        geometry = feature["geometry"]
        polygons = (
            geometry["coordinates"]
            if geometry["type"] == "MultiPolygon"
            else [geometry["coordinates"]]
        )
        land.extend(p[0] for p in polygons)
    rings = [f["geometry"]["coordinates"][0] for f in released]
    fig = plt.figure(figsize=(16, 10), facecolor="white")
    grid = fig.add_gridspec(
        2,
        3,
        height_ratios=[3.3, 1.5],
        left=0.045,
        right=0.97,
        top=0.84,
        bottom=0.12,
        hspace=0.3,
        wspace=0.15,
    )

    def draw(ax, limits):
        ax.set_facecolor("#edf3f6")
        ax.add_collection(
            PolyCollection(
                land, facecolors="#d1dbe0", edgecolors="white", linewidths=0.35
            )
        )
        ax.add_collection(
            PolyCollection(rings, facecolors="#d71932", edgecolors="none")
        )
        ax.set(xlim=limits[:2], ylim=limits[2:])
        ax.tick_params(labelsize=8, length=0, colors="#677580")
        for spine in ax.spines.values():
            spine.set_visible(False)

    world = fig.add_subplot(grid[0, :])
    draw(world, (-180, 180, -90, 90))
    world.set_aspect("equal")
    for i, (label, limits) in enumerate(
        [
            ("Américas · ampliação", (-125, -30, -45, 55)),
            ("África e Europa · ampliação", (-20, 55, -40, 60)),
            ("Ásia e Oceania · ampliação", (65, 165, -45, 60)),
        ]
    ):
        axis = fig.add_subplot(grid[1, i])
        draw(axis, limits)
        axis.set_title(label, loc="left", fontsize=11)
    percent = f"{metrics['global_surface_percent']:.4f}".replace(".", ",")
    km2 = f"{area:,.0f}".replace(",", ".")
    fig.text(
        0.045,
        0.955,
        "PYRO · toda a cobertura validada em vermelho",
        fontsize=24,
        weight="bold",
    )
    fig.text(
        0.045,
        0.908,
        f"{percent}% do globo   •   {km2} km²   •   {len(released)} células",
        fontsize=19,
        color="#d71932",
        weight="bold",
    )
    fig.text(
        0.045,
        0.875,
        "Percentual da superfície total da Terra, incluindo oceanos. Área validada em algum período; disponibilidade varia por estação.",
        fontsize=10,
        color="#52636d",
    )
    fig.text(
        0.045,
        0.065,
        "Vermelho = área real das células validadas. Países inteiros não são considerados cobertos.",
        fontsize=11,
    )
    fig.text(
        0.045,
        0.04,
        "Referência: MODIS MOD14A2, compostos de até 8 dias; estimativas condicionadas à observação válida. Fonte: artefatos v6 do projeto.",
        fontsize=9,
        color="#52636d",
    )
    output = ROOT / "docs/figures"
    output.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        path = output / ("cobertura_validada_v6." + ext)
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
    print(
        {
            "validated_cells": len(released),
            "area_km2": area,
            "global_surface_percent": metrics["global_surface_percent"],
        }
    )


if __name__ == "__main__":
    main()
