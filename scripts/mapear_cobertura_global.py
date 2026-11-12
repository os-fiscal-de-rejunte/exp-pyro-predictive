"""Desenha a cobertura real do piloto; símbolos ampliados não representam área."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Polygon
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "artifacts/global_v5"
LABELS = {
    "brasil_cerrado": "Brasil · Centro-Oeste",
    "argentina_chaco": "Argentina · Chaco",
    "california": "EUA · Califórnia",
    "canada_boreal": "Canadá · Boreal",
    "portugal": "Portugal",
    "grecia": "Grécia",
    "angola": "Angola",
    "sudao": "Sudão",
    "india": "Índia",
    "tailandia": "Tailândia",
    "australia_norte": "Austrália · Norte",
    "australia_leste": "Austrália · Leste",
}


def main() -> None:
    countries = json.loads(
        (DATA / "catalog/countries.geojson").read_text(encoding="utf-8")
    )
    areas = json.loads((DATA / "processed/areas.geojson").read_text(encoding="utf-8"))
    policy = json.loads((DATA / "release_policy.json").read_text(encoding="utf-8"))
    config = json.loads((DATA / "config.json").read_text(encoding="utf-8"))
    indexed = {f["properties"]["region"]: f for f in areas["features"]}
    largest = max(
        abs((x - r["longitude"]) * 111.195 * math.cos(math.radians(r["latitude"])))
        for r in config["regions"]
        for x, y in indexed[r["name"]]["geometry"]["coordinates"][0]
    )
    extent = math.ceil((largest + 3) / 10) * 10
    red, pale, ink = "#c72135", "#f4cbd0", "#23313c"
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "text.color": ink, "axes.labelcolor": ink}
    )
    fig = plt.figure(figsize=(20, 14), facecolor="white")
    fig.text(0.04, 0.962, "Cobertura atual do PYRO", fontsize=28, weight="bold")
    fig.text(
        0.04,
        0.929,
        "12 recortes estudados  ·  5 com estimativa liberada  ·  7 experimentais",
        fontsize=15,
    )
    fig.text(
        0.97, 0.961, "PILOTO v5  /  2020–2025", ha="right", fontsize=12, color="#657580"
    )
    grid = fig.add_gridspec(
        3,
        6,
        height_ratios=[4.4, 1.35, 1.35],
        left=0.04,
        right=0.97,
        top=0.875,
        bottom=0.11,
        hspace=0.72,
        wspace=0.30,
    )
    ax = fig.add_subplot(grid[0, :], facecolor="#edf2f5")
    for feature in countries["features"]:
        geometry = feature["geometry"]
        polygons = (
            geometry["coordinates"]
            if geometry["type"] == "MultiPolygon"
            else [geometry["coordinates"]]
        )
        for poly in polygons:
            ax.add_patch(
                Polygon(poly[0], facecolor="#d5dde2", edgecolor="white", linewidth=0.35)
            )
    ax.set(xlim=(-180, 180), ylim=(-90, 90))
    ax.set_xticks(range(-180, 181, 60))
    ax.set_yticks(range(-60, 91, 30))
    ax.tick_params(labelsize=9, colors="#71808a", length=0)
    ax.grid(color="white", linewidth=0.6, alpha=0.7)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i, region in enumerate(config["regions"], 1):
        feature = indexed[region["name"]]
        eligible = policy["groups"][region["name"]]["eligible"]
        ring = feature["geometry"]["coordinates"][0]
        ax.add_patch(
            Polygon(
                ring, facecolor=red if eligible else pale, edgecolor=red, linewidth=0.5
            )
        )
        lon, lat = region["longitude"], region["latitude"]
        ax.scatter(
            lon,
            lat,
            s=260,
            facecolor=red if eligible else "white",
            edgecolor=red,
            linewidth=1.5,
            zorder=5,
        )
        ax.text(
            lon,
            lat,
            str(i),
            ha="center",
            va="center",
            fontsize=10,
            weight="bold",
            color="white" if eligible else red,
            zorder=6,
        )
        local = fig.add_subplot(
            grid[1 + (i - 1) // 6, (i - 1) % 6], facecolor="#f6f8f9"
        )
        # Aproximação tangente local apenas para desenhar o recorte em quilômetros.
        xy = [
            ((x - lon) * 111.195 * math.cos(math.radians(lat)), (y - lat) * 111.195)
            for x, y in ring
        ]
        local.add_patch(
            Polygon(
                xy,
                facecolor=red if eligible else pale,
                edgecolor=red,
                hatch=None if eligible else "////",
                linewidth=1.2,
            )
        )
        local.set(
            xlim=(-extent, extent),
            ylim=(-extent, extent),
            xticks=[-40, 0, 40],
            yticks=[-40, 0, 40],
        )
        local.set_aspect("equal")
        local.grid(color="#d6dfe5", linewidth=0.5, alpha=0.65)
        local.tick_params(labelsize=7, length=0, colors="#71808a")
        local.set_title(
            f"{i} · {LABELS[region['name']]}", fontsize=10, weight="bold", pad=8
        )
        local.text(
            0.5,
            -0.24,
            "Estimativa liberada" if eligible else "Experimental · consulta bloqueada",
            transform=local.transAxes,
            ha="center",
            fontsize=8.5,
            color=red if eligible else "#8c6870",
        )
        if (i - 1) % 6 == 0:
            local.set_ylabel("km", fontsize=8)
        for spine in local.spines.values():
            spine.set_color("#d6dfe5")
    legend = [
        Patch(
            facecolor=red, edgecolor=red, label="5 recortes com estimativa condicional"
        ),
        Patch(
            facecolor=pale,
            edgecolor=red,
            hatch="////",
            label="7 recortes experimentais: sem estimativa liberada",
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color=red,
            markerfacecolor="white",
            markersize=9,
            label="Números no mapa: localizadores ampliados",
        ),
    ]
    fig.legend(
        handles=legend,
        loc="upper left",
        bbox_to_anchor=(0.035, 0.913),
        ncol=3,
        frameon=False,
        fontsize=10,
    )
    fig.text(
        0.04,
        0.055,
        "Quadros: geometria real em km (~1.443 km² por recorte). Círculos no mapa: localizadores ampliados.",
        fontsize=11,
    )
    fig.text(
        0.04,
        0.032,
        "Estimativa histórica de detecção MODIS em até 8 dias, condicionada à observação válida. Datas sem suporte retornam dados insuficientes.",
        fontsize=10,
        color="#657580",
    )
    fig.text(
        0.97, 0.055, "Área liberada: ~7.217 km²", ha="right", fontsize=11, weight="bold"
    )
    fig.text(
        0.97,
        0.032,
        "~0,0014% da superfície do globo",
        ha="right",
        fontsize=10,
        color="#657580",
    )
    out = ROOT / "docs/figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "cobertura_global_v5.png", dpi=170, facecolor="white")
    fig.savefig(out / "cobertura_global_v5.svg", facecolor="white")
    plt.close(fig)
    print(out / "cobertura_global_v5.png")


if __name__ == "__main__":
    main()
