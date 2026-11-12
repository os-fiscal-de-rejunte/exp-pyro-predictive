"""Relatório local dos dados geográficos: amostras reais, mapas e JSON."""

from __future__ import annotations

import base64
import html
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

from .appeears import AppEEARSClient, AppEEARSError
from .geography import LAND_CLASSES, collect, parse_csv, sampling_grid, snapshot
from .io import read_json, utc_now, write_json
from .runtime import ResourceLimit

ROOT = Path(__file__).resolve().parents[2]


def describe_geography(
    latitude: float,
    longitude: float,
    reference_date: str,
    *,
    cache_dir: Path = Path("runs/geography_cache"),
    fetch: bool = True,
    config_path: Path = ROOT / "config/geography.json",
) -> dict:
    import pandas as pd

    config = read_json(config_path)
    points = sampling_grid(latitude, longitude)
    client = AppEEARSClient(
        cache_dir,
        max_requests=config["max_requests"],
        max_download_bytes=config["max_download_bytes"],
    )
    issued = pd.Timestamp(reference_date)
    issued = issued.tz_convert("UTC").tz_localize(None) if issued.tz else issued
    state = {"status": "cache_only"}
    paths = sorted((cache_dir / "downloads").glob("*/*.csv"))
    try:
        if fetch:
            last_observed = min(
                (issued - pd.Timedelta(days=1)).date(),
                pd.Timestamp.now(tz="UTC").date(),
            )
            state = collect(
                client, points, f"{issued.year-3}-01-01", str(last_observed), config
            )
            paths = sorted((cache_dir / "downloads").glob("*/*.csv"))
        if not fetch and any(
            not (cache_dir / ("layers_" + product + ".json")).exists()
            for product in config["products"]
        ):
            raise AppEEARSError("Metadados ausentes no cache; modo sem acesso à rede.")
        metadata = {product: client.metadata(product) for product in config["products"]}
        observations = parse_csv(paths, config, metadata, retrieved_at=utc_now())
    except (AppEEARSError, ResourceLimit) as exc:
        state = {
            "status": "resource_limit"
            if isinstance(exc, ResourceLimit)
            else "source_unavailable",
            "reason": str(exc),
        }
        observations = pd.DataFrame()
    if not observations.empty:
        # Usa a data efetiva de obtenção de cada arquivo para o contrato operacional.
        for path in paths:
            sidecar = path.with_suffix(".csv.json")
            if sidecar.exists():
                obtained = (
                    pd.Timestamp(read_json(sidecar)["retrieved_utc"])
                    .tz_convert("UTC")
                    .tz_localize(None)
                )
                observations.loc[
                    observations.source_file == path.name, "retrieved_at"
                ] = obtained
        observations.to_parquet(cache_dir / "observations.parquet", index=False)
    result = snapshot(observations, points, reference_date)
    for sample in result.get("samples", []):
        for entry in sample.get("layers", {}).values():
            entry["resolution_nominal_m"] = config["products"][entry["product"]][
                "resolution_m"
            ]
            entry["date_kind"] = "nominal_product_period_start"
            entry["spatial_support"] = "nearest_product_pixel"
    if result["status"] == "ok" and state["status"] == "pending":
        result["status"] = "partial"
    if result["status"] == "insufficient_data" and state["status"] not in (
        "ok",
        "cache_only",
    ):
        result["status"] = state["status"]
    result.update(
        schema_version="geographic-context-1.0",
        requested_location={"latitude": latitude, "longitude": longitude},
        collection=state,
        source="NASA AppEEARS",
        point_count=len(points),
        model_uses_geography=False,
        accuracy_effect="Ainda não medido; contexto geográfico não altera automaticamente o modelo v6.",
        collection_complete=state["status"] == "ok",
    )
    write_json(cache_dir / "last_context.json", result)
    return result


def export_view(context: dict, output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "geography.json", context)
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [p["longitude"], p["latitude"]],
                },
                "properties": {
                    k: v for k, v in p.items() if k not in ("latitude", "longitude")
                },
            }
            for p in context.get("samples", [])
        ],
    }
    write_json(output / "samples.geojson", geojson)
    features = context.get("features", {})
    cards = []
    for name, label, unit in [
        ("geo_ndvi", "NDVI médio", "adimensional"),
        ("geo_tree_percent", "Cobertura arbórea média", "%"),
        ("geo_elevation_m", "Altitude média", "m"),
        ("geo_elevation_range_m", "Amplitude de altitude amostrada", "m"),
    ]:
        value = features.get(name)
        text = f"{value:.3f} {unit}" if value is not None else "Sem dado disponível"
        cards.append(
            f"<article><h2>{html.escape(label)}</h2><p>{html.escape(text)}</p></article>"
        )
    figure = ""
    samples = context.get("samples", [])
    available = any(p.get("layers") for p in samples)
    if available:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
        for axis, layer, label, cmap in zip(
            axes,
            ["LC_Type1", "_250m_16_days_NDVI", "NASADEM_HGT"],
            ["Classe de cobertura do solo", "Vegetação · NDVI", "Altitude · metros"],
            ["tab20", "YlGn", "terrain"],
        ):
            selected = [p for p in samples if layer in p["layers"]]
            if selected:
                options = {}
                if layer == "LC_Type1":
                    cmap = ListedColormap(
                        [
                            "#12652b",
                            "#0b4d22",
                            "#4e8b3d",
                            "#77a94b",
                            "#33853b",
                            "#b49d58",
                            "#c9b67b",
                            "#8fbd53",
                            "#becb67",
                            "#d9d57b",
                            "#58a6a3",
                            "#edc86b",
                            "#d85b58",
                            "#b9a761",
                            "#e8f5ff",
                            "#c8beb0",
                            "#327fbb",
                        ]
                    )
                    options["norm"] = BoundaryNorm([v + 0.5 for v in range(18)], cmap.N)
                elif layer == "_250m_16_days_NDVI":
                    options.update(vmin=-0.2, vmax=1.0)
                image = axis.scatter(
                    [p["longitude"] for p in selected],
                    [p["latitude"] for p in selected],
                    c=[p["layers"][layer]["value"] for p in selected],
                    cmap=cmap,
                    s=170,
                    marker="s",
                    **options,
                )
                ticks = (
                    sorted({int(p["layers"][layer]["value"]) for p in selected})
                    if layer == "LC_Type1"
                    else None
                )
                fig.colorbar(image, ax=axis, shrink=0.7, ticks=ticks)
            else:
                axis.text(
                    0.5,
                    0.5,
                    "Sem dado utilizável",
                    transform=axis.transAxes,
                    ha="center",
                )
            axis.set(title=label, xlabel="Longitude", ylabel="Latitude")
            axis.ticklabel_format(useOffset=False)
        stream = io.BytesIO()
        fig.savefig(stream, format="png", dpi=140)
        fig.savefig(output / "geography.png", dpi=140)
        plt.close(fig)
        figure = (
            '<img alt="Dados nos pontos amostrados, sem interpolação" src="data:image/png;base64,'
            + base64.b64encode(stream.getvalue()).decode()
            + '">'
        )
    rows = []
    for point in samples:
        for layer, entry in point.get("layers", {}).items():
            label = (
                LAND_CLASSES.get(int(entry["value"]), str(entry["value"]))
                if layer == "LC_Type1"
                else f"{entry['value']:.4f} {entry['unit']}"
            )
            rows.append(
                "<tr>"
                + "".join(
                    "<td>" + html.escape(str(v)) + "</td>"
                    for v in [
                        point["id"],
                        layer,
                        label,
                        entry["date"],
                        entry["product"],
                    ]
                )
                + "</tr>"
            )
    loc = context.get("requested_location", {})
    page = f"""<!doctype html><html lang="pt-BR"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PYRO · contexto geográfico NASA</title>
<style>body{{font:16px system-ui,sans-serif;background:#f1f4f0;color:#182b23;margin:0}}main{{max-width:1200px;margin:40px auto;padding:24px}}h1{{font-size:32px}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px}}article{{padding:20px;background:white;border:1px solid #c9d4cd;border-radius:8px}}h2{{font-size:16px}}article p{{font-size:24px}}.notice{{padding:18px;background:#fff4d6;border-left:4px solid #9d6e1c}}img{{width:100%;height:auto;margin:24px 0}}table{{border-collapse:collapse;width:100%;background:white;font-size:13px}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #ddd}}.table{{overflow-x:auto}}a{{color:#075c3d}}</style>
<main><h1>PYRO · contexto geográfico NASA</h1><p>Local: {html.escape(str(loc))} · Referência: {html.escape(str(context.get('reference_date')))}</p>
<p class="notice">Estado: <strong>{html.escape(str(context['status']))}</strong>. Amostragem de pontos na célula, sem interpolação ou imagem contínua do terreno. Dados geográficos não aumentam a acurácia por simples consulta: é necessário treinar e avaliar o modelo com essas variáveis.</p>
<div class="cards">{''.join(cards)}</div>{figure}<p>Quando faltar autenticação, execute <code>scripts/configurar_appeears.py</code> no terminal e retome a consulta. Dados sem qualidade suficiente permanecem ausentes. As datas de disponibilidade histórica usam hipóteses de atraso, sem comprovação de publicação original.</p>
<p><a href="geography.json">Resultado JSON</a> · <a href="samples.geojson">Pontos GeoJSON</a></p>
<div class="table"><table><thead><tr><th>Amostra</th><th>Camada</th><th>Valor</th><th>Data do produto</th><th>Produto</th></tr></thead><tbody>{''.join(rows) or '<tr><td colspan="5">Nenhuma observação geográfica utilizável disponível nesta consulta.</td></tr>'}</tbody></table></div></main></html>"""
    path = output / "index.html"
    path.write_text(page, encoding="utf-8")
    return path
