"""Amostras geográficas AppEEARS, qualidade por produto e alinhamento retrospectivo."""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .appeears import AppEEARSClient, AppEEARSError, AuthenticationRequired
from .expansion_grid import PIXEL, RADIUS, point_cell
from .io import sha256_file, utc_now, write_json

LAND_CLASSES = {
    1: "Floresta de coníferas perene",
    2: "Floresta de folhas largas perene",
    3: "Floresta de coníferas decídua",
    4: "Floresta de folhas largas decídua",
    5: "Floresta mista",
    6: "Vegetação arbustiva fechada",
    7: "Vegetação arbustiva aberta",
    8: "Savana arborizada",
    9: "Savana",
    10: "Campo/herbáceas",
    11: "Área úmida permanente",
    12: "Agricultura",
    13: "Área urbana",
    14: "Mosaico agricultura/vegetação",
    15: "Neve/gelo permanente",
    16: "Solo exposto",
    17: "Água",
}


def sampling_grid(latitude: float, longitude: float, *, side: int = 3) -> list[dict]:
    """Amostras na célula equivalente; não são um raster interpolado."""
    if side not in (1, 3, 5):
        raise ValueError("Grade amostral permitida: 1, 3 ou 5 pontos por lado.")
    cell = point_cell(latitude, longitude)
    points = []
    for i, fy in enumerate((np.arange(side) + 0.5) / side):
        for j, fx in enumerate((np.arange(side) + 0.5) / side):
            y = cell["yllcorner"] + fy * 41 * PIXEL
            x = cell["xllcorner"] + fx * 41 * PIXEL
            lat = math.degrees(y / RADIUS)
            lon = math.degrees(x / (RADIUS * math.cos(math.radians(lat))))
            if abs(lon) > 180 or abs(lat) >= 90:
                continue
            points.append(
                {
                    "id": f"{cell['region_id']}_s{i}{j}",
                    "region_id": cell["region_id"],
                    "latitude": lat,
                    "longitude": lon,
                    "row": i,
                    "column": j,
                }
            )
    return points


def tasks_for_points(
    points: list[dict], start: str, end: str, config: dict
) -> list[dict]:
    begin, finish = pd.Timestamp(start), pd.Timestamp(end)
    if (
        pd.isna(begin)
        or pd.isna(finish)
        or begin > finish
        or finish.date() > pd.Timestamp.now(tz="UTC").date()
    ):
        raise ValueError("Intervalo geográfico inválido; não é previsão futura.")
    if len({p["id"] for p in points}) != len(points):
        raise ValueError("Identificadores amostrais duplicados.")
    for point in points:
        point_cell(float(point["latitude"]), float(point["longitude"]))
    products = config["products"]
    out = []
    size = min(100, int(config["max_points_per_task"]))
    if size < 1:
        raise ValueError("Tamanho de lote inválido.")
    for offset in range(0, len(points), size):
        group = points[offset : offset + size]
        for static in (False, True):
            coordinates = [
                {k: p[k] for k in ("id", "latitude", "longitude")}
                for p in group
                if not static or -56 <= p["latitude"] <= 60
            ]
            if not coordinates:
                continue
            layers = [
                {"product": product, "layer": layer}
                for product, spec in products.items()
                if (spec["temporal"] == "static") == static
                for layer in spec["layers"]
            ]
            if not layers:
                continue
            dates = (
                [{"startDate": "02-11-2000", "endDate": "02-21-2000"}]
                if static
                else [
                    {
                        "startDate": begin.strftime("%m-%d-%Y"),
                        "endDate": finish.strftime("%m-%d-%Y"),
                    }
                ]
            )
            params = {"coordinates": coordinates, "layers": layers, "dates": dates}
            digest = hashlib.sha256(
                json.dumps(params, sort_keys=True).encode()
            ).hexdigest()[:16]
            out.append(
                {
                    "task_type": "point",
                    "task_name": "PYRO-geography-" + digest,
                    "params": params,
                }
            )
    return out


def collect(
    client: AppEEARSClient, points: list[dict], start: str, end: str, config: dict
) -> dict:
    """Retoma a mesma seleção; tarefas pendentes não são reenviadas."""
    for product, spec in config["products"].items():
        metadata = client.metadata(product)
        missing = set(spec["layers"]) - set(metadata)
        if missing:
            raise AppEEARSError(
                f"Camadas não disponíveis em {product}: {sorted(missing)}"
            )
        if any(metadata[layer].get("Available") is False for layer in spec["layers"]):
            raise AppEEARSError(
                "Produto/camada indisponível no catálogo AppEEARS: " + product
            )
    payloads = tasks_for_points(points, start, end, config)
    result = {
        "status": "ok",
        "source": "NASA AppEEARS",
        "updated_utc": utc_now(),
        "tasks": [],
        "files": [],
    }
    identifiers = []
    for payload in payloads:
        try:
            identifier = client.submit(payload)
            identifiers.append(identifier)
        except AuthenticationRequired:
            result.update(
                status="authentication_required",
                reason="Execute scripts/configurar_appeears.py no terminal local.",
            )
            break
        except AppEEARSError as exc:
            result.update(status="source_unavailable", reason=str(exc))
            break
    deadline = time.monotonic() + config["wait_seconds"]
    for identifier in identifiers:
        try:
            state = client.wait(
                identifier, max_seconds=max(0, deadline - time.monotonic())
            )
            result["tasks"].append({"task_id": identifier, "status": state["status"]})
            if state["status"] != "done":
                if result["status"] == "ok":
                    result["status"] = "pending"
                continue
            result["files"].extend(str(p) for p in client.download_csv(identifier))
        except AuthenticationRequired:
            result.update(
                status="authentication_required",
                reason="Execute scripts/configurar_appeears.py no terminal local.",
            )
            break
        except AppEEARSError as exc:
            result.update(status="source_unavailable", reason=str(exc))
            break
    write_json(client.cache / "collection.json", result)
    return result


def parse_csv(
    paths: list[Path], config: dict, metadata: dict, *, retrieved_at: str
) -> pd.DataFrame:
    """CSV de pontos já tem escala aplicada pelo AppEEARS; não escala duas vezes."""
    rows = []
    for path in paths:
        source_hash = sha256_file(path)
        frame = pd.read_csv(path, dtype={"ID": str, "id": str})
        lower = {str(c).lower(): c for c in frame}
        if not {"id", "date", "latitude", "longitude"}.issubset(lower):
            continue  # arquivos auxiliares de lookup de qualidade não são observações
        dates = pd.to_datetime(frame[lower["date"]], format="%Y-%m-%d", errors="coerce")
        for product, spec in config["products"].items():
            prefix = product.replace(".", "_") + "_"
            qa_layer = (
                "_250m_16_days_VI_Quality"
                if product.startswith("MOD13")
                else "QC"
                if product.startswith("MCD12")
                else "Quality"
                if product.startswith("MOD44")
                else None
            )
            qa_col = prefix + qa_layer if qa_layer else None
            for layer in spec["layers"]:
                column = prefix + layer
                info = metadata[product][layer]
                if column not in frame or info.get("IsQA"):
                    continue
                values = pd.to_numeric(frame[column], errors="coerce")
                fill = info.get("FillValue")
                valid = values.notna() & dates.notna()
                if isinstance(fill, (float, int)):
                    valid &= values != fill
                    # Alguns preenchimentos podem aparecer já multiplicados pela escala.
                    scale = info.get("ScaleFactor")
                    if isinstance(scale, (int, float)):
                        valid &= values != fill * scale
                low, high = info.get("ValidMin"), info.get("ValidMax")
                if isinstance(low, (float, int)):
                    valid &= values >= low
                if isinstance(high, (float, int)):
                    valid &= values <= high
                qa = (
                    pd.to_numeric(frame[qa_col], errors="coerce")
                    if qa_col in frame
                    else pd.Series(np.nan, index=frame.index)
                )
                quality = pd.Series(True, index=frame.index)
                if qa_layer:
                    quality = qa.notna() & (qa.fillna(-1) == 0)
                    if product.startswith("MOD13"):
                        clean_qa = qa.where(np.isfinite(qa)).fillna(3).astype(np.int64)
                        quality = qa.notna() & ((clean_qa & 3) == 0)
                    if product.startswith("MCD12"):
                        quality = qa.isin(
                            [0, 2]
                        )  # classificação produzida; água continua como água
                    if product.startswith("MOD44"):
                        cloud_col = prefix + "Cloud"
                        quality &= (
                            (pd.to_numeric(frame[cloud_col], errors="coerce") == 0)
                            if cloud_col in frame
                            else False
                        )
                for index in frame.index:
                    if pd.isna(dates[index]):
                        continue
                    date = dates[index]
                    temporal = spec["temporal"]
                    end = (
                        date + pd.DateOffset(years=1)
                        if temporal == "annual"
                        else date + pd.Timedelta(days=16)
                        if temporal == "16day"
                        else date
                    )
                    available = (
                        pd.Timestamp(spec["available_from_assumed"])
                        if temporal == "static"
                        else end
                        + pd.Timedelta(days=spec["availability_lag_days_assumed"])
                    )
                    lat, lon = (
                        float(frame.at[index, lower["latitude"]]),
                        float(frame.at[index, lower["longitude"]]),
                    )
                    point_cell(lat, lon)
                    rows.append(
                        {
                            "point_id": str(frame.at[index, lower["id"]]),
                            "latitude": lat,
                            "longitude": lon,
                            "product": product,
                            "layer": layer,
                            "date": date,
                            "period_end_exclusive": end,
                            "available_at_assumed": available,
                            "retrieved_at": pd.Timestamp(retrieved_at)
                            .tz_convert("UTC")
                            .tz_localize(None),
                            "availability_verified": False,
                            "value": float(values[index]) if valid[index] else np.nan,
                            "quality_usable": bool(valid[index] and quality[index]),
                            "quality_status": "conservative_filter"
                            if qa_layer
                            else "fill_and_range_only",
                            "unit": info.get("Units", ""),
                            "resolution_m": spec["resolution_m"],
                            "source_file": path.name,
                            "source_sha256": source_hash,
                        }
                    )
    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows)
    keys = ["point_id", "product", "layer", "date"]
    duplicates = result.duplicated(keys, keep=False)
    for _, group in result.loc[duplicates].groupby(keys):
        if len(group[["value", "quality_usable"]].drop_duplicates()) > 1:
            raise ValueError(
                "Sobreposição AppEEARS com valores conflitantes; selecione uma versão da extração."
            )
    return result.drop_duplicates(keys).sort_values(keys).reset_index(drop=True)


def snapshot(
    observations: pd.DataFrame,
    points: list[dict],
    reference_date: str,
    *,
    operational: bool = False,
) -> dict:
    """Agrega amostras disponíveis sem chamar pontos amostrados de raster completo."""
    issued = pd.Timestamp(reference_date)
    issued = issued.tz_convert("UTC").tz_localize(None) if issued.tz else issued
    point_ids = {p["id"] for p in points}
    if observations.empty:
        return {
            "status": "insufficient_data",
            "features": {},
            "samples": [],
            "reference_date": issued.isoformat(),
        }
    rows = observations.loc[
        observations.point_id.isin(point_ids)
        & observations.quality_usable
        & (observations.available_at_assumed < issued)
    ].copy()
    if operational:
        rows = rows.loc[rows.retrieved_at < issued]
    fresh = (rows["product"] != "MOD13Q1.061") | ((issued - rows["date"]).dt.days <= 90)
    rows = (
        rows.loc[fresh]
        .sort_values("date")
        .drop_duplicates(["point_id", "product", "layer"], keep="last")
    )
    features = {}
    mapping = {
        "_250m_16_days_NDVI": "geo_ndvi",
        "_250m_16_days_EVI": "geo_evi",
        "Percent_Tree_Cover": "geo_tree_percent",
        "Percent_NonTree_Vegetation": "geo_nontree_percent",
        "Percent_NonVegetated": "geo_bare_percent",
        "NASADEM_HGT": "geo_elevation_m",
    }
    for layer, feature in mapping.items():
        values = rows.loc[rows.layer == layer, "value"]
        features[feature] = float(values.mean()) if len(values) else None
        features[feature + "_sample_n"] = len(values)
    land = rows.loc[rows.layer == "LC_Type1", "value"]
    for code in range(1, 18):
        features[f"geo_landcover_{code:02d}"] = (
            float((land == code).mean()) if len(land) else None
        )
    features["geo_landcover_sample_n"] = len(land)
    elevation = rows.loc[rows.layer == "NASADEM_HGT", "value"]
    features["geo_elevation_range_m"] = (
        float(elevation.max() - elevation.min()) if len(elevation) >= 2 else None
    )
    samples = []
    for point in points:
        one = rows.loc[rows.point_id == point["id"]]
        sample = {
            **point,
            "layers": {
                str(r.layer): {
                    "value": r.value,
                    "date": str(r.date.date()),
                    "product": r.product,
                    "unit": r.unit,
                }
                for r in one.itertuples()
            },
        }
        samples.append(sample)
    return {
        "status": "ok" if len(rows) else "insufficient_data",
        "source": "NASA AppEEARS",
        "reference_date": issued.isoformat(),
        "availability_mode": "operational_cache"
        if operational
        else "retrospective_assumed_lags",
        "availability_verified": False,
        "sampling": "grade amostral; não representa todos os pixels da célula",
        "features": features,
        "samples": samples,
        "spatial_interpolation": False,
        "interpretation": "Índices de vegetação não medem diretamente combustível seco; amplitude de altitude não é declividade.",
    }
