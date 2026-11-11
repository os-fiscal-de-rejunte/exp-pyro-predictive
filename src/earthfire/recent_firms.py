"""Prepara um retrato NRT FIRMS e consulta qualquer célula por coordenadas."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .ingest import grid_indices, parse_datetime, region_id
from .io import read_json, sha256_file, write_json
from .nasa_complement import download_firms
from .nasa_quality import firms_validation_masks
from .power import EXPECTED_UNITS, NasaPowerClient, WeatherUnavailable, grid_center, power_frame
from .runtime import Run


def normalize(frame: pd.DataFrame, product: str) -> tuple[pd.DataFrame, dict[str, int]]:
    data = frame.copy()
    masks = firms_validation_masks(data, product)
    _, acquired = parse_datetime(data)
    coordinates = ~(masks["latitude_invalid"] | masks["longitude_invalid"])
    valid = coordinates & acquired.notna()
    report = {"input_rows": len(data), "invalid_coordinates": int((~coordinates).sum()),
              "invalid_datetime": int(acquired.isna().sum()), "retained": int(valid.sum())}
    for key, mask in masks.items():
        report[key] = int(mask.sum())
    data = data.loc[valid].copy()
    data["latitude"] = pd.to_numeric(data.latitude)
    data["longitude"] = pd.to_numeric(data.longitude)
    data["satellite_raw"] = data.satellite
    if product == "modis":
        data["satellite"] = data.satellite.replace({"T": "Terra", "A": "Aqua"})
    data["acquired_utc"] = acquired.loc[valid]
    data["product"] = product
    ilat, ilon = grid_indices(data.latitude, data.longitude)
    data["region_id"] = [region_id(i * .5 - 90., j * .625 - 180.) for i,j in zip(ilat, ilon)]
    return data, report


def prepare(run: Run, source_run: str | None = None) -> dict[str, Any]:
    source = run.root / "runs" / (source_run or run.config["firms_run"]) / "firms_raw"
    manifest = read_json(source / "manifest.json")
    output = run.path / "recent"
    output.mkdir(parents=True, exist_ok=True)
    reports, frames = {}, []
    for filename, product in (("noaa20_south_america_7d.csv", "viirs-jpss1"),
                              ("modis_south_america_7d.csv", "modis")):
        run.guard()
        path = source / filename
        digest = sha256_file(path)
        if digest != manifest["files"][filename]["sha256"]:
            raise ValueError("CSV NRT alterado desde a aquisição.")
        frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        normalized, report = normalize(frame, product)
        normalized["source_file"] = filename
        normalized["source_sha256"] = digest
        normalized["source_row"] = normalized.index + 2
        reports[filename] = {**report, "source_sha256": digest,
            "retrieved_utc": manifest["files"][filename]["retrieved_utc"],
            "date_min": str(normalized.acquired_utc.min()), "date_max": str(normalized.acquired_utc.max())}
        frames.append(normalized)
    combined = pd.concat(frames, ignore_index=True)
    run.guard()
    source_cols = [x for x in combined.columns if x not in {"source_file", "source_sha256", "source_row", "region_id", "acquired_utc"}]
    duplicate = combined.duplicated(source_cols)
    combined = combined.loc[~duplicate].copy()
    combined.to_parquet(output / "observations.parquet", index=False)
    summary = {"scope": "arquivos FIRMS NRT South America de sete dias; cobertura desconhecida",
               "files": reports, "exact_duplicates_removed": int(duplicate.sum()), "retained": len(combined),
               "output_sha256": sha256_file(output / "observations.parquet"),
               "limitation": "Não equivale ao arquivo anual de 2026 nem demonstra observação sem detecção."}
    write_json(output / "summary.json", summary)
    run.stage("recent_prepare", "complete", retained=len(combined))
    return summary


def lookup(run: Run, latitude: float, longitude: float) -> dict[str, Any]:
    lat, lon = grid_center(latitude, longitude)
    rid = region_id(lat, lon)
    path = run.path / "recent/observations.parquet"
    data = pd.read_parquet(path)
    chosen = data.loc[data.region_id == rid]
    products = {}
    for product, satellite in (("viirs-jpss1", "N20"), ("modis", "Terra"), ("modis", "Aqua")):
        selected = chosen.loc[(chosen["product"] == product) & (chosen.satellite == satellite)]
        key = f"{product}/{satellite}"
        products[key] = {"records": len(selected) if len(selected) else None,
            "state": "positive_records" if len(selected) else "missing_unknown",
            "first_acquisition_utc": selected.acquired_utc.min().isoformat() if len(selected) else None,
            "last_acquisition_utc": selected.acquired_utc.max().isoformat() if len(selected) else None,
            "versions": sorted(selected.version.unique().tolist())}
    west, east = ((lon - .3125 + 180) % 360) - 180, ((lon + .3125 + 180) % 360) - 180
    return {"schema_version": "1.0", "status": "location_recognized", "requested": {"latitude": latitude, "longitude": longitude},
            "analysis_cell": {"region_id": rid, "center_latitude": lat, "center_longitude": lon,
                "bounds": [west, max(-90., lat - .25), east, min(90., lat + .25)], "crosses_antimeridian": west > east},
            "products": products, "event_probability": None,
            "limitation": "Registros NRT de South America, sem prova de cobertura; célula reconhecida não implica previsão calibrada."}


def add_weather_context(run: Run, result: dict[str, Any], latitude: float, longitude: float,
                        end: str | None = None) -> dict[str, Any]:
    finish = datetime.strptime(end, "%Y-%m-%d").date() if end else datetime.now(timezone.utc).date() - timedelta(days=3)
    if finish > datetime.now(timezone.utc).date() - timedelta(days=2):
        raise ValueError("O histórico POWER deve terminar pelo menos dois dias antes de hoje.")
    begin = finish - timedelta(days=7)
    params = ("T2M", "RH2M", "PRECTOTCORR", "WS10M")
    try:
        run.guard()
        client = NasaPowerClient(run.root / "runs/power_location_cache_v3", max_requests=run.config["operational"]["nasa_power_requests"])
        envelope = client.fetch(latitude, longitude, begin.isoformat(), finish.isoformat(), parameters=params)
        frame = power_frame(envelope)
        units = envelope["response"].get("parameters", {})
        if any(units.get(name, {}).get("units") != EXPECTED_UNITS[name] for name in params):
            raise WeatherUnavailable("Unidades POWER não reconhecidas.")
        if len(frame) != 8 or frame[list(params)].isna().any(axis=None):
            raise WeatherUnavailable("Dados POWER incompletos no intervalo solicitado.")
        result["weather_history"] = {"status": "ok", "source": "NASA POWER Daily UTC", "start": begin.isoformat(),
            "end": finish.isoformat(), "retrieved_utc": envelope["retrieved_utc"],
            "temperature_mean_c": float(frame.T2M.mean()), "relative_humidity_mean_percent": float(frame.RH2M.mean()),
            "precipitation_sum_mm": float(frame.PRECTOTCORR.sum()), "wind_mean_m_s": float(frame.WS10M.mean()),
            "limitation": "Contexto meteorológico histórico da célula POWER; não é previsão futura."}
    except WeatherUnavailable as exc:
        result["weather_history"] = {"status": "weather_unavailable", "reason": str(exc)}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/nasa_complement_v3b.json"))
    parser.add_argument("--run-id", default="recent_nasa_v3_20261111")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    sub.add_parser("refresh")
    p = sub.add_parser("lookup")
    p.add_argument("--latitude", type=float, required=True)
    p.add_argument("--longitude", type=float, required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--weather", action="store_true", help="Consulta histórico de oito dias NASA POWER")
    p.add_argument("--weather-end", help="Último dia histórico POWER, em AAAA-MM-DD")
    args = parser.parse_args()
    run = Run(args.config, args.run_id)
    if args.command == "prepare":
        result = prepare(run)
    elif args.command == "refresh":
        download_firms(run, ("noaa20_south_america_7d.csv", "modis_south_america_7d.csv"))
        result = prepare(run, source_run=run.run_id)
    else:
        result = lookup(run, args.latitude, args.longitude)
        if args.weather:
            result = add_weather_context(run, result, args.latitude, args.longitude, args.weather_end)
    if args.command == "lookup" and args.output:
        write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
