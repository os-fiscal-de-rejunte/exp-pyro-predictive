"""Baixa pequenos recortes temporais da retrospectiva NASA GEOS-S2S-3."""
from __future__ import annotations

import argparse
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import requests
import numpy as np
import pandas as pd

from .io import read_json, replace_with_retry, sha256_file, utc_now, write_json
from .ingest import region_id
from .runtime import ResourceLimit, Run

BASE = "https://data.nas.nasa.gov/geos-s2s-3/SubC"


def download_cycle(run: Run, initial_date: str, days: int = 8, ensemble: int = 1) -> dict[str, Any]:
    initial = date.fromisoformat(initial_date)
    if not 1 <= days <= 10 or not 1 <= ensemble <= 5:
        raise ValueError("Solicite 1–10 dias e membro de ensemble 1–5.")
    root = run.path / "s2s_hindcast"
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {"requests": 0, "files": {}, "temporary_files": []}
    session = requests.Session()
    stamp = initial.strftime("%Y%m%d")
    run.stage("s2s_download", "running", initial_date=initial_date, ensemble=ensemble)
    for offset in range(days):
        valid = initial + timedelta(days=offset)
        filename = f"{stamp}.sfc_tavg_3hr_glo_L720x361_sfc.dailymean.{valid:%Y%m%d}.t2mprectot.nc4"
        target = root / f"ens{ensemble}" / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        item = manifest["files"].get(filename)
        if target.exists():
            if not item or sha256_file(target) != item["sha256"]:
                raise ValueError("Arquivo de reforecast incompatível com o manifesto.")
            continue
        temp = target.with_name(target.name + ".part")
        if str(temp) not in manifest["temporary_files"]:
            manifest["temporary_files"].append(str(temp))
        url = f"{BASE}/{stamp}/ens{ensemble}/{filename}"
        error = "sem resposta"
        for attempt in range(min(3, run.config["operational"]["network_retries"])):
            run.guard()
            if manifest["requests"] >= 80:
                raise ResourceLimit("Limite local de requisições S2S atingido.")
            delay = 1. - (time.time() - manifest.get("last_request_epoch", 0))
            if delay > 0: time.sleep(delay)
            manifest["requests"] += 1
            manifest["last_request_epoch"] = time.time()
            write_json(manifest_path, manifest)
            try:
                with session.get(url, timeout=run.config["operational"]["network_timeout_seconds"], stream=True) as response:
                    if response.status_code == 429 or response.status_code >= 500:
                        error = f"HTTP {response.status_code}"
                        retry = response.headers.get("Retry-After", "")
                        if retry.isdigit() and int(retry) > 30:
                            raise ResourceLimit("S2S solicitou espera superior a 30 segundos.")
                        if attempt < 2: time.sleep(max(2 ** attempt, int(retry) if retry.isdigit() else 0))
                        continue
                    if 400 <= response.status_code < 500:
                        raise ValueError(f"Arquivo S2S indisponível: HTTP {response.status_code}")
                    response.raise_for_status()
                    expected = int(response.headers.get("Content-Length", "0") or "0")
                    if expected > 16 * 1024 * 1024:
                        raise ResourceLimit("Arquivo S2S excede o limite individual de 16 MiB.")
                    size = 0
                    with temp.open("wb") as output:
                        for piece in response.iter_content(chunk_size=128 * 1024):
                            size += len(piece)
                            if size > 16 * 1024 * 1024:
                                raise ResourceLimit("Arquivo S2S excedeu o limite individual de 16 MiB.")
                            output.write(piece)
                    if expected and size != expected:
                        raise IOError("Arquivo S2S incompleto.")
                    with temp.open("rb") as stream:
                        magic = stream.read(8)
                    if not (magic.startswith(b"\x89HDF\r\n\x1a\n") or magic.startswith(b"CDF\x01") or magic.startswith(b"CDF\x02")):
                        raise ValueError("Resposta S2S não é um arquivo NetCDF/HDF5 válido.")
                    digest = sha256_file(temp)
                    replace_with_retry(temp, target)
                    item = {"source": url, "initialized": initial_date, "valid_date": valid.isoformat(),
                            "ensemble": ensemble, "bytes": size, "sha256": digest, "retrieved_utc": utc_now(),
                            "last_modified": response.headers.get("Last-Modified")}
                    manifest["files"][filename] = item
                    write_json(manifest_path, manifest)
                    break
            except (requests.RequestException, OSError) as exc:
                error = f"{type(exc).__name__}: {exc}"
                if attempt < 2: time.sleep(2 ** attempt)
        else:
            raise ConnectionError(f"Download S2S não concluído: {error}")
    session.close()
    run.stage("s2s_download", "complete", initial_date=initial_date, days=days, ensemble=ensemble)
    return {"files": len(manifest["files"]), "requests": manifest["requests"], "directory": str(root),
            "limitation": "Reforecast produzido retrospectivamente; inicialização histórica não comprova disponibilidade naquela data."}


def extract_cycle(run: Run, initial_date: str, ensemble: int = 1) -> dict[str, Any]:
    from netCDF4 import Dataset, num2date

    root = run.path / "s2s_hindcast"
    manifest = read_json(root / "manifest.json")
    selected = [value for value in manifest["files"].values()
                if value["initialized"] == initial_date and value["ensemble"] == ensemble]
    if not selected:
        raise ValueError("Nenhum arquivo S2S compatível foi baixado.")
    records = []
    for item in sorted(selected, key=lambda value: value["valid_date"]):
        run.guard()
        path = root / f"ens{ensemble}" / Path(item["source"]).name
        if sha256_file(path) != item["sha256"]:
            raise ValueError("Arquivo S2S alterado após o download.")
        # A biblioteca NetCDF em algumas instalações Windows não abre caminhos
        # absolutos com caracteres fora do alfabeto ASCII do projeto pai.
        with Dataset(str(path.relative_to(run.root))) as dataset:
            latitudes = np.asarray(dataset.variables["lat"][:], dtype=float)
            longitudes = np.asarray(dataset.variables["lon"][:], dtype=float)
            t2m, precip, time_var = (dataset.variables[x] for x in ("T2M", "PRECTOTCORR", "time"))
            if t2m.units != "K" or precip.units != "kg m-2 s-1" or len(time_var) != 1:
                raise ValueError("Unidades ou dimensão temporal S2S inesperadas.")
            actual_time = num2date(time_var[:], time_var.units, calendar=getattr(time_var, "calendar", "standard"))[0]
            if actual_time.strftime("%Y-%m-%d") != item["valid_date"]:
                raise ValueError("Data de validade S2S não coincide com o nome do arquivo.")
            for region in run.config["regions"]:
                i = int(np.argmin(np.abs(latitudes - region["latitude"])))
                j = int(np.argmin(np.abs(longitudes - region["longitude"])))
                temperature = t2m[0, i, j]
                rainfall = precip[0, i, j]
                if np.ma.is_masked(temperature) or np.ma.is_masked(rainfall):
                    continue
                temperature_c, rain_mm = float(temperature) - 273.15, float(rainfall) * 86400.
                if not (-100 <= temperature_c <= 80 and 0 <= rain_mm < 3000):
                    raise ValueError("Temperatura ou precipitação S2S fora de limites físicos simples.")
                records.append({"region_id": region_id(region["latitude"], region["longitude"]),
                    "initialized": initial_date, "valid_date": item["valid_date"], "ensemble": ensemble,
                    "latitude_grid": float(latitudes[i]), "longitude_grid": float(longitudes[j]),
                    "temperature_c": temperature_c, "precipitation_mm_day": rain_mm,
                    "source_sha256": item["sha256"]})
    frame = pd.DataFrame(records)
    out = root / f"sample_{initial_date}_ens{ensemble}.parquet"
    frame.to_parquet(out, index=False)
    power_path = run.root / "runs" / run.config["weather_run"] / "weather_zarr/daily.parquet"
    power = pd.read_parquet(power_path)
    power["valid_date"] = pd.to_datetime(power.date).dt.strftime("%Y-%m-%d")
    paired = frame.merge(power[["region_id", "valid_date", "T2M", "PRECTOTCORR"]],
                         on=["region_id", "valid_date"], how="inner", validate="one_to_one")
    def compare(column: str, reference: str) -> dict[str, Any]:
        diff = (paired[column] - paired[reference]).dropna()
        return {"n": len(diff), "mean_difference": float(diff.mean()) if len(diff) else None,
                "mae": float(diff.abs().mean()) if len(diff) else None}
    summary = {"source": "NASA GEOS-S2S-3 SubC retrospective", "initialized": initial_date, "ensemble": ensemble,
        "records": len(frame), "days": len(selected), "grid_resolution_degrees": [.5, .5],
        "parquet_sha256": sha256_file(out), "power_source_sha256": sha256_file(power_path),
        "comparison_with_power_train_period": {"temperature_c": compare("temperature_c", "T2M"),
                                                 "precipitation_mm_day": compare("precipitation_mm_day", "PRECTOTCORR")},
        "limitations": ["Reforecast produzido em 2026: inicialização histórica não prova disponibilidade na época.",
            "Uma única inicialização e um membro não bastam para calibrar o desvio entre fontes.",
            "Sem correção automática nem uso como previsão meteorológica operacional do modelo."]}
    write_json(root / f"sample_{initial_date}_ens{ensemble}.json", summary)
    run.stage("s2s_extract", "complete", records=len(frame), days=len(selected), output=str(out))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/nasa_complement_v3b.json"))
    parser.add_argument("--run-id", default="s2s_nasa_v3_20261111")
    parser.add_argument("--initial-date", default="2021-09-03")
    parser.add_argument("--days", type=int, default=8)
    parser.add_argument("--ensemble", type=int, default=1)
    parser.add_argument("--extract", action="store_true", help="Extrai as áreas configuradas após o download")
    args = parser.parse_args()
    run = Run(args.config, args.run_id)
    print(download_cycle(run, args.initial_date, args.days, args.ensemble))
    if args.extract:
        print(extract_cycle(run, args.initial_date, args.ensemble))


if __name__ == "__main__":
    main()
