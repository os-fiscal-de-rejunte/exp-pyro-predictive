"""Recorte de ciclos GEOS-FP arquivados, sem confundir previsão e reanálise POWER."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from .io import read_json, sha256_file, utc_now, write_json
from .runtime import ResourceLimit, Run

BASE = "https://opendap.nccs.nasa.gov/dods/GEOS-5/fp/0.25_deg/fcast"
COLLECTIONS = {"tavg1_2d_slv_Nx": ("t2m", "qv2m", "ps", "u10m", "v10m"),
               "tavg1_2d_flx_Nx": ("prectot",)}
MONTHS = {name: index for index, name in enumerate(
    ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"), 1)}


def parse_grads_start(metadata: str) -> pd.Timestamp:
    match = re.search(r'String grads_min "(\d{2}):(\d{2})z(\d{1,2})([a-z]{3})(\d{4})"', metadata, re.I)
    if not match:
        raise ValueError("GEOS não informou início temporal interpretável.")
    hour, minute, day, month, year = match.groups()
    return pd.Timestamp(datetime(int(year), MONTHS[month.lower()], int(day), int(hour), int(minute), tzinfo=timezone.utc))


def parse_ascii(body: str, variable: str) -> tuple[np.ndarray, np.ndarray, float, float]:
    lines = body.splitlines()
    match = re.match(rf"^{re.escape(variable)}, \[(\d+)\]\[1\]\[1\]$", lines[0].strip()) if lines else None
    if not match:
        raise ValueError(f"Resposta GEOS inválida para {variable}.")
    count = int(match.group(1))
    data = []
    for line in lines[1:]:
        if re.match(r"^[a-zA-Z_][\w.]*, \[", line):
            break
        entry = re.match(r"^\[(\d+)\]\[0\],\s*([-+0-9.eE]+)\s*$", line.strip())
        if entry:
            if int(entry.group(1)) != len(data):
                raise ValueError("Índice GEOS fora de ordem.")
            data.append(float(entry.group(2)))
    def vector(name: str) -> np.ndarray:
        for i,line in enumerate(lines):
            found = re.match(rf"^{name}, \[(\d+)\]$", line.strip())
            if found:
                n = int(found.group(1))
                pieces = []
                for following in lines[i + 1:]:
                    if not following.strip() or re.match(r"^[a-zA-Z_][\w.]*, \[", following):
                        break
                    pieces.extend(float(x.strip()) for x in following.split(",") if x.strip())
                if len(pieces) != n:
                    raise ValueError(f"Eixo GEOS {name} incompleto.")
                return np.array(pieces)
        raise ValueError(f"Eixo GEOS {name} ausente.")
    times, lat, lon = vector("time"), vector("lat"), vector("lon")
    if len(data) != count or len(times) != count or len(lat) != 1 or len(lon) != 1:
        raise ValueError("Dimensões GEOS inesperadas.")
    values = np.asarray(data, dtype=float)
    values[np.abs(values) >= 1e14] = np.nan
    return values, times, float(lat[0]), float(lon[0])


def daily_weather(hourly: pd.DataFrame) -> pd.DataFrame:
    """Transformações físicas explícitas; não aplica correção para NASA POWER."""
    frame = hourly.copy()
    frame["T2M"] = frame.t2m - 273.15
    vapor_pressure = frame.qv2m * frame.ps / (0.622 + 0.378 * frame.qv2m)
    saturation = 610.94 * np.exp(17.625 * frame.T2M / (frame.T2M + 243.04))
    frame["RH2M_derived"] = 100 * vapor_pressure / saturation
    frame["WS10M"] = np.hypot(frame.u10m, frame.v10m)
    frame["precip_geos_mm"] = frame.prectot * 3600 if "prectot" in frame else np.nan
    if not frame.T2M.dropna().between(-100, 80).all() or not frame.RH2M_derived.dropna().between(0, 105).all():
        raise ValueError("Temperatura ou umidade GEOS fisicamente inválida.")
    frame["date"] = frame.time.dt.floor("D")
    grouped = frame.groupby("date")
    count = grouped[["T2M", "RH2M_derived", "WS10M"]].count()
    out = grouped[["T2M", "RH2M_derived", "WS10M"]].mean()
    out["T2M_MAX_hourly_mean"] = grouped.T2M.max()
    out["T2M_MIN_hourly_mean"] = grouped.T2M.min()
    out["precip_geos_mm"] = grouped.precip_geos_mm.sum(min_count=24)
    out = out.loc[count.eq(24).all(axis=1)].reset_index()
    out["date"] = out.date.dt.strftime("%Y-%m-%d")
    return out


class GeosClient:
    def __init__(self, run: Run, max_requests: int = 80) -> None:
        self.run, self.max_requests = run, max_requests
        self.base = run.path / "geos_forecast"
        self.cache = self.base / "cache"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.base / "manifest.json"
        self.manifest = read_json(self.manifest_path) if self.manifest_path.exists() else {"requests": 0, "files": {}}
        self.session = requests.Session()

    def get(self, url: str) -> str:
        key = hashlib.sha256(url.encode()).hexdigest()
        path = self.cache / (key + ".txt")
        if path.exists():
            item = self.manifest["files"].get(key)
            if not item or sha256_file(path) != item["sha256"]:
                raise ValueError("Cache GEOS incompatível.")
            return path.read_text(encoding="utf-8")
        error = "sem resposta"
        for attempt in range(min(3, self.run.config["operational"]["network_retries"])):
            self.run.guard()
            if self.manifest["requests"] >= self.max_requests:
                raise ResourceLimit("Limite de requisições GEOS atingido.")
            delay = 1. - (time.time() - self.manifest.get("last_request_epoch", 0))
            if delay > 0:
                time.sleep(delay)
            self.manifest["requests"] += 1
            self.manifest["last_request_epoch"] = time.time()
            write_json(self.manifest_path, self.manifest)
            try:
                response = self.session.get(url, timeout=min(30, self.run.config["operational"]["network_timeout_seconds"]))
                if response.status_code == 429 or response.status_code >= 500:
                    error = f"HTTP {response.status_code}"
                    retry_after = response.headers.get("Retry-After", "")
                    if retry_after.isdigit() and int(retry_after) > 30:
                        raise ResourceLimit(f"GEOS solicitou esperar {retry_after}s; operação interrompida.")
                    if attempt < 2: time.sleep(max(2 ** attempt, int(retry_after) if retry_after.isdigit() else 0))
                    continue
                if 400 <= response.status_code < 500:
                    raise ValueError(f"GEOS recusou a consulta: HTTP {response.status_code}")
                response.raise_for_status()
                if "<html" in response.text[:300].lower() or len(response.content) > 2 * 1024 * 1024:
                    raise ValueError("GEOS retornou erro HTML ou resposta maior que o limite.")
                path.write_text(response.text, encoding="utf-8")
                self.manifest["files"][key] = {"url": url, "sha256": sha256_file(path), "retrieved_utc": utc_now()}
                write_json(self.manifest_path, self.manifest)
                return response.text
            except (requests.RequestException, OSError) as exc:
                error = f"{type(exc).__name__}: {exc}"
                if attempt < 2: time.sleep(2 ** attempt)
        raise ConnectionError(f"GEOS indisponível: {error}")

    def fetch(self, latitude: float, longitude: float, cycle: str) -> dict[str, Any]:
        if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
            raise ValueError("Coordenadas inválidas.")
        if not re.fullmatch(r"\d{8}_(00|06|12|18)", cycle):
            raise ValueError("Ciclo esperado AAAAMMDD_HH, com HH em 00, 06, 12 ou 18.")
        initialized = pd.Timestamp(datetime.strptime(cycle, "%Y%m%d_%H"), tz="UTC")
        if initialized > pd.Timestamp.now(tz="UTC"):
            raise ValueError("Ciclo GEOS no futuro.")
        lat_index = min(720, max(0, int(math.floor((latitude + 90) / .25 + .5))))
        lon_index = int(math.floor(((longitude + 180) % 360) / .3125 + .5)) % 1152
        pieces = []
        sources = []
        for collection, variables in COLLECTIONS.items():
            base = f"{BASE}/{collection}/{collection}.{cycle}"
            dds, das = self.get(base + ".dds"), self.get(base + ".das")
            first = parse_grads_start(das)
            frame = None
            times_reference = None
            for variable in variables:
                match = re.search(rf"Float32 {variable}\[time = (\d+)\]", dds)
                if not match:
                    raise ValueError(f"Variável {variable} ausente na coleção GEOS.")
                count = int(match.group(1))
                url = f"{base}.ascii?{variable}[0:1:{count-1}][{lat_index}:1:{lat_index}][{lon_index}:1:{lon_index}]"
                values, times, lat, lon = parse_ascii(self.get(url), variable)
                if len(times) > 1 and not np.allclose(np.diff(times), 1 / 24, atol=1e-6):
                    raise ValueError("O eixo GEOS não é horário e contínuo.")
                if times_reference is not None and not np.allclose(times, times_reference):
                    raise ValueError("Eixos de tempo GEOS divergentes.")
                times_reference = times
                if frame is None:
                    frame = pd.DataFrame({"time": pd.date_range(first, periods=count, freq="h")})
                frame[variable] = values
            if frame is None:
                raise ValueError("Coleção GEOS vazia.")
            pieces.append(frame)
            sources.append({"collection": collection, "source": base, "latitude_grid": lat, "longitude_grid": lon})
        hourly = pieces[0].merge(pieces[1], on="time", how="inner", validate="one_to_one")
        hourly = hourly.loc[hourly.time >= initialized].copy()
        daily = daily_weather(hourly)
        daily = daily.astype(object).where(pd.notna(daily), None)
        return {"source": "NASA GEOS-FP forecast via NASA NCCS OPeNDAP", "cycle_initialized_utc": initialized.isoformat(),
                "retrieved_utc": utc_now(), "requested_location": {"latitude": latitude, "longitude": longitude},
                "grids": sources, "daily": daily.to_dict(orient="records"),
                "units": {"T2M": "C", "T2M_MAX_hourly_mean": "C", "T2M_MIN_hourly_mean": "C",
                          "RH2M_derived": "%", "precip_geos_mm": "mm/day", "WS10M": "m/s"},
                "limitations": ["Ciclo inicializado não comprova disponibilidade histórica na data de emissão.",
                    "RH é derivada por fórmula; precipitação GEOS não é PRECTOTCORR do POWER.",
                    "Diferenças GEOS/POWER não foram calibradas no treino; não usar diretamente como se fossem equivalentes.",
                    "NASA GEOS-FP é experimental e pode ter horizonte variável."]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Recorta uma previsão histórica NASA GEOS-FP")
    parser.add_argument("--config", type=Path, default=Path("config/nasa_complement_v3b.json"))
    parser.add_argument("--run-id", default="geos_nasa_v3_20261111")
    parser.add_argument("--latitude", type=float, default=-11.5)
    parser.add_argument("--longitude", type=float, default=-46.875)
    parser.add_argument("--cycle", default="20261103_00")
    args = parser.parse_args()
    run = Run(args.config, args.run_id)
    output = run.path / "geos_forecast" / f"forecast_{args.cycle}.json"
    try:
        result = GeosClient(run).fetch(args.latitude, args.longitude, args.cycle)
    except ConnectionError as exc:
        result = {"status": "weather_unavailable", "source": "NASA GEOS-FP",
                  "cycle": args.cycle, "latitude": args.latitude, "longitude": args.longitude,
                  "retrieved_utc": utc_now(), "reason": str(exc), "daily": None}
        write_json(output, result)
        run.stage("geos_forecast", "unavailable", cycle=args.cycle, output=str(output))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    result["status"] = "ok"
    write_json(output, result)
    run.stage("geos_forecast", "complete", cycle=args.cycle, daily_records=len(result["daily"]), output=str(output))
    print(f"{len(result['daily'])} dias GEOS preservados em {output}")


if __name__ == "__main__":
    main()
