"""Cliente NASA POWER Daily; cache bruto local e orçamento global de requisições."""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
import uuid
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import requests

from .io import read_json, utc_now, write_json

PARAMETERS = ("T2M", "T2M_MAX", "T2M_MIN", "RH2M", "PRECTOTCORR", "WS10M")
EXPECTED_UNITS = {"T2M": "C", "T2M_MAX": "C", "T2M_MIN": "C", "RH2M": "%", "PRECTOTCORR": "mm/day", "WS10M": "m/s"}
URL = "https://power.larc.nasa.gov/api/temporal/daily/point"


class WeatherUnavailable(RuntimeError):
    pass


def grid_center(latitude: float, longitude: float) -> tuple[float, float]:
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        raise ValueError("Coordenadas fora dos limites geográficos")
    # Centro da grade MERRA-2 de 0,5° × 0,625°. A geometria retornada também é registrada.
    lat = min(90.0, max(-90.0, -90.0 + math.floor((latitude + 90.0) / 0.5 + 0.5) * 0.5))
    lon = -180.0 + math.floor(((longitude + 180.0) % 360) / 0.625 + 0.5) * 0.625
    if lon >= 180:
        lon -= 360
    return lat, lon


class NasaPowerClient:
    def __init__(self, cache_dir: Path, max_requests: int = 500, timeout: float = 45,
                 retries: int = 3, session: requests.Session | None = None, min_interval: float = 1.0):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_requests, self.timeout, self.retries = max_requests, timeout, min(3, max(1, retries))
        self.session = session or requests.Session()
        self.min_interval, self._last_request = min_interval, 0.0
        self.counter_path = self.cache_dir / "request_budget.json"

    def fetch(self, latitude: float, longitude: float, start: str, end: str,
              parameters: Sequence[str] = PARAMETERS, community: str = "AG") -> dict[str, Any]:
        lat, lon = grid_center(float(latitude), float(longitude))
        begin, finish = date.fromisoformat(start), date.fromisoformat(end)
        if begin > finish or finish > date.today():
            raise ValueError("Datas inválidas: POWER Daily não é previsão futura.")
        if not parameters or len(parameters) > 20 or len(set(parameters)) != len(parameters):
            raise ValueError("Informe entre 1 e 20 parâmetros distintos.")
        if community not in {"AG", "RE", "SB"}:
            raise ValueError("Comunidade POWER inválida.")
        query = {"parameters": ",".join(parameters), "community": community, "longitude": lon,
                 "latitude": lat, "start": begin.strftime("%Y%m%d"), "end": finish.strftime("%Y%m%d"),
                 "format": "JSON", "time-standard": "UTC"}
        key = hashlib.sha256(json.dumps(query, sort_keys=True).encode()).hexdigest()
        target = self.cache_dir / (key + ".json")
        if target.exists():
            return read_json(target)
        # Uma resposta já arquivada para intervalo maior satisfaz uma consulta menor
        # na mesma célula, reduzindo pedidos redundantes ao serviço.
        for cached in self.cache_dir.glob("*.json"):
            if cached == self.counter_path:
                continue
            envelope = read_json(cached)
            previous = envelope.get("request", {})
            same = all(previous.get(k) == query[k] for k in ("latitude", "longitude", "community", "time-standard", "format"))
            have = set(str(previous.get("parameters", "")).split(","))
            if same and set(parameters).issubset(have) and previous.get("start", "99999999") <= query["start"] and previous.get("end", "00000000") >= query["end"]:
                envelope["effective_request"] = query
                return envelope
        return self._retrieve(query, target)

    @contextmanager
    def _request_lock(self):
        """Serializa clientes que compartilham o cache; não remove trava de terceiros."""
        lock = self.cache_dir / ".request.lock"
        token = f"{os.getpid()}:{uuid.uuid4().hex}"
        deadline = time.monotonic() + 30
        while True:
            try:
                with lock.open("x", encoding="utf-8") as stream:
                    stream.write(token)
                break
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise WeatherUnavailable("Outra consulta POWER ainda detém a trava do cache.")
                time.sleep(.2)
        try:
            state = read_json(self.counter_path) if self.counter_path.exists() else {"requests": 0}
            manifest = state.setdefault("temporary_files", [])
            if str(lock) not in manifest:
                manifest.append(str(lock))
                write_json(self.counter_path, state)
            yield
        finally:
            if lock.exists() and lock.read_text(encoding="utf-8") == token:
                lock.unlink()

    def _retrieve(self, query: dict[str, Any], target: Path) -> dict[str, Any]:
        with self._request_lock():
            # Outro processo pode ter concluído esta mesma consulta enquanto aguardávamos.
            if target.exists():
                return read_json(target)
            return self._request_with_retries(query, target)

    def _request_with_retries(self, query: dict[str, Any], target: Path) -> dict[str, Any]:
        last_error = "sem resposta"
        for attempt in range(self.retries):
            state = read_json(self.counter_path) if self.counter_path.exists() else {"requests": 0}
            if state["requests"] >= self.max_requests:
                raise WeatherUnavailable("Orçamento de requisições NASA POWER esgotado.")
            wait = self.min_interval - (time.time() - state.get("last_request_epoch", 0))
            if wait > 0:
                time.sleep(wait)
            state["requests"] += 1
            state["last_request_epoch"] = time.time()
            write_json(self.counter_path, state)
            try:
                self._last_request = time.monotonic()
                response = self.session.get(URL, params=query, timeout=self.timeout)
                if response.status_code == 429 or response.status_code >= 500:
                    last_error = f"HTTP {response.status_code}"
                    if attempt + 1 < self.retries:
                        retry_after = response.headers.get("Retry-After", "")
                        if retry_after.isdigit() and int(retry_after) > 30:
                            raise WeatherUnavailable(f"Serviço solicitou aguardar {retry_after}s; consulta interrompida sem antecipar a repetição.")
                        time.sleep(max(2 ** attempt, int(retry_after) if retry_after.isdigit() else 0))
                    continue
                response.raise_for_status()
                raw = response.json()
                if not isinstance(raw.get("properties", {}).get("parameter"), dict):
                    raise WeatherUnavailable("Resposta POWER sem séries de parâmetros.")
                envelope = {"request": query, "retrieved_utc": utc_now(), "url": URL, "response": raw}
                write_json(target, envelope)
                return envelope
            except (requests.RequestException, ValueError) as exc:
                last_error = str(exc)
                if attempt + 1 < self.retries:
                    time.sleep(2 ** attempt)
        raise WeatherUnavailable(f"NASA POWER indisponível após {self.retries} tentativa(s): {last_error}")


def power_frame(envelope: dict[str, Any]) -> pd.DataFrame:
    raw = envelope["response"]
    request = envelope.get("effective_request", envelope.get("request", {}))
    allowed = set(str(request.get("parameters", "")).split(","))
    data = {k: v for k, v in raw["properties"]["parameter"].items() if not allowed or k in allowed}
    frame = pd.DataFrame(data).rename_axis("date").reset_index()
    frame["date"] = pd.to_datetime(frame["date"], format="%Y%m%d", errors="raise")
    fill = raw.get("header", {}).get("fill_value", -999)
    for name in data:
        frame[name] = pd.to_numeric(frame[name], errors="coerce").replace(float(fill), float("nan"))
    if "start" in request and "end" in request:
        start, end = pd.to_datetime(request["start"], format="%Y%m%d"), pd.to_datetime(request["end"], format="%Y%m%d")
        frame = frame.loc[frame.date.between(start, end)]
    return frame.sort_values("date").reset_index(drop=True)
