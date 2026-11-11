"""Cliente opcional da API FIRMS; a chave nunca é gravada em saídas ou logs."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import requests

from .credentials import load_firms_key
from .io import read_json, replace_with_retry, sha256_file, utc_now, write_json
from .runtime import ResourceLimit, Run

SOURCES = {"MODIS_SP", "MODIS_NRT", "VIIRS_NOAA20_SP", "VIIRS_NOAA20_NRT"}
ENDPOINT = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"


def validate_query(source: str, bbox: tuple[float, float, float, float], start: str, days: int) -> None:
    if source not in SOURCES:
        raise ValueError("Produto FIRMS não suportado.")
    west, south, east, north = bbox
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ValueError("Área deve ser west,south,east,north dentro dos limites geográficos.")
    if days not in range(1, 6):
        raise ValueError("A API FIRMS aceita de 1 a 5 dias por consulta.")
    if date.fromisoformat(start) + timedelta(days=days - 1) > date.today():
        raise ValueError("O período FIRMS não pode avançar para datas futuras.")


def fetch_area(run: Run, source: str, bbox: tuple[float, float, float, float], start: str, days: int,
               session: requests.Session | None = None) -> dict[str, Any]:
    validate_query(source, bbox, start, days)
    key = load_firms_key()
    if not key:
        return {"status": "authentication_required", "reason": "Configure FIRMS_MAP_KEY ou use o Gerenciador de Credenciais do Windows."}
    if not key.isalnum() or len(key) > 128:
        raise ValueError("Formato de FIRMS_MAP_KEY inválido.")
    query = {"source": source, "bbox": list(bbox), "start": start, "days": days}
    query_hash = hashlib.sha256(json.dumps(query, sort_keys=True).encode()).hexdigest()
    folder = run.path / "firms_api"
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / "manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {"requests": 0, "files": {}}
    target = folder / (query_hash + ".csv")
    if target.exists():
        item = manifest["files"].get(query_hash)
        if not item or item["sha256"] != sha256_file(target):
            raise ValueError("Cache FIRMS incompatível com o manifesto.")
        return {"status": "ok", **item, "path": str(target)}
    area = ",".join(str(x) for x in bbox)
    url = f"{ENDPOINT}/{key}/{source}/{area}/{days}/{start}"
    client = session or requests.Session()
    last = "sem resposta"
    for attempt in range(min(3, run.config["operational"]["network_retries"])):
        run.guard()
        if manifest["requests"] >= 500:
            raise ResourceLimit("Limite local de consultas FIRMS atingido.")
        delay = 1. - (time.time() - manifest.get("last_request_epoch", 0))
        if delay > 0: time.sleep(delay)
        manifest["requests"] += 1
        manifest["last_request_epoch"] = time.time()
        write_json(manifest_path, manifest)
        response = None
        try:
            response = client.get(url, timeout=run.config["operational"]["network_timeout_seconds"], stream=True)
            if response.status_code == 429 or response.status_code >= 500:
                last = f"HTTP {response.status_code}"
                retry_after = response.headers.get("Retry-After", "")
                if retry_after.isdigit() and int(retry_after) > 30:
                    raise ResourceLimit("FIRMS solicitou uma espera superior a 30 segundos.")
                if attempt < 2: time.sleep(max(2 ** attempt, int(retry_after) if retry_after.isdigit() else 0))
                continue
            if 400 <= response.status_code < 500:
                return {"status": "unavailable", "reason": f"FIRMS recusou a solicitação: HTTP {response.status_code}."}
            expected = int(response.headers.get("Content-Length", "0") or "0")
            if expected > 64 * 1024 * 1024:
                raise ResourceLimit("Resposta FIRMS excede 64 MiB por consulta; reduza a área ou o período.")
            content = bytearray()
            for piece in response.iter_content(chunk_size=128 * 1024):
                content.extend(piece)
                if len(content) > 64 * 1024 * 1024:
                    raise ResourceLimit("Resposta FIRMS excede 64 MiB por consulta; reduza a área ou o período.")
            response.close()
            if expected and len(content) != expected:
                raise IOError("Resposta FIRMS incompleta.")
            header = content.decode("utf-8-sig").splitlines()[0] if content else ""
            if not {"latitude", "longitude", "acq_date", "acq_time", "satellite"}.issubset(set(header.split(","))):
                return {"status": "unavailable", "reason": "A resposta FIRMS não contém um CSV reconhecido."}
            temporary = target.with_name(target.name + ".part")
            manifest.setdefault("temporary_files", []).append(str(temporary))
            write_json(manifest_path, manifest)
            temporary.write_bytes(content)
            replace_with_retry(temporary, target)
            item = {"query": query, "endpoint": ENDPOINT + "/[MAP_KEY]/...", "retrieved_utc": utc_now(),
                    "sha256": sha256_file(target), "bytes": len(content),
                    "limitation": "O CSV contém detecções; não estabelece cobertura nem ausência de fogo."}
            manifest["files"][query_hash] = item
            write_json(manifest_path, manifest)
            return {"status": "ok", **item, "path": str(target)}
        except (requests.RequestException, IOError) as exc:
            # URLs da biblioteca podem conter a chave; nunca registrar a mensagem da exceção.
            last = type(exc).__name__
            if attempt < 2: time.sleep(2 ** attempt)
        finally:
            if response is not None:
                response.close()
    return {"status": "unavailable", "reason": f"Consulta FIRMS falhou após retentativas: {last}."}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/nasa_complement_v3b.json"))
    parser.add_argument("--run-id", default="firms_api_v3")
    parser.add_argument("--source", choices=sorted(SOURCES), required=True)
    parser.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    parser.add_argument("--start", required=True)
    parser.add_argument("--days", type=int, default=5)
    args = parser.parse_args()
    result = fetch_area(Run(args.config, args.run_id), args.source, tuple(args.bbox), args.start, args.days)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
