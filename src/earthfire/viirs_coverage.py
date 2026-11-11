"""Descobre e baixa grânulos NASA VJ114A1 sem confundir 1 km com VIIRS 375 m."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from .io import read_json, replace_with_retry, sha256_file, utc_now, write_json
from .runtime import ResourceLimit, Run

CMR_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"
DATA_HOST = "data.lpdaac.earthdatacloud.nasa.gov"
PRODUCT = "VJ114A1"
NAME = re.compile(r"^VJ114A1\.A\d{7}\.h\d{2}v\d{2}\.002\.\d+\.h5$")
HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"


def query_granules(run: Run, bbox: tuple[float, float, float, float], start: str, end: str,
                   limit: int = 5, session: requests.Session | None = None) -> dict[str, Any]:
    west, south, east, north = bbox
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ValueError("Área inválida: use oeste sul leste norte, sem cruzar o antimeridiano.")
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    max_days = run.config["operational"].get("earthdata_max_query_days", 32)
    max_granules = run.config["operational"].get("earthdata_max_query_granules", 30)
    if first > last or last > datetime.now(timezone.utc).date() or (last - first).days + 1 > max_days:
        raise ValueError("Datas inválidas ou intervalo maior que o limite configurado.")
    if not 1 <= limit <= max_granules:
        raise ValueError("A quantidade de grânulos excede o limite configurado.")
    params = {"short_name": PRODUCT, "version": "002",
              "temporal": f"{start}T00:00:00Z,{end}T23:59:59Z",
              "bounding_box": ",".join(str(x) for x in bbox), "page_size": limit,
              "sort_key": "start_date"}
    client = session or requests.Session()
    last_error = "sem resposta"
    for attempt in range(min(3, run.config["operational"]["network_retries"])):
        run.guard()
        try:
            response = client.get(CMR_URL, params=params, timeout=run.config["operational"]["network_timeout_seconds"])
            if response.status_code == 429 or response.status_code >= 500:
                last_error = f"HTTP {response.status_code}"
                retry = response.headers.get("Retry-After", "")
                if retry.isdigit() and int(retry) > 30:
                    raise ResourceLimit("CMR solicitou espera superior a 30 segundos.")
                if attempt < 2: time.sleep(max(2 ** attempt, int(retry) if retry.isdigit() else 0))
                continue
            if 400 <= response.status_code < 500:
                raise ValueError(f"CMR recusou a consulta: HTTP {response.status_code}.")
            response.raise_for_status()
            entries = response.json()["feed"]["entry"]
            granules = []
            for item in entries:
                title = item.get("title", "")
                candidate = next((link.get("href", "") for link in item.get("links", [])
                                  if link.get("href", "").endswith(".h5")), "")
                parsed = urlparse(candidate)
                filename = Path(parsed.path).name
                if (parsed.scheme != "https" or parsed.netloc != DATA_HOST or
                        not parsed.path.startswith("/lp-prod-protected/VJ114A1.002/") or
                        not NAME.fullmatch(filename) or filename != title + ".h5"):
                    raise ValueError("CMR retornou um endereço de dados não reconhecido.")
                granules.append({"title": title, "filename": filename, "url": candidate,
                                 "start_utc": item.get("time_start"), "end_utc": item.get("time_end"),
                                 "size_mb_reported": item.get("granule_size")})
            hits = int(response.headers.get("CMR-Hits", len(granules)))
            digest = hashlib.sha256(response.content).hexdigest()
            raw_path = run.path / "viirs_coverage" / f"cmr_{digest}.json"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            if not raw_path.exists():
                raw_path.write_bytes(response.content)
            elif sha256_file(raw_path) != digest:
                raise ValueError("Cache CMR incompatível.")
            return {"status": "ok", "product": PRODUCT + ".002", "query": params,
                    "retrieved_utc": utc_now(), "cmr_response_sha256": digest, "cmr_response": str(raw_path),
                    "total_hits": hits, "returned": len(granules), "truncated": hits > len(granules),
                    "granules": granules, "limitation": "Composto diário VIIRS de 1 km; não é a máscara do produto de 375 m."}
        except requests.RequestException as exc:
            last_error = type(exc).__name__
            if attempt < 2: time.sleep(2 ** attempt)
    raise ConnectionError(f"Consulta CMR indisponível: {last_error}")


def download_granules(run: Run, discovery: dict[str, Any], token: str | None,
                      session: requests.Session | None = None) -> dict[str, Any]:
    if not token:
        return {"status": "authentication_required", "downloaded": 0,
                "reason": "Defina EARTHDATA_TOKEN no ambiente com um token Earthdata válido."}
    if discovery.get("product") != PRODUCT + ".002":
        raise ValueError("Produto incorreto para este coletor.")
    target_dir = run.path / "viirs_coverage"
    target_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = target_dir / "manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {"requests": 0, "files": {}, "temporary_files": []}
    client = session or requests.Session()
    count = 0
    for item in discovery["granules"]:
        filename, url = item["filename"], item["url"]
        parsed = urlparse(url)
        if (not NAME.fullmatch(filename) or parsed.scheme != "https" or parsed.netloc != DATA_HOST or
                not parsed.path.startswith("/lp-prod-protected/VJ114A1.002/") or Path(parsed.path).name != filename):
            raise ValueError("Endereço de grânulo inseguro ou incompatível.")
        target = target_dir / filename
        if target.exists():
            if filename not in manifest["files"] or sha256_file(target) != manifest["files"][filename]["sha256"]:
                raise ValueError("Grânulo existente não corresponde ao manifesto.")
            count += 1
            continue
        temp = target.with_name(target.name + ".part")
        if str(temp) not in manifest["temporary_files"]:
            manifest["temporary_files"].append(str(temp))
        error = "sem resposta"
        for attempt in range(min(3, run.config["operational"]["network_retries"])):
            run.guard()
            if manifest["requests"] >= run.config["operational"].get("earthdata_max_requests", 100):
                raise ResourceLimit("Limite de requisições de grânulos VIIRS atingido.")
            delay = 1. - (time.time() - manifest.get("last_request_epoch", 0))
            if delay > 0: time.sleep(delay)
            manifest["requests"] += 1
            manifest["last_request_epoch"] = time.time()
            write_json(manifest_path, manifest)
            try:
                with client.get(url, headers={"Authorization": "Bearer " + token}, stream=True,
                                timeout=run.config["operational"]["network_timeout_seconds"]) as response:
                    if response.status_code in (401, 403):
                        return {"status": "authentication_required", "downloaded": count,
                                "reason": f"Earthdata recusou o acesso: HTTP {response.status_code}."}
                    if response.status_code == 429 or response.status_code >= 500:
                        error = f"HTTP {response.status_code}"
                        retry = response.headers.get("Retry-After", "")
                        if retry.isdigit() and int(retry) > 30:
                            raise ResourceLimit("Earthdata solicitou espera superior a 30 segundos.")
                        if attempt < 2: time.sleep(max(2 ** attempt, int(retry) if retry.isdigit() else 0))
                        continue
                    if 400 <= response.status_code < 500:
                        return {"status": "data_unavailable", "downloaded": count,
                                "failed_file": filename,
                                "reason": f"Earthdata recusou o grânulo: HTTP {response.status_code}."}
                    response.raise_for_status()
                    expected = int(response.headers.get("Content-Length", "0") or "0")
                    if expected > 128 * 1024 * 1024:
                        raise ResourceLimit("Grânulo excede o limite de 128 MiB.")
                    size = 0
                    with temp.open("wb") as stream:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            size += len(chunk)
                            if size > 128 * 1024 * 1024:
                                raise ResourceLimit("Grânulo excedeu o limite de 128 MiB.")
                            stream.write(chunk)
                            if size % (8 * 1024 * 1024) == 0: run.guard()
                    if expected and size != expected:
                        raise IOError("Download incompleto.")
                    with temp.open("rb") as stream:
                        if stream.read(8) != HDF5_MAGIC:
                            return {"status": "authentication_required", "downloaded": count,
                                    "reason": "Earthdata não retornou um arquivo HDF5; verifique o acesso e as autorizações."}
                    digest = sha256_file(temp)
                    replace_with_retry(temp, target)
                    manifest["files"][filename] = {**item, "bytes": size, "sha256": digest, "retrieved_utc": utc_now()}
                    write_json(manifest_path, manifest)
                    count += 1
                    break
            except (requests.RequestException, OSError) as exc:
                # Não expor mensagens de rede que eventualmente contenham credenciais.
                error = type(exc).__name__
                if attempt < 2: time.sleep(2 ** attempt)
        else:
            return {"status": "data_unavailable", "downloaded": count, "failed_file": filename,
                    "reason": f"Download Earthdata falhou: {error}."}
    return {"status": "ok", "downloaded": count, "manifest": str(manifest_path),
            "limitation": "Grânulos ainda precisam ser auditados, georreferenciados e integrados ao modelo."}


def download_with_gaps(run: Run, discovery: dict[str, Any], token: str,
                       session: requests.Session | None = None) -> dict[str, Any]:
    """Prossegue após falhas individuais registradas, sem transformar lacunas em zero."""
    folder = run.path / "viirs_coverage"
    failure_path = folder / "unavailable_granules.json"
    failures = read_json(failure_path) if failure_path.exists() else {"files": {}}
    pending = [x for x in discovery["granules"] if x["filename"] not in failures["files"]]
    consecutive = 0
    while pending:
        manifest_path = folder / "manifest.json"
        before = len(read_json(manifest_path)["files"]) if manifest_path.exists() else 0
        outcome = download_granules(run, {**discovery, "granules": pending}, token, session=session)
        if outcome["status"] == "ok":
            break
        if outcome["status"] != "data_unavailable" or not outcome.get("failed_file"):
            raise ConnectionError(f"Earthdata: {outcome['status']}")
        filename = outcome["failed_file"]
        failures["files"][filename] = {"status": outcome["status"], "reason": outcome["reason"], "recorded_utc": utc_now()}
        write_json(failure_path, failures)
        after = len(read_json(manifest_path)["files"]) if manifest_path.exists() else 0
        consecutive = 1 if after > before else consecutive + 1
        if consecutive >= 3:
            raise ConnectionError("Três grânulos consecutivos indisponíveis; coleta pausada para evitar ciclo de falhas.")
        pending = [item for item in pending if item["filename"] != filename]
    saved = read_json(folder / "manifest.json")["files"] if (folder / "manifest.json").exists() else {}
    names = [item["filename"] for item in discovery["granules"] if item["filename"] in saved]
    missing = [item["filename"] for item in discovery["granules"] if item["filename"] not in saved]
    return {"status": "partial" if missing else "ok", "files": names, "unavailable_files": missing}


def main() -> None:
    parser = argparse.ArgumentParser(description="Busca cobertura VIIRS NOAA-20 no catálogo NASA CMR")
    parser.add_argument("--config", type=Path, default=Path("config/nasa_complement_v3b.json"))
    parser.add_argument("--run-id", default="viirs_coverage_v3")
    parser.add_argument("--bbox", nargs=4, type=float, metavar=("WEST", "SOUTH", "EAST", "NORTH"), required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--max-granules", type=int, default=5)
    parser.add_argument("--download", action="store_true", help="Baixa os grânulos com EARTHDATA_TOKEN")
    args = parser.parse_args()
    run = Run(args.config, args.run_id)
    discovery = query_granules(run, tuple(args.bbox), args.start, args.end, args.max_granules)
    if args.download:
        discovery["download"] = download_granules(run, discovery, os.environ.get("EARTHDATA_TOKEN"))
    output = run.path / "viirs_coverage/discovery.json"
    write_json(output, discovery)
    run.stage("viirs_discovery", "complete", count=discovery["returned"], output=str(output))
    print(json.dumps(discovery, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
