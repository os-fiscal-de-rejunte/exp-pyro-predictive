"""Cliente AppEEARS: autenticação local, cache, tarefas retomáveis e downloads limitados."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlsplit

import requests

from .io import read_json, sha256_file, utc_now, write_json

BASE = "https://appeears.earthdatacloud.nasa.gov/api"
TOKEN_TARGET = "PYRO/APPEEARS_TOKEN"
# CDN observado no redirecionamento HTTPS do endpoint oficial AppEEARS.
DOWNLOAD_CDN_HOSTS = {"d2v96idf0qj40l.cloudfront.net"}


class AppEEARSError(RuntimeError):
    """Falha explícita sem incluir credenciais ou corpo de autenticação."""


class AuthenticationRequired(AppEEARSError):
    pass


def local_token() -> str | None:
    if os.environ.get("APPEEARS_TOKEN"):
        return os.environ["APPEEARS_TOKEN"]
    if os.name == "nt":
        try:
            import win32cred
            import pywintypes
        except ImportError:
            return None

        try:
            item = win32cred.CredRead(TOKEN_TARGET, win32cred.CRED_TYPE_GENERIC)
        except pywintypes.error as exc:
            if exc.winerror == 1168:
                return None
            raise AuthenticationRequired(
                "Não foi possível consultar a credencial AppEEARS local."
            ) from None
        blob = item["CredentialBlob"]
        return (
            blob.decode("utf-16-le").rstrip("\x00")
            if isinstance(blob, bytes)
            else str(blob)
        )
    return None


class AppEEARSClient:
    def __init__(
        self,
        cache: Path,
        *,
        token: str | None = None,
        timeout: float = 45,
        retries: int = 3,
        max_requests: int = 200,
        max_download_bytes: int = 200_000_000,
        session: requests.Session | None = None,
        guard=None,
    ):
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.token = token or local_token()
        self.session = session or requests.Session()
        self.timeout, self.retries = timeout, min(3, max(1, retries))
        self.max_requests, self.max_download_bytes = max_requests, max_download_bytes
        self.counter = self.cache / "requests.json"
        self.guard = guard

    def authenticate(self, username: str, password: str) -> None:
        if not username or not password:
            raise AuthenticationRequired(
                "Informe usuário e senha Earthdata no ambiente local."
            )
        response = self._request("POST", "login", auth=(username, password))
        token = response.json().get("token")
        if not isinstance(token, str) or not token:
            raise AuthenticationRequired(
                "A NASA não retornou um token AppEEARS válido."
            )
        self.token = token

    def _request(
        self, method: str, endpoint: str, *, protected: bool = False, **kwargs
    ):
        if ".." in endpoint or not re.fullmatch(r"[A-Za-z0-9_./-]+", endpoint):
            raise ValueError("Endpoint AppEEARS inválido.")
        if protected and not self.token:
            username, password = (
                os.environ.get("EARTHDATA_USERNAME"),
                os.environ.get("EARTHDATA_PASSWORD"),
            )
            if username and password:
                self.authenticate(username, password)
            else:
                raise AuthenticationRequired(
                    "Configure o token AppEEARS local; execute scripts/configurar_appeears.py."
                )
        headers = {"Authorization": "Bearer " + self.token} if protected else {}
        attempts = self.retries if method == "GET" else 1
        for attempt in range(attempts):
            if self.guard is not None:
                self.guard()
            state = (
                read_json(self.counter) if self.counter.exists() else {"requests": 0}
            )
            if state["requests"] >= self.max_requests:
                raise AppEEARSError("Orçamento de requisições AppEEARS atingido.")
            time.sleep(max(0, 1 - (time.time() - state.get("last_epoch", 0))))
            state.update(requests=state["requests"] + 1, last_epoch=time.time())
            write_json(self.counter, state)
            try:
                response = self.session.request(
                    method,
                    BASE + "/" + endpoint,
                    headers=headers,
                    timeout=self.timeout,
                    allow_redirects=False,
                    **kwargs,
                )
            except requests.RequestException:
                if attempt + 1 < attempts:
                    time.sleep(2**attempt)
                    continue
                raise AppEEARSError(
                    "Falha de rede AppEEARS; nenhuma credencial foi registrada."
                ) from None
            if response.status_code in (401, 403):
                raise AuthenticationRequired(
                    "Autenticação AppEEARS ausente, expirada ou sem autorização; gere um novo token."
                )
            if response.status_code == 429:
                raise AppEEARSError(
                    "AppEEARS limitou as requisições (HTTP 429); retome mais tarde."
                )
            if (
                method == "GET"
                and endpoint.startswith("bundle/")
                and kwargs.get("stream")
            ):
                current_url = BASE + "/" + endpoint
                for _ in range(3):
                    if response.status_code not in (301, 302, 303, 307, 308):
                        break
                    location = urljoin(
                        current_url, response.headers.get("Location", "")
                    )
                    parsed = urlsplit(location)
                    host = parsed.hostname or ""
                    if (
                        parsed.scheme != "https"
                        or parsed.username
                        or parsed.password
                        or not (
                            host.endswith(".nasa.gov")
                            or host.endswith(".amazonaws.com")
                            or host in DOWNLOAD_CDN_HOSTS
                        )
                    ):
                        response.close()
                        raise AppEEARSError(
                            "Redirecionamento de download fora dos destinos NASA/AWS/CDN autorizados."
                        )
                    response.close()
                    state = read_json(self.counter)
                    if state["requests"] >= self.max_requests:
                        raise AppEEARSError("Orçamento de download AppEEARS atingido.")
                    state["requests"] += 1
                    write_json(self.counter, state)
                    if self.guard:
                        self.guard()
                    try:
                        response = self.session.request(
                            "GET",
                            location,
                            headers=headers if location.startswith(BASE + "/") else {},
                            timeout=self.timeout,
                            allow_redirects=False,
                            stream=True,
                        )
                        current_url = location
                    except requests.RequestException:
                        raise AppEEARSError(
                            "Falha no download redirecionado; token não enviado ao armazenamento externo."
                        ) from None
            if response.status_code >= 500 and attempt + 1 < attempts:
                response.close()
                time.sleep(2**attempt)
                continue
            if not 200 <= response.status_code < 300:
                raise AppEEARSError(
                    f"AppEEARS retornou HTTP {response.status_code} em {endpoint.split('/')[0]}."
                )
            return response
        raise AppEEARSError("AppEEARS indisponível.")

    def metadata(self, product: str | None = None) -> Any:
        if product is not None and (
            ".." in product or not re.fullmatch(r"[A-Za-z0-9_.]+", product)
        ):
            raise ValueError("Identificador de produto inválido.")
        endpoint = "product" + ("/" + product if product else "")
        path = self.cache / (
            "products.json" if product is None else "layers_" + product + ".json"
        )
        if path.exists():
            envelope = read_json(path)
            return envelope["data"]
        data = self._request("GET", endpoint).json()
        write_json(
            path,
            {"source": BASE + "/" + endpoint, "retrieved_utc": utc_now(), "data": data},
        )
        return data

    def submit(self, payload: dict) -> str:
        key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        path = self.cache / "tasks" / (key + ".json")
        if path.exists():
            record = read_json(path)
            if record.get("task_id"):
                return record["task_id"]
            if record.get("state") != "authentication_failed":
                raise AppEEARSError(
                    "Submissão anterior sem confirmação. Confira no AppEEARS antes de repetir; evite tarefas duplicadas."
                )
        if not self.token and not (
            os.environ.get("EARTHDATA_USERNAME")
            and os.environ.get("EARTHDATA_PASSWORD")
        ):
            raise AuthenticationRequired(
                "Configure o AppEEARS antes de solicitar a extração."
            )
        record = {
            "request": payload,
            "submitted_utc": utc_now(),
            "state": "submission_pending",
        }
        write_json(path, record)
        try:
            response = self._request(
                "POST", "task", protected=True, json=payload
            ).json()
        except AuthenticationRequired:
            write_json(path, {**record, "state": "authentication_failed"})
            raise
        task_id = response.get("task_id")
        if not isinstance(task_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", task_id):
            raise AppEEARSError("Resposta de submissão sem task_id válido.")
        record.update(task_id=task_id, state="submitted")
        write_json(path, record)
        return task_id

    def task(self, task_id: str) -> dict:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", task_id):
            raise ValueError("Identificador de tarefa inválido.")
        return self._request("GET", "task/" + task_id, protected=True).json()

    def wait(self, task_id: str, *, max_seconds: float = 300) -> dict:
        deadline = time.monotonic() + max_seconds
        while True:
            state = self.task(task_id)
            status = state.get("status")
            # Não arquiva user_id, token ou outros identificadores da conta.
            write_json(
                self.cache / "status" / (task_id + ".json"),
                {"task_id": task_id, "status": status, "checked_utc": utc_now()},
            )
            if status == "done":
                return state
            if status in ("error", "failed", "cancelled", "expired"):
                raise AppEEARSError(
                    "Tarefa AppEEARS terminou com estado " + str(status)
                )
            if time.monotonic() >= deadline:
                return {"task_id": task_id, "status": "pending"}
            time.sleep(min(30, max(0, deadline - time.monotonic())))

    def download_csv(self, task_id: str) -> list[Path]:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", task_id):
            raise ValueError("Identificador de tarefa inválido.")
        bundle = self._request("GET", "bundle/" + task_id, protected=True).json()
        folder = self.cache / "downloads" / task_id
        folder.mkdir(parents=True, exist_ok=True)
        paths, total = [], 0
        for item in bundle.get("files", []):
            remote = item.get("file_name", "")
            if not remote.lower().endswith(".csv"):
                continue
            filename = Path(remote.replace("\\", "/")).name
            if not filename or filename in (".", ".."):
                raise AppEEARSError("Nome de arquivo inseguro.")
            path = folder / filename
            sidecar = path.with_suffix(".csv.json")
            if sidecar.exists() and path.exists():
                old = read_json(sidecar)
                if (
                    old["file_id"] != item["file_id"]
                    or sha256_file(path) != old["sha256"]
                    or (item.get("sha256") and item["sha256"] != old["sha256"])
                ):
                    raise AppEEARSError("CSV AppEEARS alterado.")
                paths.append(path)
                total += path.stat().st_size
                if total > self.max_download_bytes:
                    raise AppEEARSError(
                        "Cache excede o limite de download desta consulta."
                    )
                continue
            size = int(item.get("file_size", 0))
            if total + size > self.max_download_bytes:
                raise AppEEARSError("Limite de download AppEEARS atingido.")
            response = self._request(
                "GET",
                "bundle/" + task_id + "/" + str(item["file_id"]),
                protected=True,
                stream=True,
            )
            tmp = path.with_suffix(".csv.part")
            started, checked = time.monotonic(), time.monotonic()
            with response, tmp.open("wb") as output:
                for chunk in response.iter_content(65536):
                    if time.monotonic() - started > max(60, self.timeout * 3):
                        raise AppEEARSError(
                            "Tempo total do download atingido; temporário preservado."
                        )
                    if self.guard and time.monotonic() - checked >= 5:
                        self.guard()
                        checked = time.monotonic()
                    total += len(chunk)
                    if total > self.max_download_bytes:
                        raise AppEEARSError(
                            "Limite de download AppEEARS atingido; temporário preservado."
                        )
                    output.write(chunk)
            if item.get("sha256") and sha256_file(tmp) != item["sha256"]:
                raise AppEEARSError(
                    "Checksum do arquivo difere do bundle NASA; temporário preservado."
                )
            tmp.replace(path)
            write_json(
                sidecar,
                {
                    "task_id": task_id,
                    "file_id": item["file_id"],
                    "retrieved_utc": utc_now(),
                    "sha256": sha256_file(path),
                    "source": BASE + "/bundle/" + task_id,
                },
            )
            paths.append(path)
        return paths
