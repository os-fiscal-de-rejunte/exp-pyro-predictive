"""Credenciais locais do usuário para consultas opcionais ao FIRMS."""
from __future__ import annotations

import argparse
import getpass
import hmac
import os
import sys

TARGET = "PYRO/FIRMS_MAP_KEY"


def _windows_backend():
    if sys.platform != "win32":
        return None
    try:
        import pywintypes
        import win32cred
    except ImportError:
        return None
    return win32cred, pywintypes


def read_windows_firms_key() -> str | None:
    backend = _windows_backend()
    if backend is None:
        return None
    win32cred, pywintypes = backend
    try:
        record = win32cred.CredRead(TARGET, win32cred.CRED_TYPE_GENERIC)
    except pywintypes.error as exc:
        if exc.winerror == 1168:
            return None
        raise RuntimeError(f"Não foi possível ler a credencial Windows (código {exc.winerror}).") from None
    blob = record["CredentialBlob"]
    return blob.decode("utf-16-le").rstrip("\x00") if isinstance(blob, bytes) else str(blob)


def load_firms_key() -> str | None:
    return os.environ.get("FIRMS_MAP_KEY") or read_windows_firms_key()


def store_firms_key(key: str) -> None:
    key = key.strip()
    if not key.isalnum() or len(key) > 128:
        raise ValueError("Formato de chave FIRMS inválido.")
    backend = _windows_backend()
    if backend is None:
        raise RuntimeError("Gerenciador de Credenciais Windows ou pywin32 indisponível.")
    previous = read_windows_firms_key()
    if previous is not None:
        if hmac.compare_digest(previous, key):
            return
        raise RuntimeError("Já existe outra chave FIRMS armazenada; nenhuma alteração foi feita.")
    win32cred, _ = backend
    win32cred.CredWrite({"Type": win32cred.CRED_TYPE_GENERIC, "TargetName": TARGET,
                        "UserName": "PYRO", "CredentialBlob": key,
                        "Persist": win32cred.CRED_PERSIST_LOCAL_MACHINE,
                        "Comment": "Chave NASA FIRMS para PYRO"}, 0)
    saved = read_windows_firms_key()
    if saved is None or not hmac.compare_digest(saved, key):
        raise RuntimeError("Não foi possível verificar a credencial armazenada.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gerencia credenciais locais do PYRO.")
    parser.add_argument("action", choices=["store-firms"])
    parser.parse_args()
    store_firms_key(getpass.getpass("Chave FIRMS: "))
    print("Chave armazenada no Gerenciador de Credenciais do Windows.")


if __name__ == "__main__":
    main()
