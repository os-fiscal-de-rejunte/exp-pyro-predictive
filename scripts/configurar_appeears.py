"""Gera token AppEEARS e armazena somente o token no cofre Windows."""

from __future__ import annotations

import getpass
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earthfire.appeears import AppEEARSClient, TOKEN_TARGET  # noqa: E402


def main() -> None:
    username = input("Usuário NASA Earthdata: ").strip()
    password = getpass.getpass("Senha NASA Earthdata (oculta): ")
    session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    client = AppEEARSClient(
        ROOT / "runs/appeears_auth" / session_id, token="", max_requests=3
    )
    try:
        client.authenticate(username, password)
    finally:
        password = ""
    import win32cred

    win32cred.CredWrite(
        {
            "Type": win32cred.CRED_TYPE_GENERIC,
            "TargetName": TOKEN_TARGET,
            "UserName": "PYRO",
            "CredentialBlob": client.token,
            "Persist": win32cred.CRED_PERSIST_LOCAL_MACHINE,
        },
        0,
    )
    print(
        "Token AppEEARS armazenado no cofre Windows. Senha não foi salva. O token não foi exibido."
    )


if __name__ == "__main__":
    main()
