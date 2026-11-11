"""Procura uma credencial informada sem ecoar ou gravar seu conteúdo."""
from __future__ import annotations

import getpass
import json
from pathlib import Path


def contains(path: Path, needle: bytes) -> bool:
    carry = b""
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            data = carry + block
            if needle in data:
                return True
            carry = data[-(len(needle) - 1):]
    return False


def main() -> None:
    credential = getpass.getpass("Credencial para verificação local: ").encode()
    if len(credential) < 16:
        raise ValueError("Credencial vazia ou curta demais para verificação.")
    root = Path(__file__).resolve().parents[1]
    locations = list((root / "runs").glob("viirs_*")) + [root / "runs/processes_nasa_v3_20261111",
                 root / "docs", root / "src/earthfire", root / "scripts"]
    paths = [path for location in locations if location.exists() for path in location.rglob("*")
             if path.is_file() and path.suffix.lower() in {".json", ".log", ".md", ".py", ".txt"}]
    paths += [root / name for name in ("README.md", "STATUS.md", "OVERNIGHT_REPORT.md", "REPORT_VIIRS_V4.md",
                                      "runs/budget_nasa_v3_20261111.json")]
    found = [str(path.relative_to(root)) for path in paths if path.exists() and contains(path, credential)]
    print(json.dumps({"checked_files": len(paths), "matches": found}, ensure_ascii=False))


if __name__ == "__main__":
    main()
