"""Audita arquivos adicionados desde uma execução anterior, sem alterar DATA."""
from __future__ import annotations

import argparse
from pathlib import Path

from .audit import audit_archive
from .io import read_json, sha256_file, write_json
from .runtime import Run


def audit_added(run: Run, previous_inventory: Path) -> Path:
    prior = read_json(previous_inventory)
    if not prior.get("complete"):
        raise ValueError("O inventário anterior não está completo.")
    known = {entry["archive"]: entry["sha256"] for entry in prior["archives"]}
    paths = sorted((run.root / run.config["data_dir"]).glob("*.zip"))
    output = run.path / "audit_added"
    output.mkdir(parents=True, exist_ok=True)
    added = [p for p in paths if p.name not in known]
    missing = sorted(set(known) - {p.name for p in paths})
    if missing:
        raise ValueError(f"Arquivos do inventário anterior ausentes: {missing}")
    run.stage("audit_added", "running", new_archives=len(added))
    items = []
    for path in added:
        run.guard()
        digest = sha256_file(path)
        report = output / (path.stem + ".json")
        if report.exists():
            data = read_json(report)
            if data.get("sha256") != digest:
                raise ValueError("Arquivo adicionado mudou desde o checkpoint.")
        else:
            data = audit_archive(path, digest, run, full=True)
            write_json(report, data)
        items.append({"archive": path.name, "sha256": digest, "report": str(report), "rows": data["row_count"]})
        write_json(output / "inventory.json", {"complete": False, "archives": items})
    write_json(output / "inventory.json", {"complete": True, "archives": items,
        "prior_inventory": str(previous_inventory), "prior_archives_not_rehashed_in_this_stage": len(known)})
    run.stage("audit_added", "complete", archives=len(items), rows=sum(x["rows"] for x in items))
    return output / "inventory.json"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("config/default.json"))
    p.add_argument("--run-id", default="audit_zarr_v2_20261111")
    p.add_argument("--previous", type=Path, default=Path("runs/full_20261111/audit_full/inventory.json"))
    args = p.parse_args()
    print(audit_added(Run(args.config, args.run_id), args.previous))


if __name__ == "__main__":
    main()
