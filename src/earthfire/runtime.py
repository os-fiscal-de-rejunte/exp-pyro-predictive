from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any

import psutil

from .io import read_json, utc_now, write_json


class ResourceLimit(RuntimeError):
    """Interrupção controlada por orçamento ou incompatibilidade de retomada."""


class Run:
    def __init__(self, config_path: Path, run_id: str, root: Path = Path(".")) -> None:
        if not run_id or any(x in run_id for x in ("/", "\\", "..")):
            raise ValueError("run_id inválido")
        self.root = root.resolve()
        self.config = read_json(config_path)
        self.run_id = run_id
        self.path = self.root / self.config["runs_dir"] / run_id
        self.path.mkdir(parents=True, exist_ok=True)
        # Um identificador compartilhado faz várias etapas obedecerem ao mesmo relógio.
        budget_id = os.environ.get("EARTHFIRE_BUDGET_ID", run_id)
        if not budget_id or any(x in budget_id for x in ("/", "\\", "..")):
            raise ValueError("EARTHFIRE_BUDGET_ID inválido")
        self.budget_path = self.root / self.config["runs_dir"] / ("budget_" + budget_id + ".json")
        self.manifest_path = self.path / "run.json"
        self.config_hash = hashlib.sha256(json.dumps(self.config, sort_keys=True).encode()).hexdigest()
        ops = self.config["operational"]
        os.environ["OMP_NUM_THREADS"] = str(ops["max_cpu_threads"])
        os.environ["MKL_NUM_THREADS"] = str(ops["max_cpu_threads"])
        if self.manifest_path.exists():
            self.manifest = read_json(self.manifest_path)
            if self.manifest["config_sha256"] != self.config_hash:
                raise ResourceLimit("Configuração mudou: inicie outro run_id.")
        else:
            available = psutil.virtual_memory().available
            free = shutil.disk_usage(self.root).free
            self.manifest = {
                "run_id": run_id, "started_utc": utc_now(), "started_epoch": time.time(),
                "config_sha256": self.config_hash, "config": self.config,
                "initial_available_ram": available, "memory_limit_bytes": int(available * ops["memory_fraction_initial_available"]),
                "initial_free_disk": free, "new_files_limit_bytes": int(min(free * ops["new_files_fraction_initial_free"], ops["new_files_max_bytes"])),
                "training_runs": 0, "stages": {}, "temporary_files": [],
            }
            self.save()
        if not self.budget_path.exists():
            available = psutil.virtual_memory().available
            free = shutil.disk_usage(self.root).free
            write_json(self.budget_path, {"budget_id": budget_id, "started_epoch": time.time(), "started_utc": utc_now(),
                "initial_available_ram": available, "memory_limit_bytes": int(available * ops["memory_fraction_initial_available"]),
                "initial_free_disk": free, "new_files_limit_bytes": int(min(free * ops["new_files_fraction_initial_free"], ops["new_files_max_bytes"])),
                "training_jobs": []})
        self.process_dir = self.root / self.config["runs_dir"] / ("processes_" + budget_id)
        process = psutil.Process()
        write_json(self.process_dir / f"{process.pid}.json", {"pid": process.pid, "create_time": process.create_time(), "run_id": run_id})
        self.logger = logging.getLogger("earthfire." + run_id)
        if not self.logger.handlers:
            h = logging.FileHandler(self.path / "pipeline.log", encoding="utf-8")
            h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            self.logger.addHandler(h)
            self.logger.setLevel(logging.INFO)

    def save(self) -> None:
        write_json(self.manifest_path, self.manifest)

    def guard(self, require_training_seconds: int = 0) -> None:
        ops = self.config["operational"]
        budget = read_json(self.budget_path)
        remaining = ops["total_seconds"] - (time.time() - budget["started_epoch"])
        if remaining <= ops["reserve_seconds"] + require_training_seconds:
            raise ResourceLimit("Reserva de encerramento atingida; checkpoint preservado.")
        # Conta apenas arquivos do run, sem tocar em DATA ou em arquivos pessoais.
        # Redução no espaço livre é conservadora e também inclui arquivos fora do run.
        new_disk = max(0, budget["initial_free_disk"] - shutil.disk_usage(self.root).free)
        if new_disk > budget["new_files_limit_bytes"]:
            raise ResourceLimit("Limite de arquivos da execução atingido.")
        rss = 0
        for record_path in self.process_dir.glob("*.json"):
            record = read_json(record_path)
            try:
                process = psutil.Process(record["pid"])
                if abs(process.create_time() - record["create_time"]) < .01:
                    rss += process.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        if rss > budget["memory_limit_bytes"]:
            raise ResourceLimit("Limite de memória dos processos registrados atingido.")

    def claim_training(self, variant: str) -> None:
        if self.run_id.startswith("quick_"):
            return
        budget = read_json(self.budget_path)
        job = self.run_id + ":" + variant
        if job not in budget["training_jobs"]:
            if len(budget["training_jobs"]) >= self.config["operational"]["max_training_runs"]:
                raise ResourceLimit("Limite compartilhado de treinamentos completos atingido.")
            budget["training_jobs"].append(job)
            write_json(self.budget_path, budget)

    def stage(self, name: str, state: str, **details: Any) -> None:
        self.manifest["stages"][name] = {"state": state, "updated_utc": utc_now(), **details}
        self.save()
        self.logger.info("stage=%s state=%s details=%s", name, state, details)
