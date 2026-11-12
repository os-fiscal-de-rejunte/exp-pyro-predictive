"""Expansão progressiva até esgotar dados, tempo, disco, memória ou orçamento de rede."""

from __future__ import annotations

import argparse
import gc
import os
import time
from pathlib import Path

import psutil

from .io import read_json, sha256_file, utc_now, write_json
from .runtime import ResourceLimit, Run


def progress(run: Run, state: dict) -> None:
    state["updated_utc"] = utc_now()
    write_json(run.path / "progress.json", state)
    budget = read_json(run.budget_path)
    remaining = run.config["operational"]["total_seconds"] - (
        time.time() - budget["started_epoch"]
    )
    text = f"""# Expansão máxima — em execução

- Atualização UTC: {state['updated_utc']}
- Etapa: {state.get('stage','coleta')}
- Blocos concluídos: {state.get('blocks',0)}
- Células com schema de features preparado: {state.get('cells',0)}; clima pode estar ausente.
- Candidatas sazonais com clima na validação (provisório): {state.get('provisional',0)}
- Área provisória: {state.get('provisional',0)*1443.3649192455174:,.1f} km²; ainda não é liberação final.
- Consultas de máscara: {state.get('mask_requests',0)}
- Tempo restante da janela: {max(0,remaining)/3600:.2f} h, incluindo reserva final.
- Checkpoints: `{run.path.relative_to(run.root)}` e `runs/maxcoverage_v6_20261111/mask_manifest.json`.
- Motivo de parada: {state.get('stop_reason','nenhum')}

O teste permanece reservado até a seleção final. O piloto v5 segue preservado.
"""
    (run.root / "EXPANSION_STATUS.md").write_text(text, encoding="utf-8")
    status_path = run.root / "STATUS.md"
    if status_path.exists():
        status = status_path.read_text(encoding="utf-8")
        begin, end = "<!-- V6_PROGRESS_BEGIN -->", "<!-- V6_PROGRESS_END -->"
        summary = f"{begin}\n\nÚltima atualização UTC: {state['updated_utc']}. Etapa: {state.get('stage')}. Blocos preparados: {state.get('blocks',0)}; células com features: {state.get('cells',0)}. Checkpoints: `{run.path.relative_to(run.root)}`. Dados provisórios; detalhes e recursos em `EXPANSION_STATUS.md` e `resources.json`.\n\n{end}"
        if begin in status and end in status:
            status = (
                status[: status.index(begin)]
                + summary
                + status[status.index(end) + len(end) :]
            )
        else:
            position = status.find("\n## Expansão v5")
            if position >= 0:
                status = status[:position] + "\n" + summary + "\n" + status[position:]
        status_path.write_text(status, encoding="utf-8")


def execute(config: Path, run_id: str, selection: Path) -> dict:
    os.environ["EARTHFIRE_BUDGET_ID"] = "maxcoverage_v6_20261111"
    # O primeiro lote foi iniciado em separado durante a implementação.
    while True:
        active = False
        for p in Path("runs/processes_maxcoverage_v6_20261111").glob("*.json"):
            record = read_json(p)
            if record["run_id"] != "maxcoverage_v6_20261111":
                continue
            try:
                process = psutil.Process(record["pid"])
                if abs(process.create_time() - record["create_time"]) < 0.01:
                    active = True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        if not active:
            break
        time.sleep(10)
    run = Run(config, run_id)
    # Define os limites antes de carregar os backends numéricos.
    from .expansion_collect import BlockCollector
    from .expansion_features import prepare_parent
    from .expansion_sparse_weather import SparseWeather

    plan = read_json(selection)
    write_json(run.path / "selection.json", plan)
    run.stage(
        "expansion_plan",
        "complete",
        blocks=len(plan["blocks"]),
        selection_sha256=sha256_file(selection),
    )
    collector = BlockCollector(run)
    weather = None
    seen = set()
    results = []
    state = {
        "stage": "coleta e preparação",
        "blocks": 0,
        "cells": 0,
        "provisional": 0,
        "mask_requests": collector.manifest["requests"],
    }
    progress(run, state)
    first_log = run.root / "runs/maxcoverage_v6_wave01.log"
    halted_rate = first_log.exists() and "HTTP 429" in first_log.read_text(
        encoding="utf-8", errors="replace"
    )
    try:
        weather = SparseWeather(run)
        size = run.config["expansion"]["wave_size"]
        for start in range(0, len(plan["blocks"]), size):
            run.guard(require_training_seconds=2400)
            parents = plan["blocks"][start : start + size]
            known = collector.manifest["partitions"]
            todo = [p for p in parents if p["name"] not in known]
            if halted_rate and todo:
                state["stop_reason"] = "NASA ORNL limitou a coleta (HTTP 429)"
                break
            # Reserva o tempo observado para completar o lote antes de iniciar outro.
            durations = [
                v.get("seconds", 0) for v in collector.manifest["files"].values()
            ]
            average = sum(durations) / max(1, len(durations))
            estimate = max(300, average * 30 * len(todo) / 2)
            if todo:
                run.guard(require_training_seconds=int(2400 + estimate))
            stop = None
            if todo:
                try:
                    collector.wave(todo)
                except ResourceLimit as exc:
                    stop = str(exc)
            for parent in parents:
                if parent["name"] not in collector.manifest["partitions"]:
                    continue
                item = prepare_parent(run, parent["name"], weather, seen)
                results.append(item)
                state.update(
                    blocks=len(results),
                    cells=len(seen),
                    provisional=sum(x.get("provisional", 0) for x in results),
                    mask_requests=collector.manifest["requests"],
                )
                progress(run, state)
                if item.get("weather_limit"):
                    stop = "Orçamento/recursos de meteorologia atingidos"
            gc.collect()
            if stop:
                state["stop_reason"] = stop
                break
        else:
            state["stop_reason"] = "Fila de candidatos elegíveis esgotada"
    except (ResourceLimit, KeyboardInterrupt) as exc:
        state["stop_reason"] = str(exc) or "interrupção solicitada"
    finally:
        collector.stop.set()
        if weather is not None:
            weather.store.session.close()
    del weather
    gc.collect()
    state["stage"] = "treinamento e avaliação final"
    progress(run, state)
    from .expansion_evaluate import finalize

    report = finalize(run)
    state["stage"] = "concluído"
    state["released"] = report.get("released_cells", 0)
    progress(run, state)
    run.stage("expansion", "complete", **state)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("config/maxcoverage_v6_runtime.json")
    )
    parser.add_argument("--run-id", default="maxcoverage_v6b_20261111")
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path("runs/maxcoverage_v6b_20261111_selection/selection.json"),
    )
    args = parser.parse_args()
    print(execute(args.config, args.run_id, args.selection), flush=True)


if __name__ == "__main__":
    main()
