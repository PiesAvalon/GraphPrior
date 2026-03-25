from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .types import AnalysisRecord, BugReport, GraphModel, GraphNode, PriorRecord, ValidationRecord


def default_stage_run_id(stage: str, source_run_id: str | None = None) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stage}_{ts}"


def ensure_generate_dirs(root: Path, project: str, run_id: str) -> tuple[Path, Path]:
    generate_root = root / "cases" / project / run_id
    logs_root = root / "logs" / "generate" / project / run_id
    generate_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)
    return generate_root, logs_root


def ensure_stage_run_dirs(root: Path, stage: str, project: str, generation_run_id: str, run_id: str) -> tuple[Path, Path]:
    del generation_run_id
    stage_root = root / "cases" / project / run_id
    logs_root = root / "logs" / stage / project / run_id
    stage_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)
    return stage_root, logs_root


def locate_generate_root(root: Path, project: str, run_id: str) -> Path:
    return root / "cases" / project / run_id


def locate_generate_cases_dir(root: Path, project: str, run_id: str) -> Path:
    return locate_generate_root(root, project, run_id) / "cases"


def locate_stage_run_root(
    root: Path,
    stage: str,
    project: str,
    generation_run_id: str,
    run_id: str | None = None,
) -> Path:
    if run_id is not None:
        return root / "cases" / project / run_id

    base = root / "cases" / project
    if not base.exists():
        return base / f"{stage}_missing"

    candidates: list[Path] = []
    for path in sorted(base.iterdir()):
        if not path.is_dir():
            continue
        if not path.name.startswith(f"{stage}_"):
            continue
        manifest_name = f"{stage}_manifest.json"
        manifest_path = path / manifest_name
        if manifest_path.exists():
            payload = read_json(manifest_path)
            if payload.get("generation_run_id") == generation_run_id:
                candidates.append(path)
                continue
        if generation_run_id in path.name:
            candidates.append(path)
    if candidates:
        return candidates[-1]
    return base / f"{stage}_missing"


def ensure_generate_case_dir(root: Path, project: str, run_id: str, case_id: str) -> Path:
    case_dir = root / "cases" / project / run_id / "cases" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    return case_dir


def ensure_test_case_report_dir(root: Path, project: str, generation_run_id: str, run_id: str, case_id: str) -> Path:
    del generation_run_id
    case_dir = root / "cases" / project / run_id / "reports" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    return case_dir


def write_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(message.rstrip() + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_graph_model(path: Path, model: GraphModel) -> None:
    payload = {
        "case_id": model.case_id,
        "project": model.project,
        "seed_name": model.seed_name,
        "parent_case_id": model.parent_case_id,
        "nodes": [asdict(node) for node in model.nodes],
        "metadata": model.metadata,
        "native_artifact_path": model.native_artifact_path,
    }
    write_json(path, payload)


def read_graph_model(path: Path) -> GraphModel:
    payload = read_json(path)
    nodes = [
        GraphNode(
            node_id=int(item["node_id"]),
            op_type=str(item["op_type"]),
            inputs=tuple(int(x) for x in item.get("inputs", [])),
            attrs=dict(item.get("attrs", {})),
        )
        for item in payload.get("nodes", [])
    ]
    return GraphModel(
        case_id=str(payload["case_id"]),
        project=str(payload["project"]),
        nodes=nodes,
        seed_name=payload.get("seed_name"),
        parent_case_id=payload.get("parent_case_id"),
        metadata=dict(payload.get("metadata", {})),
        native_artifact_path=payload.get("native_artifact_path"),
    )


def write_generation_manifest(path: Path, data: dict[str, Any]) -> None:
    write_json(path, data)


def write_bug_report(path: Path, report: BugReport) -> None:
    write_json(path, asdict(report))


def write_analysis_record(path: Path, records: list[AnalysisRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def write_prior_record(path: Path, records: list[PriorRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def write_validation_record(path: Path, record: ValidationRecord) -> None:
    write_json(path, asdict(record))
