from __future__ import annotations

import csv
from pathlib import Path

from GraphPrior.io import (
    default_stage_run_id,
    ensure_stage_run_dirs,
    ensure_test_case_report_dir,
    locate_generate_cases_dir,
    locate_generate_root,
    read_graph_model,
    read_json,
    write_bug_report,
    write_json,
    write_log,
)
from GraphPrior.projects.registry import get_project
from GraphPrior.types import BugReport, GraphModel


ROOT = Path(__file__).resolve().parents[1]


def _load_models(project: str, generation_run_id: str, case_ids: list[str] | None = None) -> list[GraphModel]:
    cases_dir = locate_generate_cases_dir(ROOT, project, generation_run_id)
    manifest_path = locate_generate_root(ROOT, project, generation_run_id) / "generation_manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    models: list[GraphModel] = []
    target = set(case_ids or [])
    for case_dir in sorted(cases_dir.iterdir() if cases_dir.exists() else []):
        if not case_dir.is_dir():
            continue
        case_id = case_dir.name
        if target and case_id not in target:
            continue
        gm_path = case_dir / "graph_model.json"
        if gm_path.exists():
            model = read_graph_model(gm_path)
            model.metadata.setdefault("_generation_run_id", generation_run_id)
            if "generation_random_seed" in manifest:
                model.metadata.setdefault("_generation_random_seed", manifest["generation_random_seed"])
            elif "random_seed" in manifest:
                model.metadata.setdefault("_generation_random_seed", manifest["random_seed"])
            models.append(model)
    return models


def run_test(
    project: str,
    generation_run_id: str,
    test_run_id: str | None = None,
    case_ids: list[str] | None = None,
    diff_threshold: float = 1e-5,
) -> list[BugReport]:
    test_run_id = test_run_id or default_stage_run_id("test", generation_run_id)
    run_cases_root, run_logs_root = ensure_stage_run_dirs(ROOT, "test", project, generation_run_id, test_run_id)
    log_path = run_logs_root / "test.log"
    write_log(log_path, f"[test] project={project} generation_run_id={generation_run_id} test_run_id={test_run_id}")

    adapter = get_project(project)
    models = _load_models(project, generation_run_id, case_ids)
    reports: list[BugReport] = []
    for model in models:
        report = adapter.detect_bug(model, models=models, diff_threshold=diff_threshold)
        reports.append(report)
        case_dir = ensure_test_case_report_dir(ROOT, project, generation_run_id, test_run_id, model.case_id)
        write_bug_report(case_dir / "bug_report.json", report)

    jsonl_path = run_cases_root / "test_results.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for report in reports:
            fh.write(
                "{\"case_id\":\"%s\",\"project\":\"%s\",\"has_bug\":%s,\"bug_type\":%s,\"runtime_s\":%.10f}\n"
                % (
                    report.case_id,
                    report.project,
                    "true" if report.has_bug else "false",
                    "null" if report.bug_type is None else '"%s"' % report.bug_type,
                    report.runtime_s,
                )
            )

    with (run_cases_root / "test_results.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["case_id", "project", "has_bug", "bug_type", "runtime_s"])
        writer.writeheader()
        for r in reports:
            writer.writerow({
                "case_id": r.case_id,
                "project": r.project,
                "has_bug": r.has_bug,
                "bug_type": r.bug_type,
                "runtime_s": r.runtime_s,
            })

    write_json(
        run_cases_root / "test_manifest.json",
        {
            "project": project,
            "generation_run_id": generation_run_id,
            "test_run_id": test_run_id,
            "num_cases": len(reports),
            "diff_threshold": diff_threshold,
            "selected_case_ids": case_ids,
        },
    )
    write_log(log_path, f"[test] completed tested={len(reports)}")
    return reports
