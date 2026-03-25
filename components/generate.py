from __future__ import annotations

import shutil
from pathlib import Path

from GraphPrior.io import (
    default_stage_run_id,
    ensure_generate_case_dir,
    ensure_generate_dirs,
    locate_generate_cases_dir,
    read_graph_model,
    read_json,
    write_generation_manifest,
    write_graph_model,
    write_json,
    write_log,
)
from GraphPrior.projects.registry import get_project
from GraphPrior.types import GenerationRequest, GenerationResult, GraphModel


ROOT = Path(__file__).resolve().parents[1]


def _default_run_id(project: str) -> str:
    del project
    return default_stage_run_id("gen")


def _case_index(project: str, case_id: str) -> int:
    prefix = f"{project}_"
    if not case_id.startswith(prefix):
        raise ValueError(f"Unexpected case id for {project}: {case_id}")
    return int(case_id[len(prefix):])


def _load_existing_models(root: Path, project: str, run_id: str) -> list[GraphModel]:
    cases_dir = locate_generate_cases_dir(root, project, run_id)
    models: list[GraphModel] = []
    if not cases_dir.exists():
        return models

    for case_dir in cases_dir.iterdir():
        if not case_dir.is_dir():
            continue
        graph_path = case_dir / "graph_model.json"
        if graph_path.exists():
            models.append(read_graph_model(graph_path))

    models.sort(key=lambda model: _case_index(project, model.case_id))
    for expected, model in enumerate(models):
        actual = _case_index(project, model.case_id)
        if actual != expected:
            raise ValueError(
                f"Cannot resume run {run_id}: expected contiguous case ids but found {model.case_id} at position {expected}"
            )
    return models


def _validate_resume_inputs(
    project: str,
    run_id: str,
    request: GenerationRequest,
    existing_models: list[GraphModel],
    manifest_path: Path,
) -> None:
    if len(existing_models) > request.num_cases:
        raise ValueError(
            f"Cannot resume run {run_id}: existing cases ({len(existing_models)}) exceed requested total ({request.num_cases})"
        )

    if not manifest_path.exists():
        return

    manifest = read_json(manifest_path)
    expected = {
        "project": project,
        "run_id": run_id,
        "seed_name": request.seed_name,
        "random_seed": request.random_seed,
        "hyper_params": request.hyper_params,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(
                f"Cannot resume run {run_id}: manifest field {key!r}={manifest.get(key)!r} does not match request {value!r}"
            )


def _write_manifest(run_cases_root: Path, request: GenerationRequest, case_ids: list[str]) -> None:
    manifest = {
        "project": request.project,
        "run_id": request.run_id,
        "num_cases": len(case_ids),
        "target_num_cases": request.num_cases,
        "seed_name": request.seed_name,
        "random_seed": request.random_seed,
        "hyper_params": request.hyper_params,
        "case_ids": case_ids,
    }
    write_generation_manifest(run_cases_root / "generation_manifest.json", manifest)


def _normalize_native_layout(case_dir: Path, model: GraphModel) -> GraphModel:
    native_dir = case_dir / "native"
    native_dir.mkdir(parents=True, exist_ok=True)
    if not model.native_artifact_path:
        return model
    src = Path(model.native_artifact_path)
    if not src.exists():
        return model
    try:
        src.relative_to(case_dir)
        return model
    except ValueError:
        pass
    dst = native_dir / src.name
    if src.resolve() != dst.resolve():
        if src.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    model.native_artifact_path = str(dst)
    model.metadata.setdefault("native", {})
    model.metadata["native"]["artifact_path"] = str(dst)
    return model


def _persist_case(root: Path, project: str, run_id: str, model: GraphModel) -> GraphModel:
    case_dir = ensure_generate_case_dir(root, project, run_id, model.case_id)
    model = _normalize_native_layout(case_dir, model)
    write_graph_model(case_dir / "graph_model.json", model)
    write_json(case_dir / "metadata.json", model.metadata)
    return model


def run_generate(
    project: str,
    num_cases: int,
    random_seed: int,
    seed_name: str | None = "resnet50",
    run_id: str | None = None,
    hyper_params: dict | None = None,
    resume: bool = False,
) -> GenerationResult:
    run_id = run_id or _default_run_id(project)
    req = GenerationRequest(
        project=project,
        run_id=run_id,
        num_cases=num_cases,
        random_seed=random_seed,
        seed_name=seed_name,
        hyper_params=hyper_params or {},
    )

    run_cases_root, run_logs_root = ensure_generate_dirs(ROOT, project, run_id)
    log_path = run_logs_root / "generate.log"
    write_log(log_path, f"[generate] project={project} run_id={run_id} num_cases={num_cases} resume={resume}")

    output_root = run_cases_root / "cases"
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = run_cases_root / "generation_manifest.json"
    existing_models = _load_existing_models(ROOT, project, run_id)
    if existing_models and not resume:
        raise ValueError(
            f"Run {run_id} already has {len(existing_models)} persisted cases. Re-run with resume=True to continue."
        )

    _validate_resume_inputs(project, run_id, req, existing_models, manifest_path)
    if len(existing_models) == req.num_cases:
        write_log(log_path, f"[generate] nothing_to_do existing={len(existing_models)}")
        _write_manifest(run_cases_root, req, [model.case_id for model in existing_models])
        return GenerationResult(project=project, run_id=run_id, cases=existing_models)

    adapter = get_project(project)
    _write_manifest(run_cases_root, req, [model.case_id for model in existing_models])

    models = list(existing_models)
    for model in adapter.generate_models(req, output_root=output_root, existing_models=existing_models):
        persisted = _persist_case(ROOT, project, run_id, model)
        models.append(persisted)
        existing_models.append(persisted)
        _write_manifest(run_cases_root, req, [item.case_id for item in models])
        write_log(log_path, f"[generate] persisted case_id={persisted.case_id} total={len(models)}")

    if len(models) != req.num_cases:
        raise RuntimeError(
            f"Generation stopped early for run {run_id}: expected {req.num_cases} cases but found {len(models)}"
        )

    write_log(log_path, f"[generate] completed generated={len(models)}")
    return GenerationResult(project=project, run_id=run_id, cases=models)
