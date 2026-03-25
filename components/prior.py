from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from GraphPrior.io import (
    default_stage_run_id,
    ensure_stage_run_dirs,
    locate_generate_cases_dir,
    locate_stage_run_root,
    read_graph_model,
    write_json,
    write_log,
    write_prior_record,
)
from GraphPrior.types import PriorRecord

from ._evaluation_core import graphprior_order
from ._graphprior_core import graphprior_analysis
from ._legacy_types import NodeSpec, TestCase


ROOT = Path(__file__).resolve().parents[1]


def _normalize_value(value):
    if isinstance(value, list):
        return tuple(_normalize_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_normalize_value(item) for item in value)
    return value


def _load_models(project: str, generation_run_id: str):
    cases_dir = locate_generate_cases_dir(ROOT, project, generation_run_id)
    models = []
    for case_dir in sorted(cases_dir.iterdir() if cases_dir.exists() else []):
        gm_path = case_dir / "graph_model.json"
        if gm_path.exists():
            models.append(read_graph_model(gm_path))
    return models


def _parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _load_bug_flags(project: str, generation_run_id: str, test_run_id: str) -> dict[str, bool]:
    test_root = locate_stage_run_root(ROOT, "test", project, generation_run_id, test_run_id)
    test_csv = test_root / "test_results.csv"
    if not test_csv.exists():
        raise FileNotFoundError("test_results.csv is required before running prior in offline replay mode")

    bug_flags: dict[str, bool] = {}
    with test_csv.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            case_id = row.get("case_id")
            if not case_id:
                continue
            bug_flags[case_id] = _parse_bool(row.get("has_bug", "false"))
    return bug_flags


def _to_test_case(model):
    def _attrs_for(op: str, attrs: dict) -> dict:
        normalized = {key: _normalize_value(value) for key, value in attrs.items()}
        if op == "conv2d":
            normalized["out_channels"] = int(normalized.get("out_channels", 1))
            normalized["kernel_size"] = tuple(normalized.get("kernel_size", (1, 1)))
            normalized["stride"] = tuple(normalized.get("stride", (1, 1)))
            if "weight" in normalized:
                normalized["weight"] = np.asarray(normalized["weight"], dtype=np.float32)
            return normalized
        if op == "linear":
            normalized["out_dim"] = int(normalized.get("out_dim", 1))
            if "weight" in normalized:
                normalized["weight"] = np.asarray(normalized["weight"], dtype=np.float32)
            return normalized
        if op == "batchnorm":
            normalized["num_features"] = int(normalized.get("num_features", 1))
            return normalized
        if op == "maxpool2d":
            normalized["kernel_size"] = tuple(normalized.get("kernel_size", (1, 1)))
            normalized["stride"] = tuple(normalized.get("stride", (1, 1)))
            return normalized
        return normalized

    nodes = [
        NodeSpec(
            node_id=node.node_id,
            op=node.op_type.split(":", 1)[-1],
            inputs=tuple(node.inputs),
            attrs=_attrs_for(node.op_type.split(":", 1)[-1], dict(node.attrs)),
        )
        for node in model.nodes
    ]
    return TestCase(
        case_id=model.case_id,
        project=model.project,
        mutation_depth=int(model.metadata.get("mutation_depth", 1)),
        input_shape=tuple(model.metadata.get("input_shape", (64, 64, 3))),
        batch_size=int(model.metadata.get("batch_size", 1)),
        nodes=nodes,
        parent_case_id=model.parent_case_id,
        metadata=model.metadata,
    )


def run_prior(
    project: str,
    generation_run_id: str,
    test_run_id: str,
    prior_run_id: str | None = None,
    k3_sample_budget_per_node: int = 16,
    k3_sample_seed: int = 2026,
    k3_max_triplets: int | None = None,
) -> list[PriorRecord]:
    prior_run_id = prior_run_id or default_stage_run_id("prior", generation_run_id)
    run_cases_root, run_logs_root = ensure_stage_run_dirs(ROOT, "prior", project, generation_run_id, prior_run_id)
    log_path = run_logs_root / "prior.log"
    write_log(
        log_path,
        f"[prior] project={project} generation_run_id={generation_run_id} test_run_id={test_run_id} prior_run_id={prior_run_id}",
    )

    models = _load_models(project, generation_run_id)
    cases = [_to_test_case(m) for m in models]
    if not cases:
        write_log(log_path, "[prior] no cases found")
        return []

    bug_flags = _load_bug_flags(project, generation_run_id, test_run_id)
    _, scores, _, clusters = graphprior_analysis(
        cases,
        k3_sample_budget_per_node=k3_sample_budget_per_node,
        k3_sample_seed=k3_sample_seed,
        k3_max_triplets=k3_max_triplets,
    )
    case_ids = [c.case_id for c in cases]
    missing_bug_flags = [cid for cid in case_ids if cid not in bug_flags]
    if missing_bug_flags:
        missing = ", ".join(missing_bug_flags[:10])
        raise ValueError(f"Missing bug flags for {len(missing_bug_flags)} cases: {missing}")

    order = graphprior_order(
        case_ids=case_ids,
        scores=scores,
        clusters=clusters,
        bug_flags=bug_flags,
    )
    write_log(log_path, f"[prior] using_real_bug_flags count={len(case_ids)}")

    prior_records: list[PriorRecord] = []
    for idx, cid in enumerate(order, start=1):
        prior_records.append(PriorRecord(case_id=cid, project=project, rank=idx, score=float(scores.get(cid, 0.0))))

    write_prior_record(run_cases_root / "prior_order.json", prior_records)
    with (run_cases_root / "prior_order.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["case_id", "project", "rank", "score"])
        writer.writeheader()
        for r in prior_records:
            writer.writerow({"case_id": r.case_id, "project": r.project, "rank": r.rank, "score": r.score})

    with (run_cases_root / "prior_order_list.json").open("w", encoding="utf-8") as fh:
        json.dump(order, fh, indent=2)

    write_json(
        run_cases_root / "prior_manifest.json",
        {
            "project": project,
            "generation_run_id": generation_run_id,
            "test_run_id": test_run_id,
            "prior_run_id": prior_run_id,
            "num_cases": len(prior_records),
            "k3_sample_budget_per_node": k3_sample_budget_per_node,
            "k3_sample_seed": k3_sample_seed,
            "k3_max_triplets": k3_max_triplets,
        },
    )
    write_log(log_path, f"[prior] completed ranked={len(prior_records)}")
    return prior_records
