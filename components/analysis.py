from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from GraphPrior.io import (
    default_stage_run_id,
    ensure_stage_run_dirs,
    locate_generate_cases_dir,
    read_graph_model,
    write_analysis_record,
    write_json,
    write_log,
)
from GraphPrior.types import AnalysisRecord, GraphModel

from ._graphprior_core import graphprior_analysis
from ._legacy_types import NodeSpec, TestCase


ROOT = Path(__file__).resolve().parents[1]


def _normalize_value(value):
    if isinstance(value, list):
        return tuple(_normalize_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_normalize_value(item) for item in value)
    return value


def _load_models(project: str, generation_run_id: str) -> list[GraphModel]:
    cases_dir = locate_generate_cases_dir(ROOT, project, generation_run_id)
    models: list[GraphModel] = []
    if not cases_dir.exists():
        return models
    for case_dir in sorted(cases_dir.iterdir()):
        if not case_dir.is_dir():
            continue
        gm_path = case_dir / "graph_model.json"
        if gm_path.exists():
            models.append(read_graph_model(gm_path))
    return models


def _to_test_case(model: GraphModel) -> TestCase:
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

    nodes: list[NodeSpec] = []
    for node in model.nodes:
        op = node.op_type.split(":", 1)[-1]
        nodes.append(
            NodeSpec(
                node_id=node.node_id,
                op=op,
                inputs=tuple(node.inputs),
                attrs=_attrs_for(op, dict(node.attrs)),
            )
        )
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


def run_analysis(
    project: str,
    generation_run_id: str,
    analysis_run_id: str | None = None,
    wl_h: int = 2,
    simhash_bits: int = 64,
    lsh_bands: int = 8,
    lsh_min_collisions: int = 2,
    k3_sample_budget_per_node: int = 16,
    k3_sample_seed: int = 2026,
    k3_max_triplets: int | None = None,
) -> list[AnalysisRecord]:
    analysis_run_id = analysis_run_id or default_stage_run_id("analysis", generation_run_id)
    run_cases_root, run_logs_root = ensure_stage_run_dirs(ROOT, "analysis", project, generation_run_id, analysis_run_id)
    log_path = run_logs_root / "analysis.log"
    write_log(log_path, f"[analysis] project={project} generation_run_id={generation_run_id} analysis_run_id={analysis_run_id}")

    models = _load_models(project, generation_run_id)
    cases = [_to_test_case(m) for m in models]
    if not cases:
        write_log(log_path, "[analysis] no cases found")
        return []

    _, scores, coverage_metrics, clusters = graphprior_analysis(
        cases=cases,
        wl_h=wl_h,
        simhash_bits=simhash_bits,
        lsh_bands=lsh_bands,
        lsh_min_collisions=lsh_min_collisions,
        k3_sample_budget_per_node=k3_sample_budget_per_node,
        k3_sample_seed=k3_sample_seed,
        k3_max_triplets=k3_max_triplets,
    )

    cluster_size = {}
    for cluster in clusters:
        for cid in cluster:
            cluster_size[cid] = len(cluster)

    records: list[AnalysisRecord] = []
    for case in cases:
        metrics = coverage_metrics[case.case_id]
        structure = {
            "graphprior_score": float(scores[case.case_id]),
            "cluster_size": float(cluster_size.get(case.case_id, 1)),
            "node_count": float(len(case.nodes)),
        }
        records.append(
            AnalysisRecord(
                case_id=case.case_id,
                project=project,
                coverage_metrics={k: float(v) for k, v in metrics.items()},
                structure_metrics=structure,
            )
        )

    write_analysis_record(run_cases_root / "analysis.jsonl", records)
    with (run_cases_root / "analysis_summary.csv").open("w", encoding="utf-8", newline="") as fh:
        metric_keys = sorted(records[0].coverage_metrics.keys()) if records else []
        struct_keys = sorted(records[0].structure_metrics.keys()) if records else []
        fieldnames = ["case_id", "project", *metric_keys, *struct_keys]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {"case_id": record.case_id, "project": record.project}
            row.update(record.coverage_metrics)
            row.update(record.structure_metrics)
            writer.writerow(row)

    write_json(
        run_cases_root / "analysis_manifest.json",
        {
            "project": project,
            "generation_run_id": generation_run_id,
            "analysis_run_id": analysis_run_id,
            "num_cases": len(records),
            "wl_h": wl_h,
            "simhash_bits": simhash_bits,
            "lsh_bands": lsh_bands,
            "lsh_min_collisions": lsh_min_collisions,
            "k3_sample_budget_per_node": k3_sample_budget_per_node,
            "k3_sample_seed": k3_sample_seed,
            "k3_max_triplets": k3_max_triplets,
        },
    )
    write_log(log_path, f"[analysis] completed analyzed={len(records)}")
    return records
