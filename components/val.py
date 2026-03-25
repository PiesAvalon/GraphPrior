from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

from GraphPrior.io import (
    default_stage_run_id,
    ensure_stage_run_dirs,
    locate_stage_run_root,
    read_json,
    write_json,
    write_log,
)
from GraphPrior.types import ValidationRecord

from ._evaluation_core import baseline_orders, evaluate_order
from ._graphprior_core import graphprior_analysis
from .prior import _load_models, _to_test_case


ROOT = Path(__file__).resolve().parents[1]

METHOD_DISPLAY = {
    "graphprior": "GraphPrior",
    "original": "Original Order",
    "random": "Random",
    "coverage_greedy": "Coverage-Greedy",
    "coverage_only": "Coverage-Only",
    "structure_only": "Structure-Only",
}


def _bugs_in_prefix(order: list[str], bug_flags: dict[str, bool], k: int) -> int:
    if k <= 0:
        return 0
    return int(sum(1 for cid in order[: min(k, len(order))] if bug_flags.get(cid, False)))


def run_val(
    project: str,
    generation_run_id: str,
    test_run_id: str,
    prior_run_id: str,
    val_run_id: str | None = None,
) -> ValidationRecord:
    val_run_id = val_run_id or default_stage_run_id("val", generation_run_id)
    run_cases_root, run_logs_root = ensure_stage_run_dirs(ROOT, "val", project, generation_run_id, val_run_id)
    log_path = run_logs_root / "val.log"
    write_log(
        log_path,
        f"[val] project={project} generation_run_id={generation_run_id} test_run_id={test_run_id} prior_run_id={prior_run_id} val_run_id={val_run_id}",
    )

    prior_root = locate_stage_run_root(ROOT, "prior", project, generation_run_id, prior_run_id)
    prior_csv = prior_root / "prior_order.csv"
    test_csv = locate_stage_run_root(ROOT, "test", project, generation_run_id, test_run_id) / "test_results.csv"
    if not prior_csv.exists() or not test_csv.exists():
        raise FileNotFoundError("prior_order.csv and test_results.csv are required before running val")

    graphprior_order: list[str] = []
    score_by_case: dict[str, float] = {}
    with prior_csv.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            cid = row["case_id"]
            graphprior_order.append(cid)
            score_by_case[cid] = float(row.get("score", 0.0))

    bug_flags: dict[str, bool] = {}
    costs: dict[str, float] = {}
    with test_csv.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            cid = row["case_id"]
            bug_flags[cid] = row["has_bug"].lower() in {"1", "true", "yes"}
            costs[cid] = float(row["runtime_s"])

    models = _load_models(project, generation_run_id)
    cases = [_to_test_case(m) for m in models]
    if not cases:
        raise FileNotFoundError("No generated cases found to build baseline orders for validation")

    case_ids = [c.case_id for c in cases]

    manifest_path = prior_root / "prior_manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        k3_budget = int(manifest.get("k3_sample_budget_per_node", 16))
        k3_seed = int(manifest.get("k3_sample_seed", 2026))
        k3_max_triplets = manifest.get("k3_max_triplets")
    else:
        k3_budget = 16
        k3_seed = 2026
        k3_max_triplets = None

    _, scores, _, clusters = graphprior_analysis(
        cases,
        k3_sample_budget_per_node=k3_budget,
        k3_sample_seed=k3_seed,
        k3_max_triplets=k3_max_triplets,
    )

    # Keep score source consistent with the prior stage output.
    for cid, score in score_by_case.items():
        scores[cid] = float(score)

    method_orders = {
        "graphprior": graphprior_order,
        **baseline_orders(
            case_ids=case_ids,
            scores=scores,
            clusters=clusters,
            random_seed=2026,
        ),
    }

    method_metrics: dict[str, dict[str, object]] = {}
    for method_name, order in method_orders.items():
        m = evaluate_order(name=method_name, order=order, bug_flags=bug_flags, costs=costs)
        method_metrics[method_name] = {
            "display_name": METHOD_DISPLAY.get(method_name, method_name),
            "apfd": float(m.apfd),
            "apfdc": float(m.apfdc),
            "bugs_total": int(m.bugs_total),
            "topk_budget": {
                "top10pct": int(m.bugs_at_10),
                "top20pct": int(m.bugs_at_20),
                "top30pct": int(m.bugs_at_30),
                "top50pct": int(m.bugs_at_50),
            },
            "topk_cases": {
                "top10": _bugs_in_prefix(order, bug_flags, 10),
                "top20": _bugs_in_prefix(order, bug_flags, 20),
                "top30": _bugs_in_prefix(order, bug_flags, 30),
                "top50": _bugs_in_prefix(order, bug_flags, 50),
            },
        }

    apfd = float(method_metrics["graphprior"]["apfd"])
    apfdc = float(method_metrics["graphprior"]["apfdc"])

    bugs_total = sum(1 for cid in graphprior_order if bug_flags.get(cid, False))
    cum_bug = 0
    cum_time = 0.0
    curve: list[dict[str, float]] = []
    for idx, cid in enumerate(graphprior_order, start=1):
        cum_time += costs.get(cid, 0.0)
        if bug_flags.get(cid, False):
            cum_bug += 1
        curve.append(
            {
                "rank": float(idx),
                "cum_time": float(cum_time),
                "cum_bug": float(cum_bug),
                "cum_bug_ratio": float(cum_bug / bugs_total) if bugs_total > 0 else 0.0,
            }
        )

    plt.figure(figsize=(7, 4))
    plt.plot([c["cum_time"] for c in curve], [c["cum_bug"] for c in curve], marker="o", linewidth=1.5)
    plt.xlabel("Cumulative Runtime (s)")
    plt.ylabel("Cumulative Bugs Found")
    plt.title(f"Bug Discovery Curve - {project} ({val_run_id})")
    plt.tight_layout()
    plt.savefig(run_cases_root / "bug_curve.png", dpi=160)
    plt.close()

    record = ValidationRecord(project=project, run_id=val_run_id, apfd=apfd, apfdc=apfdc, bugs_found_over_time=curve)
    write_json(
        run_cases_root / "val_metrics.json",
        {
            "project": record.project,
            "run_id": record.run_id,
            "apfd": record.apfd,
            "apfdc": record.apfdc,
            "bugs_found_over_time": record.bugs_found_over_time,
            "metrics_by_method": method_metrics,
        },
    )

    with (run_cases_root / "val_metrics.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["project", "run_id", "apfd", "apfdc"])
        writer.writeheader()
        writer.writerow({"project": project, "run_id": val_run_id, "apfd": apfd, "apfdc": apfdc})

    with (run_cases_root / "val_metrics_by_method.csv").open("w", encoding="utf-8", newline="") as fh:
        fieldnames = [
            "project",
            "run_id",
            "method",
            "display_name",
            "apfd",
            "apfdc",
            "bugs_total",
            "top10pct",
            "top20pct",
            "top30pct",
            "top50pct",
            "top10",
            "top20",
            "top30",
            "top50",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for method_name, metrics in method_metrics.items():
            budget = metrics["topk_budget"]
            fixed = metrics["topk_cases"]
            writer.writerow(
                {
                    "project": project,
                    "run_id": val_run_id,
                    "method": method_name,
                    "display_name": metrics["display_name"],
                    "apfd": metrics["apfd"],
                    "apfdc": metrics["apfdc"],
                    "bugs_total": metrics["bugs_total"],
                    "top10pct": budget["top10pct"],
                    "top20pct": budget["top20pct"],
                    "top30pct": budget["top30pct"],
                    "top50pct": budget["top50pct"],
                    "top10": fixed["top10"],
                    "top20": fixed["top20"],
                    "top30": fixed["top30"],
                    "top50": fixed["top50"],
                }
            )

    with (run_cases_root / "bugs_over_time.jsonl").open("w", encoding="utf-8") as fh:
        for item in curve:
            fh.write(json.dumps(item) + "\n")

    write_json(run_cases_root / "orders_by_method.json", method_orders)

    write_json(
        run_cases_root / "val_manifest.json",
        {
            "project": project,
            "generation_run_id": generation_run_id,
            "test_run_id": test_run_id,
            "prior_run_id": prior_run_id,
            "val_run_id": val_run_id,
            "apfd": apfd,
            "apfdc": apfdc,
            "methods": list(method_orders.keys()),
            "random_seed": 2026,
            "k3_sample_budget_per_node": k3_budget,
            "k3_sample_seed": k3_seed,
            "k3_max_triplets": k3_max_triplets,
        },
    )
    write_log(log_path, f"[val] completed apfd={apfd:.6f} apfdc={apfdc:.6f}")
    for method_name, metrics in method_metrics.items():
        fixed = metrics["topk_cases"]
        write_log(
            log_path,
            (
                f"[val][{method_name}] apfd={float(metrics['apfd']):.6f} apfdc={float(metrics['apfdc']):.6f} "
                f"top10={fixed['top10']} top20={fixed['top20']} top30={fixed['top30']} top50={fixed['top50']}"
            ),
        )
    return record
