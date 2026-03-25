from __future__ import annotations

import argparse
from pathlib import Path

from GraphPrior.components.analysis import run_analysis
from GraphPrior.components.generate import run_generate
from GraphPrior.components.prior import run_prior
from GraphPrior.components.test import run_test
from GraphPrior.components.val import run_val
from GraphPrior.io import default_stage_run_id, read_json


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIFF_THRESHOLD = 1e-5


def _resolve_diff_threshold(project: str, override: float | None, config_path: str | None) -> float:
    if override is not None:
        return float(override)

    path = Path(config_path) if config_path is not None else ROOT / "global_config.json"
    try:
        payload = read_json(path)
        value = payload.get("testing", {}).get("diff_threshold", {}).get(project, DEFAULT_DIFF_THRESHOLD)
        return float(value)
    except Exception:
        return DEFAULT_DIFF_THRESHOLD


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full GraphPrior sample pipeline")
    parser.add_argument("--project", required=True, choices=["comet", "devmut", "muffin", "modelmeta"])
    parser.add_argument("--num-cases", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--seed-name", type=str, default="resnet50")
    parser.add_argument("--gen-run-id", "--run-id", dest="gen_run_id", type=str, default=None)
    parser.add_argument("--analysis-run-id", type=str, default=None)
    parser.add_argument("--test-run-id", type=str, default=None)
    parser.add_argument("--prior-run-id", type=str, default=None)
    parser.add_argument("--val-run-id", type=str, default=None)
    parser.add_argument("--diff-threshold", type=float, default=None)
    parser.add_argument("--global-config", type=str, default=None)
    parser.add_argument("--k3-sample-budget-per-node", type=int, default=16)
    parser.add_argument("--k3-sample-seed", type=int, default=2026)
    parser.add_argument("--k3-max-triplets", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    generation = run_generate(
        project=args.project,
        num_cases=args.num_cases,
        random_seed=args.seed,
        seed_name=args.seed_name,
        run_id=args.gen_run_id,
        resume=args.resume,
    )
    generation_run_id = generation.run_id
    analysis_run_id = args.analysis_run_id or default_stage_run_id("analysis", generation_run_id)
    test_run_id = args.test_run_id or default_stage_run_id("test", generation_run_id)
    prior_run_id = args.prior_run_id or default_stage_run_id("prior", generation_run_id)
    val_run_id = args.val_run_id or default_stage_run_id("val", generation_run_id)
    diff_threshold = _resolve_diff_threshold(args.project, args.diff_threshold, args.global_config)

    run_analysis(
        project=args.project,
        generation_run_id=generation_run_id,
        analysis_run_id=analysis_run_id,
        k3_sample_budget_per_node=args.k3_sample_budget_per_node,
        k3_sample_seed=args.k3_sample_seed,
        k3_max_triplets=args.k3_max_triplets,
    )
    run_test(
        project=args.project,
        generation_run_id=generation_run_id,
        test_run_id=test_run_id,
        diff_threshold=diff_threshold,
    )
    run_prior(
        project=args.project,
        generation_run_id=generation_run_id,
        test_run_id=test_run_id,
        prior_run_id=prior_run_id,
        k3_sample_budget_per_node=args.k3_sample_budget_per_node,
        k3_sample_seed=args.k3_sample_seed,
        k3_max_triplets=args.k3_max_triplets,
    )
    val = run_val(
        project=args.project,
        generation_run_id=generation_run_id,
        test_run_id=test_run_id,
        prior_run_id=prior_run_id,
        val_run_id=val_run_id,
    )
    print(
        {
            "project": args.project,
            "generation_run_id": generation_run_id,
            "analysis_run_id": analysis_run_id,
            "test_run_id": test_run_id,
            "prior_run_id": prior_run_id,
            "val_run_id": val_run_id,
            "diff_threshold": diff_threshold,
            "apfd": val.apfd,
            "apfdc": val.apfdc,
        }
    )


if __name__ == "__main__":
    main()
