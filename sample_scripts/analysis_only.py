from __future__ import annotations

import argparse

from GraphPrior.components.analysis import run_analysis
from GraphPrior.io import default_stage_run_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Run analysis only")
    parser.add_argument("--project", required=True, choices=["comet", "devmut", "muffin", "modelmeta"])
    parser.add_argument("--gen-run-id", required=True)
    parser.add_argument("--analysis-run-id", default=None)
    parser.add_argument("--wl-h", type=int, default=2)
    parser.add_argument("--simhash-bits", type=int, default=64)
    parser.add_argument("--lsh-bands", type=int, default=8)
    parser.add_argument("--lsh-min-collisions", type=int, default=2)
    parser.add_argument("--k3-sample-budget-per-node", type=int, default=16)
    parser.add_argument("--k3-sample-seed", type=int, default=2026)
    parser.add_argument("--k3-max-triplets", type=int, default=None)
    args = parser.parse_args()

    analysis_run_id = args.analysis_run_id or default_stage_run_id("analysis", args.gen_run_id)
    records = run_analysis(
        project=args.project,
        generation_run_id=args.gen_run_id,
        analysis_run_id=analysis_run_id,
        wl_h=args.wl_h,
        simhash_bits=args.simhash_bits,
        lsh_bands=args.lsh_bands,
        lsh_min_collisions=args.lsh_min_collisions,
        k3_sample_budget_per_node=args.k3_sample_budget_per_node,
        k3_sample_seed=args.k3_sample_seed,
        k3_max_triplets=args.k3_max_triplets,
    )
    print(
        {
            "project": args.project,
            "generation_run_id": args.gen_run_id,
            "analysis_run_id": analysis_run_id,
            "analyzed": len(records),
        }
    )


if __name__ == "__main__":
    main()
