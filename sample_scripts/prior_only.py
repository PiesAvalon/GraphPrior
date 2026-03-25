from __future__ import annotations

import argparse

from GraphPrior.components.prior import run_prior
from GraphPrior.io import default_stage_run_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Run prioritization only")
    parser.add_argument("--project", required=True, choices=["comet", "devmut", "muffin", "modelmeta"])
    parser.add_argument("--gen-run-id", required=True)
    parser.add_argument("--test-run-id", required=True)
    parser.add_argument("--prior-run-id", default=None)
    parser.add_argument("--k3-sample-budget-per-node", type=int, default=16)
    parser.add_argument("--k3-sample-seed", type=int, default=2026)
    parser.add_argument("--k3-max-triplets", type=int, default=None)
    args = parser.parse_args()

    prior_run_id = args.prior_run_id or default_stage_run_id("prior", args.gen_run_id)
    records = run_prior(
        project=args.project,
        generation_run_id=args.gen_run_id,
        test_run_id=args.test_run_id,
        prior_run_id=prior_run_id,
        k3_sample_budget_per_node=args.k3_sample_budget_per_node,
        k3_sample_seed=args.k3_sample_seed,
        k3_max_triplets=args.k3_max_triplets,
    )
    print(
        {
            "project": args.project,
            "generation_run_id": args.gen_run_id,
            "test_run_id": args.test_run_id,
            "prior_run_id": prior_run_id,
            "ranked": len(records),
        }
    )


if __name__ == "__main__":
    main()
