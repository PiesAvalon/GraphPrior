from __future__ import annotations

import argparse

from GraphPrior.components.generate import run_generate


def main() -> None:
    parser = argparse.ArgumentParser(description="Run generation only")
    parser.add_argument("--project", required=True, choices=["comet", "devmut", "muffin", "modelmeta"])
    parser.add_argument("--num-cases", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--seed-name", type=str, default="resnet50")
    parser.add_argument("--gen-run-id", "--run-id", dest="gen_run_id", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    result = run_generate(
        project=args.project,
        num_cases=args.num_cases,
        random_seed=args.seed,
        seed_name=args.seed_name,
        run_id=args.gen_run_id,
        resume=args.resume,
    )
    print({"project": result.project, "generation_run_id": result.run_id, "generated": len(result.cases)})


if __name__ == "__main__":
    main()
