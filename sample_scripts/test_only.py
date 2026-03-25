from __future__ import annotations

import argparse
from pathlib import Path

from GraphPrior.components.test import run_test
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
    parser = argparse.ArgumentParser(description="Run testing only")
    parser.add_argument("--project", required=True, choices=["comet", "devmut", "muffin", "modelmeta"])
    parser.add_argument("--gen-run-id", required=True)
    parser.add_argument("--test-run-id", default=None)
    parser.add_argument("--diff-threshold", type=float, default=None)
    parser.add_argument("--global-config", type=str, default=None)
    args = parser.parse_args()

    diff_threshold = _resolve_diff_threshold(args.project, args.diff_threshold, args.global_config)
    test_run_id = args.test_run_id or default_stage_run_id("test", args.gen_run_id)
    reports = run_test(
        project=args.project,
        generation_run_id=args.gen_run_id,
        test_run_id=test_run_id,
        diff_threshold=diff_threshold,
    )
    print(
        {
            "project": args.project,
            "generation_run_id": args.gen_run_id,
            "test_run_id": test_run_id,
            "diff_threshold": diff_threshold,
            "tested": len(reports),
        }
    )


if __name__ == "__main__":
    main()
