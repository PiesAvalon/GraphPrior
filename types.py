from __future__ import annotations

# If this file is imported as top-level "types" (for example when cwd=GraphPrior),
# delegate to stdlib types to avoid shadowing failures during environment bootstrapping.
if __name__ == "types":
    import os

    stdlib_types = os.path.join(os.path.dirname(os.__file__), "types.py")
    with open(stdlib_types, "r", encoding="utf-8") as fh:
        exec(compile(fh.read(), stdlib_types, "exec"), globals(), globals())
else:
    from dataclasses import dataclass, field
    from pathlib import Path
    from typing import Any

    @dataclass(frozen=True)
    class GraphNode:
        node_id: int
        op_type: str
        inputs: tuple[int, ...]
        attrs: dict[str, Any] = field(default_factory=dict)

    @dataclass
    class GraphModel:
        case_id: str
        project: str
        nodes: list[GraphNode]
        seed_name: str | None = None
        parent_case_id: str | None = None
        metadata: dict[str, Any] = field(default_factory=dict)
        native_artifact_path: str | None = None

    @dataclass
    class GenerationRequest:
        project: str
        run_id: str
        num_cases: int
        random_seed: int
        seed_name: str | None = "resnet50"
        hyper_params: dict[str, Any] = field(default_factory=dict)

    @dataclass
    class GenerationResult:
        project: str
        run_id: str
        cases: list[GraphModel]

    @dataclass
    class BugReport:
        case_id: str
        project: str
        has_bug: bool
        bug_type: str | None
        runtime_s: float
        raw_result: dict[str, Any] = field(default_factory=dict)

    @dataclass
    class AnalysisRecord:
        case_id: str
        project: str
        coverage_metrics: dict[str, float]
        structure_metrics: dict[str, float]

    @dataclass
    class PriorRecord:
        case_id: str
        project: str
        rank: int
        score: float

    @dataclass
    class ValidationRecord:
        project: str
        run_id: str
        apfd: float
        apfdc: float
        bugs_found_over_time: list[dict[str, float]]

    @dataclass
    class NativeCase:
        case_id: str
        project: str
        artifact_dir: Path
        artifacts: dict[str, str]
        metadata: dict[str, Any] = field(default_factory=dict)
