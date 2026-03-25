from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class NodeSpec:
    node_id: int
    op: str
    inputs: tuple[int, ...]
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class TestCase:
    case_id: str
    project: str
    mutation_depth: int
    input_shape: tuple[int, ...]
    batch_size: int
    nodes: list[NodeSpec]
    parent_case_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
