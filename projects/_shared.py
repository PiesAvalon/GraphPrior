from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def prepend_path(path: Path) -> None:
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def layer_nodes_from_names(names: list[str], prefix: str = "layer") -> list[dict]:
    nodes: list[dict] = []
    prev = -1
    for idx, name in enumerate(names):
        nodes.append({
            "node_id": idx,
            "op_type": f"{prefix}:{name}",
            "inputs": (prev,) if prev >= 0 else tuple(),
            "attrs": {},
        })
        prev = idx
    return nodes
