from __future__ import annotations

from .base import ProjectAdapter
from .comet import CometAdapter
from .devmut import DevMutAdapter
from .muffin import MuffinAdapter
from .modelmeta import ModelMetaAdapter

PROJECTS: dict[str, type[ProjectAdapter]] = {
    "comet": CometAdapter,
    "devmut": DevMutAdapter,
    "muffin": MuffinAdapter,
    "modelmeta": ModelMetaAdapter,
}


def get_project(name: str) -> ProjectAdapter:
    try:
        return PROJECTS[name]()
    except KeyError as exc:
        raise ValueError(f"Unknown project: {name}") from exc


def list_projects() -> list[str]:
    return list(PROJECTS.keys())
