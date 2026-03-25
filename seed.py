from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent


def _prepend(path: Path) -> None:
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)


def get_seed_model(seed_name: str = "resnet50") -> Any:
    if seed_name != "resnet50":
        raise ValueError(f"Unsupported seed model: {seed_name}")
    import torchvision.models as tv_models

    return tv_models.resnet50(weights=None)


def get_seed_for_project(project: str, seed_name: str = "resnet50") -> Any:
    project = project.lower()
    if seed_name != "resnet50":
        raise ValueError(f"Unsupported seed model: {seed_name}")

    if project == "muffin":
        return None

    if project == "comet":
        import tensorflow as tf

        return tf.keras.applications.ResNet50(
            include_top=True,
            weights=None,
            input_shape=(64, 64, 3),
            classes=1000,
        )

    if project == "devmut":
        project_root = ROOT / "projects" / "devmut"
        _prepend(project_root)
        cwd = os.getcwd()
        try:
            os.chdir(project_root)
            from network.cv.resnet.resnet50_torch import resnet50
        finally:
            os.chdir(cwd)
        return resnet50()

    if project == "modelmeta":
        return get_seed_model(seed_name)

    raise ValueError(f"Unknown project: {project}")
