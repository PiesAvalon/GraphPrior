from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path

from GraphPrior.types import BugReport, GenerationRequest, GraphModel


class ProjectAdapter(ABC):
    name: str
    uses_seed_model: bool = True

    @abstractmethod
    def generate_models(
        self,
        request: GenerationRequest,
        output_root: Path,
        existing_models: list[GraphModel] | None = None,
    ) -> Iterable[GraphModel]:
        raise NotImplementedError

    @abstractmethod
    def detect_bug(
        self,
        model: GraphModel,
        models: list[GraphModel] | None = None,
        diff_threshold: float = 1e-5,
    ) -> BugReport:
        raise NotImplementedError
