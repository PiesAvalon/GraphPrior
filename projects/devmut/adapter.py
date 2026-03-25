from __future__ import annotations

import copy
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from GraphPrior.seed import get_seed_for_project
from GraphPrior.types import BugReport, GenerationRequest, GraphModel, GraphNode

from .._shared import prepend_path
from .._replay import stable_seed
from ..base import ProjectAdapter


class DevMutAdapter(ProjectAdapter):
    name = "devmut"
    _max_generation_attempts_per_case = 50

    def _project_dir(self) -> Path:
        return Path(__file__).resolve().parent

    def _imports(self):
        prepend_path(self._project_dir())
        import torch
        from common.mutation_ms.mutator_selection_logic import MCMC
        from common.mutation_torch.model_mutation_operators import GF_mut, LS_mut, NAI_mut, NEB_mut, NS_mut, WS_mut

        return torch, MCMC, {"WS": WS_mut, "NS": NS_mut, "GF": GF_mut, "NAI": NAI_mut, "NEB": NEB_mut, "LS": LS_mut}

    def _input_size(self) -> tuple[int, int, int, int]:
        return (1, 3, 224, 224)

    def _train_config(self) -> dict[str, int]:
        return {
            "test_size": 8,
            "input_size": ["(1, 3, 224, 224)"],
            "model_name": "resnet50",
            "dtypes": ["float"],
        }

    def _case_index(self, case_id: str) -> int:
        return int(case_id.rsplit("_", 1)[-1])

    def _ordered_models(self, models: list[GraphModel] | None) -> list[GraphModel]:
        return sorted(models or [], key=lambda item: self._case_index(item.case_id))

    def _seed_everything(self, torch, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)

    def _initial_seed(self, seed_name: str, random_seed: int) -> int:
        return stable_seed(self.name, seed_name, random_seed, "init")

    def _replay_seed(self, model: GraphModel) -> int:
        replay_seed = model.metadata.get("replay_seed")
        if replay_seed is not None:
            return int(replay_seed)
        generation_seed = int(model.metadata.get("_generation_random_seed", 2026))
        return stable_seed(self.name, model.case_id, generation_seed, "legacy-replay")

    def _build_seed_model(self, torch, seed_name: str, init_seed: int):
        self._seed_everything(torch, init_seed)
        return copy.deepcopy(get_seed_for_project(self.name, seed_name)).eval()

    def _weight_layer_names(self, model) -> list[str]:
        result: list[str] = []
        for name, layer in model.layer_names.items():
            cls_name = layer.__class__.__name__.lower()
            if "conv" in cls_name or cls_name == "linear":
                result.append(name)
        return result

    def _pair_for_ls(self, model) -> tuple[str, str] | None:
        layers = self._weight_layer_names(model)
        by_type: dict[str, list[str]] = {}
        for name in layers:
            cls_name = model.get_layers(name).__class__.__name__
            by_type.setdefault(cls_name, []).append(name)
        for names in by_type.values():
            if len(names) >= 2:
                return names[0], names[1]
        return None

    def _run_mutation(self, model, op_name: str, mutation_func, rng: np.random.Generator):
        train_cfg = self._train_config()
        input_size = self._input_size()
        if op_name == "LS":
            pair = self._pair_for_ls(model)
            if pair is None:
                raise RuntimeError("No compatible layer pair for LS")
            return mutation_func(model, input_size, pair[0], pair[1], train_configs=train_cfg)

        layers = self._weight_layer_names(model)
        if not layers:
            raise RuntimeError("No mutable layers available")
        layer_name = layers[int(rng.integers(0, len(layers)))]
        return mutation_func(model, input_size, layer_name, 0.4, train_configs=train_cfg)

    def _replay_case(self, current_model, model: GraphModel, torch, mutation_ops):
        op_name = str(model.metadata.get("operator", ""))
        if op_name not in mutation_ops:
            raise RuntimeError(f"Missing replay operator for {model.case_id}: {op_name!r}")
        replay_seed = self._replay_seed(model)
        self._seed_everything(torch, replay_seed)
        replay_rng = np.random.default_rng(replay_seed)
        mutant = copy.deepcopy(current_model).eval()
        self._run_mutation(mutant, op_name, mutation_ops[op_name], replay_rng)
        return mutant.eval()

    def _prepare_replay_cache(self, models: list[GraphModel] | None):
        ordered = self._ordered_models(models)
        signature = tuple(
            (
                item.case_id,
                item.seed_name,
                item.metadata.get("operator"),
                item.metadata.get("replay_seed"),
                item.metadata.get("init_seed"),
                item.metadata.get("_generation_random_seed"),
            )
            for item in ordered
        )
        if getattr(self, "_replay_signature", None) == signature:
            return

        cwd = os.getcwd()
        os.chdir(self._project_dir())
        try:
            torch, _, mutation_ops = self._imports()
        finally:
            os.chdir(cwd)

        replay_cache: dict[str, Any] = {}
        init_seed = None
        if ordered:
            seed_name = ordered[0].seed_name or "resnet50"
            init_seed = int(
                ordered[0].metadata.get(
                    "init_seed",
                    self._initial_seed(seed_name, int(ordered[0].metadata.get("_generation_random_seed", 2026))),
                )
            )
            current_model = self._build_seed_model(torch, seed_name, init_seed)
            for case in ordered:
                current_model = self._replay_case(current_model, case, torch, mutation_ops)
                replay_cache[case.case_id] = copy.deepcopy(current_model).eval()

        self._replay_signature = signature
        self._replay_cache = replay_cache
        self._replay_init_seed = init_seed

    def _canonical_op(self, module) -> str:
        cls_name = module.__class__.__name__.lower()
        mapping = {
            "conv2d": "conv2d",
            "linear": "linear",
            "batchnorm2d": "batchnorm",
            "maxpool2d": "maxpool2d",
            "adaptiveavgpool2d": "global_avg_pool",
            "flatten": "flatten",
            "relu": "relu",
        }
        return mapping.get(cls_name, cls_name)

    def _flatten_names(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            items: list[str] = []
            for item in value:
                items.extend(self._flatten_names(item))
            return items
        return [str(value)]

    def _normalize_shape(self, value: Any) -> Any:
        if isinstance(value, (list, tuple)):
            return tuple(self._normalize_shape(item) for item in value)
        return value

    def _module_dtype(self, module) -> str | None:
        for tensor in list(module.parameters(recurse=False)) + list(module.buffers(recurse=False)):
            return str(tensor.dtype)
        return None

    def _toposort(self, deps: dict[str, list[str]]) -> list[str]:
        indegree = {name: len(parents) for name, parents in deps.items()}
        children: dict[str, list[str]] = {name: [] for name in deps}
        for name, parents in deps.items():
            for parent in parents:
                children.setdefault(parent, []).append(name)

        ready = sorted([name for name, degree in indegree.items() if degree == 0])
        order: list[str] = []
        while ready:
            name = ready.pop(0)
            order.append(name)
            for child in sorted(children.get(name, [])):
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
                    ready.sort()
        return order

    def _module_attrs(self, model, name: str, module, op: str) -> dict[str, Any]:
        attrs: dict[str, Any] = {
            "name": name,
            "input_shape": self._normalize_shape(model.get_inshape(name)) if hasattr(model, "get_inshape") else None,
            "output_shape": self._normalize_shape(model.get_outshape(name)) if hasattr(model, "get_outshape") else None,
            "dtype": self._module_dtype(module) or "torch.float32",
        }

        if hasattr(module, "out_channels"):
            attrs["out_channels"] = int(module.out_channels)
        if hasattr(module, "out_features"):
            attrs["out_dim"] = int(module.out_features)
        if hasattr(module, "kernel_size"):
            kernel_size = module.kernel_size
            if not isinstance(kernel_size, tuple):
                kernel_size = (kernel_size, kernel_size)
            attrs["kernel_size"] = tuple(int(v) for v in kernel_size)
        if hasattr(module, "stride"):
            stride = module.stride
            if not isinstance(stride, tuple):
                stride = (stride, stride)
            attrs["stride"] = tuple(int(v) for v in stride)
        if hasattr(module, "padding"):
            padding = module.padding
            if isinstance(padding, str):
                attrs["padding"] = padding
            else:
                if not isinstance(padding, tuple):
                    padding = (padding, padding)
                attrs["padding"] = tuple(int(v) for v in padding)
        if hasattr(module, "groups"):
            attrs["groups"] = int(module.groups)
        if hasattr(module, "num_features"):
            attrs["num_features"] = int(module.num_features)
        if hasattr(module, "output_size"):
            output_size = module.output_size
            if isinstance(output_size, tuple):
                attrs["output_size"] = tuple(int(v) for v in output_size)
            elif output_size is not None:
                attrs["output_size"] = (int(output_size), int(output_size))

        if op == "global_avg_pool":
            attrs["output_size"] = attrs.get("output_size", (1, 1))
        return attrs

    def _model_to_graph(self, case_id: str, model, seed_name: str | None, parent: str | None, metadata: dict) -> GraphModel:
        node_names = [name for name in model.orders.keys() if name not in {"INPUT", "OUTPUT"} and name in model.layer_names]
        node_set = set(node_names)
        deps: dict[str, list[str]] = {}
        for name in node_names:
            pred_spec = model.orders[name][0] if model.orders.get(name) else []
            parents = [
                parent
                for parent in self._flatten_names(pred_spec)
                if parent not in {"INPUT", "OUTPUT"} and parent in node_set
            ]
            deps[name] = list(dict.fromkeys(parents))

        topo_names = self._toposort(deps)
        id_map = {name: idx for idx, name in enumerate(topo_names)}
        nodes: list[GraphNode] = []
        for name in topo_names:
            module = model.get_layers(name)
            op = self._canonical_op(module)
            nodes.append(
                GraphNode(
                    node_id=id_map[name],
                    op_type=f"torch:{op}",
                    inputs=tuple(id_map[parent] for parent in deps.get(name, [])),
                    attrs=self._module_attrs(model, name, module, op),
                )
            )
        return GraphModel(
            case_id=case_id,
            project=self.name,
            nodes=nodes,
            seed_name=seed_name,
            parent_case_id=parent,
            metadata=metadata,
        )

    def generate_models(
        self,
        request: GenerationRequest,
        output_root: Path,
        existing_models: list[GraphModel] | None = None,
    ):
        del output_root
        cwd = os.getcwd()
        os.chdir(self._project_dir())
        try:
            torch, MCMC, mutation_ops = self._imports()
        finally:
            os.chdir(cwd)

        seed_name = request.seed_name or "resnet50"
        init_seed = self._initial_seed(seed_name, request.random_seed)
        current_model = self._build_seed_model(torch, seed_name, init_seed)
        selector = MCMC(list(mutation_ops.keys()))
        last_op: str | None = None
        existing_models = self._ordered_models(existing_models)

        if existing_models:
            init_seed = int(existing_models[0].metadata.get("init_seed", init_seed))
            current_model = self._build_seed_model(torch, seed_name, init_seed)
            for existing in existing_models:
                op_name = str(existing.metadata.get("operator", ""))
                if op_name in selector.mutators:
                    reward = float(existing.metadata.get("reward", 0.0) or 0.0)
                    if reward > 0:
                        selector.mutators[op_name].delta_bigger_than_zero += 1
                    selector.mutators[op_name].total += 1
                    last_op = op_name
                current_model = self._replay_case(current_model, existing, torch, mutation_ops)

        max_attempts = int(request.hyper_params.get("max_generation_attempts_per_case", self._max_generation_attempts_per_case))
        for idx in range(len(existing_models), request.num_cases):
            case_id = f"devmut_{idx:03d}"
            parent_case_id = f"devmut_{idx - 1:03d}" if idx > 0 else None
            last_error: str | None = None
            for attempt in range(1, max_attempts + 1):
                replay_seed = stable_seed(self.name, seed_name, request.random_seed, case_id, attempt, "replay")
                self._seed_everything(torch, replay_seed)
                op_name = selector.choose_mutator(last_op)
                mutant = copy.deepcopy(current_model).eval()
                mutation_result = None
                try:
                    mutation_result = self._run_mutation(mutant, op_name, mutation_ops[op_name], np.random.default_rng(replay_seed))
                    sample_seed = stable_seed(self.name, case_id, replay_seed, "sample")
                    self._seed_everything(torch, sample_seed)
                    sample = torch.randn(*self._input_size(), dtype=torch.float32)
                    with torch.no_grad():
                        origin_output = current_model(sample)
                        mutant_output = mutant(sample)
                    mutant_np = mutant_output.detach().cpu().numpy()
                    if np.isnan(mutant_np).any() or np.isinf(mutant_np).any():
                        raise RuntimeError("mutant_forward_nan_or_inf")
                    reward = float(torch.linalg.norm((origin_output - mutant_output).reshape(-1)).item())
                except Exception as exc:
                    last_error = repr(exc)
                    continue

                if reward > 0:
                    selector.mutators[op_name].delta_bigger_than_zero += 1
                selector.mutators[op_name].total += 1
                current_model = mutant
                last_op = op_name

                metadata = {
                    "operator": op_name,
                    "reward": reward,
                    "generation_attempts": attempt,
                    "mutation_result": str(mutation_result),
                    "seed_name": seed_name,
                    "input_shape": self._input_size()[1:],
                    "batch_size": self._input_size()[0],
                    "mutation_depth": idx + 1,
                    "init_seed": init_seed,
                    "replay_seed": replay_seed,
                    "generation_random_seed": request.random_seed,
                }
                yield self._model_to_graph(case_id, mutant, seed_name, parent_case_id, metadata)
                break
            else:
                raise RuntimeError(f"DevMuT failed to generate {case_id} after {max_attempts} attempts: {last_error}")

    def detect_bug(
        self,
        model: GraphModel,
        models: list[GraphModel] | None = None,
        diff_threshold: float = 1e-5,
    ) -> BugReport:
        started = time.perf_counter()
        if model.metadata.get("generation_error"):
            runtime = time.perf_counter() - started
            return BugReport(model.case_id, self.name, True, "generation_runtime_error", runtime, {"error": model.metadata["generation_error"]})

        cwd = os.getcwd()
        os.chdir(self._project_dir())
        try:
            torch, _, _ = self._imports()
        finally:
            os.chdir(cwd)

        seed_name = model.seed_name or "resnet50"
        self._prepare_replay_cache(models)
        init_seed = int(
            model.metadata.get(
                "init_seed",
                self._replay_init_seed or self._initial_seed(seed_name, int(model.metadata.get("_generation_random_seed", 2026))),
            )
        )
        seed = self._build_seed_model(torch, seed_name, init_seed)
        working = copy.deepcopy(self._replay_cache.get(model.case_id, seed)).eval()

        try:
            sample_seed = stable_seed(self.name, model.case_id, init_seed, "sample")
            self._seed_everything(torch, sample_seed)
            sample = torch.randn(*self._input_size(), dtype=torch.float32)
            base = copy.deepcopy(seed).eval()
            with torch.no_grad():
                base_out = base(sample)
                mutant_out = working(sample)
            mutant_np = mutant_out.detach().cpu().numpy()
            if np.isnan(mutant_np).any() or np.isinf(mutant_np).any():
                runtime = time.perf_counter() - started
                return BugReport(model.case_id, self.name, True, "nan_or_inf", runtime, {"score": None})
            score = float(torch.linalg.norm((base_out - mutant_out).reshape(-1)).item())
            runtime = time.perf_counter() - started
            return BugReport(
                case_id=model.case_id,
                project=self.name,
                has_bug=score > diff_threshold,
                bug_type="mutation_score_delta" if score > diff_threshold else None,
                runtime_s=runtime,
                raw_result={"score": score},
            )
        except Exception as exc:
            runtime = time.perf_counter() - started
            return BugReport(model.case_id, self.name, True, "runtime_error", runtime, {"error": repr(exc)})
