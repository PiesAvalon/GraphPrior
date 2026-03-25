from __future__ import annotations

import copy
import random
import sys
import time
import types
from pathlib import Path

import numpy as np

from GraphPrior.seed import get_seed_for_project
from GraphPrior.types import BugReport, GenerationRequest, GraphModel, GraphNode

from .._shared import prepend_path
from .._replay import stable_seed
from ..base import ProjectAdapter


class _MCMC:
    class Mutator:
        def __init__(self, name: str) -> None:
            self.name = name
            self.total = 0
            self.delta_bigger_than_zero = 0

        @property
        def score(self) -> float:
            return self.delta_bigger_than_zero / (self.total + 1e-7)

    def __init__(self, mutate_ops: list[str]) -> None:
        self.p = 1 / len(mutate_ops)
        self._mutators = [self.Mutator(name=op) for op in mutate_ops]

    @property
    def mutators(self) -> dict[str, "_MCMC.Mutator"]:
        return {item.name: item for item in self._mutators}

    def choose_mutator(self, previous: str | None) -> str:
        if previous is None:
            return self._mutators[np.random.randint(0, len(self._mutators))].name
        random.shuffle(self._mutators)
        self._mutators.sort(key=lambda item: item.score, reverse=True)
        k1 = next(i for i, item in enumerate(self._mutators) if item.name == previous)
        k2 = -1
        prob = 0.0
        while np.random.rand() >= prob:
            k2 = np.random.randint(0, len(self._mutators))
            prob = (1 - self.p) ** (k2 - k1)
        return self._mutators[k2].name


def _select_places(sequence, k: int):
    for _ in range(5):
        chosen = random.choices(list(sequence), k=k)
        subs_place = max(chosen)
        chosen.remove(subs_place)
        if max(chosen) != subs_place:
            return subs_place, chosen
    return None, None


class ModelMetaAdapter(ProjectAdapter):
    name = "modelmeta"
    _max_generation_attempts_per_case = 50

    def _project_dir(self) -> Path:
        return Path(__file__).resolve().parent

    def _imports(self):
        prepend_path(self._project_dir())
        prepend_path(self._project_dir() / "torch_mutation")
        import torch
        import torch.fx as fx
        from torch_mutation.handel_shape import handle_format
        from torch_mutation.metrics import ChebyshevDistance
        from torch_mutation.rules_torch import (
            rule1,
            rule2,
            rule3,
            rule4,
            rule5,
            rule6,
            rule7,
            rule8,
            rule9,
            rule10,
            rule11,
            rule12,
            rule13,
            rule14,
            rule15,
            rule16,
            rule17,
            rule18,
        )

        for module in (rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12, rule13, rule14, rule15, rule16, rule17, rule18):
            setattr(module, "device", "cpu")

        rules_dict = {
            torch.nn.Conv2d: [rule1, rule3, rule5, rule6, rule7, rule8],
            torch.nn.AvgPool2d: [rule1, rule3, rule12, rule13, rule14],
            torch.nn.MaxPool2d: [rule1, rule3, rule12, rule13, rule14],
            torch.nn.ReLU: [rule1, rule15],
            torch.nn.ReLU6: [rule1],
            torch.nn.BatchNorm2d: [rule1, rule4, rule9, rule10, rule11],
            torch.nn.Linear: [rule1],
            torch.nn.Flatten: [rule1],
            torch.nn.Hardsigmoid: [rule1],
            torch.nn.Sigmoid: [rule16, rule1],
            torch.nn.Softmax: [rule17, rule1],
            torch.nn.Tanh: [rule18, rule1],
            torch.nn.ConvTranspose2d: [rule1],
            torch.nn.LeakyReLU: [rule1, rule15],
            torch.nn.AdaptiveAvgPool2d: [rule1, rule12, rule13, rule14],
            torch.nn.Dropout: [rule1],
            torch.nn.Embedding: [rule1],
            torch.nn.LSTM: [rule1],
        }

        def reflect_name(option_name, option_rule):
            rule_name = option_rule.__name__.split(".")[-1]
            return f"{option_name}_mutated_{rule_name}"

        def match_rule(option_rule_name):
            mapping = {
                "rule1": rule1,
                "rule2": rule2,
                "rule3": rule3,
                "rule4": rule4,
                "rule5": rule5,
                "rule6": rule6,
                "rule7": rule7,
                "rule8": rule8,
                "rule9": rule9,
                "rule10": rule10,
                "rule11": rule11,
                "rule12": rule12,
                "rule13": rule13,
                "rule14": rule14,
                "rule15": rule15,
                "rule16": rule16,
                "rule17": rule17,
                "rule18": rule18,
            }
            return mapping[option_rule_name]

        def rule_reflect_class(option_rule, option_instance):
            if option_rule is rule1:
                name = type(option_instance).__name__
                return getattr(rule1, f"TransLayer_rule1_{name}")
            if option_rule is rule2:
                return rule2.TransLayer_rule2
            if option_rule is rule3:
                name = type(option_instance).__name__
                return getattr(rule3, f"TransLayer_rule3_{name}")
            if option_rule is rule4:
                return rule4.TransLayer_rule4
            if option_rule is rule5:
                return rule5.TransLayer_rule5
            if option_rule is rule6:
                return rule6.TransLayer_rule6
            if option_rule is rule7:
                return rule7.TransLayer_rule7
            if option_rule is rule8:
                return rule8.TransLayer_rule8
            if option_rule is rule9:
                return rule9.TransLayer_rule9
            if option_rule is rule10:
                return rule10.TransLayer_rule10
            if option_rule is rule11:
                return rule11.TransLayer_rule11
            if option_rule is rule12:
                name = type(option_instance).__name__
                return getattr(rule12, f"TransLayer_rule12_{name}")
            if option_rule is rule13:
                name = type(option_instance).__name__
                return getattr(rule13, f"TransLayer_rule13_{name}")
            if option_rule is rule14:
                name = type(option_instance).__name__
                return getattr(rule14, f"TransLayer_rule14_{name}")
            if option_rule is rule15:
                name = type(option_instance).__name__
                return getattr(rule15, f"TransLayer_rule15_{name}")
            if option_rule is rule16:
                return rule16.TransLayer_rule16
            if option_rule is rule17:
                return rule17.TransLayer_rule17
            if option_rule is rule18:
                return rule18.TransLayer_rule18
            raise ValueError(option_rule)

        cargo_module = types.ModuleType("cargo")
        cargo_module.reflect_name = reflect_name
        cargo_module.match_rule = match_rule
        cargo_module.rule_reflect_class = rule_reflect_class
        cargo_module.MCMC = _MCMC
        cargo_module.random = random
        cargo_module.np = np
        cargo_module.torch = torch
        sys.modules["cargo"] = cargo_module
        sys.modules["torch_mutation.cargo"] = cargo_module

        config_module = types.ModuleType("config")
        config_module.device = "cpu"
        config_module.rules_dict = rules_dict
        sys.modules["config"] = config_module
        sys.modules["torch_mutation.config"] = config_module

        for module_name in ("torch_mutation.MR_structure", "torch_mutation.api_mutation"):
            sys.modules.pop(module_name, None)

        from torch_mutation.MR_structure import ABSOC_A, ABSOC_B, PIOC, UOC
        from torch_mutation.api_mutation import api_mutation

        return torch, fx, handle_format, ChebyshevDistance, api_mutation, {
            "UOC": UOC,
            "PIOC": PIOC,
            "ABSOC_A": ABSOC_A,
            "ABSOC_B": ABSOC_B,
        }

    def _input_shape(self) -> tuple[int, int, int, int]:
        return (1, 3, 224, 224)

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

    def _build_seed_trace(self, torch, fx, seed_name: str, init_seed: int):
        self._seed_everything(torch, init_seed)
        seed_model = get_seed_for_project(self.name, seed_name)
        return fx.symbolic_trace(copy.deepcopy(seed_model).eval())

    def _apply_case_mutation(self, current, model: GraphModel, torch, api_mutation, mr_structures):
        idx = self._case_index(model.case_id)
        replay_seed = self._replay_seed(model)
        self._seed_everything(torch, replay_seed)
        log_dict: dict[int, dict[str, object]] = {idx: {"d_name": model.case_id}}
        selected_deadcode = str(model.metadata.get("selected_deadcode", "Dense"))
        selected_mr = str(model.metadata.get("selected_mr_structure", ""))
        api_mutation_type = str(model.metadata.get("api_mutation_type", "deadcode"))
        if selected_mr not in mr_structures:
            raise RuntimeError(f"Missing replay mutator for {model.case_id}: {selected_mr!r}")

        d = copy.deepcopy(current)
        with torch.no_grad():
            graph = d.graph
            nodelist = []
            for node in graph.nodes:
                if node.op in ["call_module", "root"] or (
                    node.op == "call_function"
                    and any(token in node.name for token in ["uoc", "pioc", "absoc_a", "absoc_b"])
                ):
                    nodelist.append(node)
            subs_place, dep_places = _select_places(range(0, len(nodelist)), 5)
            if subs_place is None or dep_places is None:
                raise RuntimeError("Cannot find suitable insertion places")

            add_module = mr_structures[selected_mr](selected_deadcode, api_mutation_type, log_dict, idx, LOG_FLAG=False)
            inserted_name = f"{selected_mr.lower()}_{idx}_{int(model.metadata.get('generation_attempts', 1))}"
            d.add_module(inserted_name, add_module)
            dep_places.sort(reverse=True)
            aa = nodelist[dep_places[-1]]
            bb = nodelist[dep_places[-2]]
            cc = nodelist[dep_places[-3]]
            dd = nodelist[dep_places[-4]]
            if selected_mr == "PIOC":
                with cc.graph.inserting_after(cc):
                    new_node = cc.graph.call_module(inserted_name, args=(cc, cc, cc))
                    cc.replace_all_uses_with(new_node)
                    new_node.update_arg(0, aa)
                    new_node.update_arg(1, bb)
                    new_node.update_arg(2, cc)
            else:
                with dd.graph.inserting_after(dd):
                    new_node = dd.graph.call_module(inserted_name, args=(dd, dd, dd, dd))
                    dd.replace_all_uses_with(new_node)
                    new_node.update_arg(0, aa)
                    new_node.update_arg(1, bb)
                    new_node.update_arg(2, cc)
                    new_node.update_arg(3, dd)
            graph.lint()
            d.recompile()
            if api_mutation_type == "seed_model":
                api_mutation(d, log_dict, idx, LOG_FLAG=False)
        return d.eval()

    def _prepare_replay_cache(self, models: list[GraphModel] | None):
        ordered = self._ordered_models(models)
        signature = tuple(
            (
                item.case_id,
                item.seed_name,
                item.metadata.get("selected_mr_structure"),
                item.metadata.get("replay_seed"),
                item.metadata.get("init_seed"),
                item.metadata.get("_generation_random_seed"),
            )
            for item in ordered
        )
        if getattr(self, "_replay_signature", None) == signature:
            return

        torch, fx, _, _, api_mutation, mr_structures = self._imports()
        replay_cache: dict[str, object] = {}
        init_seed = None
        if ordered:
            seed_name = ordered[0].seed_name or "resnet50"
            init_seed = int(
                ordered[0].metadata.get(
                    "init_seed",
                    self._initial_seed(seed_name, int(ordered[0].metadata.get("_generation_random_seed", 2026))),
                )
            )
            current = self._build_seed_trace(torch, fx, seed_name, init_seed)
            for case in ordered:
                current = self._apply_case_mutation(current, case, torch, api_mutation, mr_structures)
                replay_cache[case.case_id] = current

        self._replay_signature = signature
        self._replay_cache = replay_cache
        self._replay_init_seed = init_seed

    def _graph_from_fx(self, case_id: str, gm, seed_name: str | None, parent: str | None, metadata: dict) -> GraphModel:
        nodes: list[GraphNode] = []
        idx = 0
        id_map: dict[str, int] = {}
        for n in gm.graph.nodes:
            if n.op in {"placeholder", "output"}:
                continue
            inputs: list[int] = []
            for arg in n.all_input_nodes:
                if arg.name in id_map:
                    inputs.append(id_map[arg.name])
            nodes.append(
                GraphNode(
                    node_id=idx,
                    op_type=f"fx:{n.op}:{n.target}",
                    inputs=tuple(inputs),
                    attrs={"name": n.name},
                )
            )
            id_map[n.name] = idx
            idx += 1
        return GraphModel(case_id=case_id, project=self.name, nodes=nodes, seed_name=seed_name, parent_case_id=parent, metadata=metadata)

    def generate_models(
        self,
        request: GenerationRequest,
        output_root: Path,
        existing_models: list[GraphModel] | None = None,
    ):
        del output_root
        torch, fx, handle_format, ChebyshevDistance, api_mutation, mr_structures = self._imports()
        seed_name = request.seed_name or "resnet50"
        deadcode_names = ["Dense", "SELayer", "DenseLayer", "Inception_A", "PWDWPW_ResidualBlock", "ResidualBlock", "DropPath"]
        selector = _MCMC(list(mr_structures.keys()))
        last_used: str | None = None
        api_count = 0
        init_seed = self._initial_seed(seed_name, request.random_seed)
        current = self._build_seed_trace(torch, fx, seed_name, init_seed)
        sample_seed = stable_seed(self.name, seed_name, request.random_seed, "analysis-sample")
        self._seed_everything(torch, sample_seed)
        sample = torch.randn(*self._input_shape(), dtype=torch.float32)
        original_outputs = handle_format(current(sample))[0]
        existing_models = self._ordered_models(existing_models)

        for existing in existing_models:
            selected_mr = str(existing.metadata.get("selected_mr_structure", ""))
            if selected_mr in selector.mutators and not existing.metadata.get("generation_error"):
                reward = float(existing.metadata.get("reward", 0.0) or 0.0)
                selector.mutators[selected_mr].total += 1
                if reward > 0:
                    selector.mutators[selected_mr].delta_bigger_than_zero += 1
                last_used = selected_mr
            if existing.metadata.get("api_mutation_type") == "seed_model":
                api_count += 1
            current = self._apply_case_mutation(current, existing, torch, api_mutation, mr_structures)

        max_attempts = int(request.hyper_params.get("max_generation_attempts_per_case", self._max_generation_attempts_per_case))
        for idx in range(len(existing_models), request.num_cases):
            case_id = f"modelmeta_{idx:03d}"
            parent_case_id = f"modelmeta_{idx - 1:03d}" if idx > 0 else None
            last_error: str | None = None

            for attempt in range(1, max_attempts + 1):
                replay_seed = stable_seed(self.name, seed_name, request.random_seed, case_id, attempt, "replay")
                self._seed_everything(torch, replay_seed)
                attempt_rng = np.random.default_rng(replay_seed)
                log_dict: dict[int, dict[str, object]] = {idx: {"d_name": case_id}}
                selected_deadcode = str(deadcode_names[int(attempt_rng.integers(0, len(deadcode_names)))])
                selected_mr = selector.choose_mutator(last_used)
                api_mutation_type = "seed_model" if api_count < 8 else "deadcode"
                d = copy.deepcopy(current)

                try:
                    with torch.no_grad():
                        graph = d.graph
                        nodelist = []
                        for node in graph.nodes:
                            if node.op in ["call_module", "root"] or (
                                node.op == "call_function"
                                and any(token in node.name for token in ["uoc", "pioc", "absoc_a", "absoc_b"])
                            ):
                                nodelist.append(node)
                        subs_place, dep_places = _select_places(range(0, len(nodelist)), 5)
                        if subs_place is None or dep_places is None:
                            raise RuntimeError("Cannot find suitable insertion places")

                        add_module = mr_structures[selected_mr](selected_deadcode, api_mutation_type, log_dict, idx, LOG_FLAG=False)
                        inserted_name = f"{selected_mr.lower()}_{idx}_{attempt}"
                        d.add_module(inserted_name, add_module)
                        dep_places.sort(reverse=True)
                        aa = nodelist[dep_places[-1]]
                        bb = nodelist[dep_places[-2]]
                        cc = nodelist[dep_places[-3]]
                        dd = nodelist[dep_places[-4]]
                        if selected_mr == "PIOC":
                            with cc.graph.inserting_after(cc):
                                new_node = cc.graph.call_module(inserted_name, args=(cc, cc, cc))
                                cc.replace_all_uses_with(new_node)
                                new_node.update_arg(0, aa)
                                new_node.update_arg(1, bb)
                                new_node.update_arg(2, cc)
                        else:
                            with dd.graph.inserting_after(dd):
                                new_node = dd.graph.call_module(inserted_name, args=(dd, dd, dd, dd))
                                dd.replace_all_uses_with(new_node)
                                new_node.update_arg(0, aa)
                                new_node.update_arg(1, bb)
                                new_node.update_arg(2, cc)
                                new_node.update_arg(3, dd)
                        graph.lint()
                        d.recompile()
                        if api_mutation_type == "seed_model":
                            api_mutation(d, log_dict, idx, LOG_FLAG=False)

                        new_outputs = handle_format(d(sample))[0]
                        reward = float(ChebyshevDistance(original_outputs, new_outputs))
                except Exception as exc:
                    last_error = repr(exc)
                    continue

                selector.mutators[selected_mr].total += 1
                if reward > 0:
                    selector.mutators[selected_mr].delta_bigger_than_zero += 1
                current = d
                last_used = selected_mr
                if api_mutation_type == "seed_model":
                    api_count += 1

                metadata = {
                    "seed_name": seed_name,
                    "selected_deadcode": selected_deadcode,
                    "selected_mr_structure": selected_mr,
                    "api_mutation_type": api_mutation_type,
                    "reward": reward,
                    "generation_attempts": attempt,
                    "mutation_depth": idx + 1,
                    "init_seed": init_seed,
                    "replay_seed": replay_seed,
                    "generation_random_seed": request.random_seed,
                }
                yield self._graph_from_fx(case_id, d, seed_name, parent_case_id, metadata)
                break
            else:
                raise RuntimeError(
                    f"ModelMeta failed to generate {case_id} after {max_attempts} attempts: {last_error}"
                )

    def detect_bug(
        self,
        model: GraphModel,
        models: list[GraphModel] | None = None,
        diff_threshold: float = 1e-5,
    ) -> BugReport:
        started = time.perf_counter()
        torch, fx, handle_format, ChebyshevDistance, _, _ = self._imports()
        seed_name = model.seed_name or "resnet50"
        self._prepare_replay_cache(models)
        init_seed = int(
            model.metadata.get(
                "init_seed",
                self._replay_init_seed or self._initial_seed(seed_name, int(model.metadata.get("_generation_random_seed", 2026))),
            )
        )
        base = self._build_seed_trace(torch, fx, seed_name, init_seed)
        sample_seed = stable_seed(self.name, model.case_id, init_seed, "sample")
        self._seed_everything(torch, sample_seed)
        sample = torch.randn(*self._input_shape(), dtype=torch.float32)

        try:
            if model.metadata.get("generation_error"):
                runtime = time.perf_counter() - started
                return BugReport(model.case_id, self.name, True, "generation_runtime_error", runtime, {"error": model.metadata["generation_error"]})
            mutant = self._replay_cache.get(model.case_id, base)
            with torch.no_grad():
                base_out = handle_format(base(sample))[0]
                mutant_out = handle_format(mutant(sample))[0]
            mutant_np = np.asarray(mutant_out.detach().cpu().numpy())
            if np.isnan(mutant_np).any() or np.isinf(mutant_np).any():
                runtime = time.perf_counter() - started
                return BugReport(model.case_id, self.name, True, "nan_or_inf", runtime, {"distance": None})
            distance = float(ChebyshevDistance(base_out, mutant_out))
            runtime = time.perf_counter() - started
            return BugReport(
                case_id=model.case_id,
                project=self.name,
                has_bug=distance > diff_threshold,
                bug_type="chebyshev_distance" if distance > diff_threshold else None,
                runtime_s=runtime,
                raw_result={"distance": distance},
            )
        except Exception as exc:
            runtime = time.perf_counter() - started
            return BugReport(model.case_id, self.name, True, "runtime_error", runtime, {"error": repr(exc)})
