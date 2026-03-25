from __future__ import annotations

import copy
import os
import random
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from GraphPrior.seed import get_seed_for_project
from GraphPrior.types import BugReport, GenerationRequest, GraphModel, GraphNode

from ..base import ProjectAdapter
from .._replay import stable_seed
from .._shared import prepend_path


@dataclass
class _MutatorScore:
    name: str
    score: float = 1.0


class _CometMCMC:
    def __init__(self, mutators: list[str], p: float = 0.4) -> None:
        self.p = p
        self._mutators = [_MutatorScore(name=m) for m in mutators]

    def _index(self, name: str) -> int:
        for idx, item in enumerate(self._mutators):
            if item.name == name:
                return idx
        raise ValueError(name)

    def choose(self, last_used: str | None, rng: np.random.Generator | None = None) -> str:
        rng = rng or np.random.default_rng()
        self._mutators.sort(key=lambda item: item.score, reverse=True)
        if last_used is None:
            return self._mutators[int(rng.integers(0, len(self._mutators)))].name
        k1 = self._index(last_used)
        k2 = -1
        prob = 0.0
        while rng.random() >= prob:
            k2 = int(rng.integers(0, len(self._mutators)))
            prob = (1 - self.p) ** (k2 - k1)
        return self._mutators[k2].name

    def reward(self, name: str, value: float) -> None:
        for item in self._mutators:
            if item.name == name:
                item.score += max(0.0, float(value))
                return


class CometAdapter(ProjectAdapter):
    name = "comet"
    _max_generation_attempts_per_case = 100
    _stable_mutators = {"ARem", "ARep", "GF", "NAI", "NEB", "NS", "WS"}

    def _project_dir(self) -> Path:
        return Path(__file__).resolve().parent

    def _install_keras_compat_shims(self) -> None:
        """
        Provide legacy Keras module paths expected by trimmed COMET sources.
        """
        try:
            import keras
            from keras import backend as K
        except Exception:
            return

        if "keras.layers.core" not in sys.modules:
            core_module = types.ModuleType("keras.layers.core")
            core_module.Lambda = keras.layers.Lambda
            sys.modules["keras.layers.core"] = core_module

        if "keras.engine.input_layer" not in sys.modules:
            input_layer_module = types.ModuleType("keras.engine.input_layer")
            input_layer_module.InputLayer = keras.layers.InputLayer
            sys.modules["keras.engine.input_layer"] = input_layer_module

        if not isinstance(getattr(keras.layers.Layer, "_name", None), property):
            def _get_name(layer):
                return layer.name

            def _set_name(layer, value):
                try:
                    layer.name = value
                except Exception:
                    setattr(layer, "_name_compat", value)

            keras.layers.Layer._name = property(_get_name, _set_name)

        if not hasattr(K, "relu"):
            K.relu = keras.ops.relu
        if not hasattr(K, "tanh"):
            K.tanh = keras.ops.tanh
        if not hasattr(K, "sigmoid"):
            K.sigmoid = keras.ops.sigmoid

    def _imports(self):
        prepend_path(self._project_dir())
        self._install_keras_compat_shims()
        import tensorflow as tf
        try:
            from scripts.mutation.structure_mutation_generators import baseline_mutate_ops, generate_model_by_model_mutation
        except FileNotFoundError:
            # Some trimmed COMET distributions miss optional config files used by
            # non-baseline operators. Fall back to baseline-only operators.
            from scripts.mutation.structure_mutation_operators import (
                ARem_mut,
                ARep_mut,
                GF_mut,
                LA_mut,
                LC_mut,
                LR_mut,
                LS_mut,
                MLAMut,
                NAI_mut,
                NEB_mut,
                NS_mut,
                WS_mut,
            )

            baseline_ops = ["WS", "GF", "NEB", "NAI", "NS", "ARem", "ARep", "LA", "LC", "LR", "LS", "MLA"]
            mla_mut = MLAMut()

            def baseline_mutate_ops():
                return list(baseline_ops)

            def generate_model_by_model_mutation(
                model,
                operator,
                mutation_operator_mode=None,
                mutate_ratio=1,
                architecture_measure=None,
            ):
                del mutation_operator_mode, architecture_measure
                if operator == "WS":
                    return WS_mut(model=model, mutation_ratio=mutate_ratio)
                if operator == "GF":
                    return GF_mut(model=model, mutation_ratio=mutate_ratio)
                if operator == "NEB":
                    return NEB_mut(model=model, mutation_ratio=mutate_ratio)
                if operator == "NAI":
                    return NAI_mut(model=model, mutation_ratio=mutate_ratio)
                if operator == "NS":
                    return NS_mut(model=model)
                if operator == "ARem":
                    return ARem_mut(model=model)
                if operator == "ARep":
                    return ARep_mut(model=model)
                if operator == "LA":
                    return LA_mut(model=model)
                if operator == "LC":
                    return LC_mut(model=model)
                if operator == "LR":
                    return LR_mut(model=model)
                if operator == "LS":
                    return LS_mut(model=model)
                if operator == "MLA":
                    return mla_mut.mutate(model=model, mutated_layer_indices=None)
                return None

        try:
            from scripts.tools.utils import MetricsUtils
        except Exception:
            MetricsUtils = None
        return tf, baseline_mutate_ops, generate_model_by_model_mutation, MetricsUtils

    def _case_index(self, case_id: str) -> int:
        return int(case_id.rsplit("_", 1)[-1])

    def _ordered_models(self, models: list[GraphModel] | None) -> list[GraphModel]:
        return sorted(models or [], key=lambda item: self._case_index(item.case_id))

    def _seed_everything(self, tf, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed % (2**32))
        tf.keras.utils.set_random_seed(seed)

    def _initial_seed(self, seed_name: str, random_seed: int) -> int:
        return stable_seed(self.name, seed_name, random_seed, "init")

    def _replay_seed(self, model: GraphModel) -> int:
        replay_seed = model.metadata.get("replay_seed")
        if replay_seed is not None:
            return int(replay_seed)
        generation_seed = int(model.metadata.get("_generation_random_seed", 2026))
        return stable_seed(self.name, model.case_id, generation_seed, "legacy-replay")

    def _build_seed_model(self, tf, seed_name: str, init_seed: int):
        self._seed_everything(tf, init_seed)
        return tf.keras.models.clone_model(get_seed_for_project(self.name, seed_name))

    def _clone_model(self, tf, model, custom_objects):
        with tf.keras.utils.custom_object_scope(custom_objects()):
            clone = tf.keras.models.clone_model(model)
        clone.set_weights(model.get_weights())
        return clone

    def _prepare_replay_cache(self, models: list[GraphModel] | None):
        ordered = self._ordered_models(models)
        signature = tuple(
            (
                item.case_id,
                item.seed_name,
                item.parent_case_id,
                item.metadata.get("mutator"),
                item.metadata.get("replay_seed"),
                item.metadata.get("init_seed"),
                item.metadata.get("_generation_random_seed"),
            )
            for item in ordered
        )
        if getattr(self, "_replay_signature", None) == signature:
            return

        project_dir = self._project_dir()
        cwd = os.getcwd()
        os.chdir(project_dir)
        try:
            tf, baseline_mutate_ops, generate_model_by_model_mutation, _ = self._imports()
            from scripts.prediction.custom_objects import custom_objects as comet_custom_objects
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
            seed_model = self._build_seed_model(tf, seed_name, init_seed)
            architecture_pool: dict[str, Any] = {"seed_000": seed_model}
            del baseline_mutate_ops
            for case in ordered:
                parent_name = str(case.parent_case_id or case.metadata.get("parent_case_id") or "seed_000")
                parent_model = architecture_pool.get(parent_name)
                if parent_model is None:
                    raise RuntimeError(f"Missing replay parent {parent_name!r} for {case.case_id}")
                replay_seed = self._replay_seed(case)
                self._seed_everything(tf, replay_seed)
                mutator = str(case.metadata.get("mutator", ""))
                if mutator == "WS_fallback":
                    mutant = self._fallback_weight_shift(tf, parent_model, np.random.default_rng(replay_seed))
                else:
                    parent_clone = self._clone_model(tf, parent_model, comet_custom_objects)
                    cwd = os.getcwd()
                    os.chdir(project_dir)
                    try:
                        mutant = generate_model_by_model_mutation(
                            model=parent_clone,
                            operator=mutator,
                            mutation_operator_mode="diverse",
                            architecture_measure=None,
                        )
                    finally:
                        os.chdir(cwd)
                if mutant is None:
                    raise RuntimeError(f"Replay produced no mutant for {case.case_id}")
                architecture_pool[case.case_id] = mutant
                replay_cache[case.case_id] = mutant

        self._replay_signature = signature
        self._replay_cache = replay_cache
        self._replay_init_seed = init_seed

    def _canonical_op(self, layer) -> str:
        cls_name = layer.__class__.__name__.lower()
        mapping = {
            "conv2d": "conv2d",
            "dense": "linear",
            "batchnormalization": "batchnorm",
            "maxpooling2d": "maxpool2d",
            "globalaveragepooling2d": "global_avg_pool",
            "flatten": "flatten",
            "add": "add",
            "concatenate": "concat",
            "averagepooling2d": "avgpool2d",
            "zeropadding2d": "pad2d",
            "activation": "activation",
            "relu": "relu",
        }
        return mapping.get(cls_name, cls_name)

    def _fallback_weight_shift(self, tf, parent_model, rng: np.random.Generator):
        clone = tf.keras.models.clone_model(parent_model)
        clone.set_weights(parent_model.get_weights())
        weights = clone.get_weights()
        for idx, weight in enumerate(weights):
            if getattr(weight, "size", 0):
                noise = rng.standard_normal(weight.shape).astype(weight.dtype) * 1e-3
                weights[idx] = weight + noise
                clone.set_weights(weights)
                return clone
        raise RuntimeError("no_mutable_weights_for_fallback")

    def _flatten_tensors(self, value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            items: list[Any] = []
            for item in value:
                items.extend(self._flatten_tensors(item))
            return items
        return [value]

    def _tensor_shape(self, tensor: Any) -> tuple[Any, ...] | tuple[tuple[Any, ...], ...] | None:
        shape = getattr(tensor, "shape", None)
        if shape is None:
            return None
        try:
            return tuple(None if dim is None else int(dim) for dim in shape)
        except TypeError:
            return None

    def _shape_summary(self, tensors: list[Any]) -> Any:
        shapes = [shape for tensor in tensors if (shape := self._tensor_shape(tensor)) is not None]
        if not shapes:
            return None
        if len(shapes) == 1:
            return shapes[0]
        return tuple(shapes)

    def _tensor_dtype(self, tensors: list[Any]) -> str | None:
        for tensor in tensors:
            dtype = getattr(tensor, "dtype", None)
            if dtype is not None:
                return str(dtype)
        return None

    def _keras_history(self, tensor: Any) -> tuple[Any, int, int] | None:
        history = getattr(tensor, "_keras_history", None)
        if history is None:
            return None

        operation = getattr(history, "operation", None) or getattr(history, "layer", None)
        node_index = getattr(history, "node_index", None)
        tensor_index = getattr(history, "tensor_index", None)
        if operation is not None and node_index is not None:
            return operation, int(node_index), int(tensor_index or 0)

        if isinstance(history, tuple) and len(history) >= 2:
            operation = history[0]
            node_index = int(history[1])
            tensor_index = int(history[2]) if len(history) > 2 else 0
            return operation, node_index, tensor_index
        return None

    def _toposort(self, deps: dict[tuple[str, int], list[tuple[str, int]]]) -> list[tuple[str, int]]:
        indegree = {key: len(value) for key, value in deps.items()}
        children: dict[tuple[str, int], list[tuple[str, int]]] = {key: [] for key in deps}
        for key, parents in deps.items():
            for parent in parents:
                children.setdefault(parent, []).append(key)

        ready = sorted([key for key, degree in indegree.items() if degree == 0])
        order: list[tuple[str, int]] = []
        while ready:
            key = ready.pop(0)
            order.append(key)
            for child in sorted(children.get(key, [])):
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
                    ready.sort()
        return order

    def _node_attrs(self, layer, node, op: str) -> dict[str, Any]:
        input_tensors = self._flatten_tensors(getattr(node, "input_tensors", []))
        output_tensors = self._flatten_tensors(getattr(node, "output_tensors", []))
        attrs: dict[str, Any] = {
            "name": layer.name,
            "input_shape": self._shape_summary(input_tensors),
            "output_shape": self._shape_summary(output_tensors),
            "dtype": self._tensor_dtype(output_tensors) or self._tensor_dtype(input_tensors),
        }

        if hasattr(layer, "filters"):
            attrs["out_channels"] = int(layer.filters)
        if hasattr(layer, "units"):
            attrs["out_dim"] = int(layer.units)
        if hasattr(layer, "kernel_size"):
            attrs["kernel_size"] = tuple(int(v) for v in layer.kernel_size)
        if hasattr(layer, "strides"):
            attrs["stride"] = tuple(int(v) for v in layer.strides)
        if hasattr(layer, "padding"):
            attrs["padding"] = str(layer.padding)
        if hasattr(layer, "pool_size"):
            attrs["kernel_size"] = tuple(int(v) for v in layer.pool_size)
        if hasattr(layer, "axis"):
            attrs["axis"] = int(layer.axis) if isinstance(layer.axis, int) else tuple(int(v) for v in layer.axis)
        if hasattr(layer, "epsilon"):
            attrs["epsilon"] = float(layer.epsilon)
        if hasattr(layer, "groups"):
            attrs["groups"] = int(layer.groups)
        if hasattr(layer, "activation") and getattr(layer.activation, "__name__", None):
            attrs["activation"] = str(layer.activation.__name__)

        output_shape = attrs.get("output_shape")
        if op == "batchnorm" and output_shape and isinstance(output_shape, tuple) and output_shape:
            axis = attrs.get("axis", -1)
            if isinstance(axis, int):
                resolved_axis = axis if axis >= 0 else len(output_shape) + axis
                if 0 <= resolved_axis < len(output_shape):
                    feature_dim = output_shape[resolved_axis]
                    if feature_dim is not None:
                        attrs["num_features"] = int(feature_dim)

        if op == "global_avg_pool":
            attrs["output_size"] = (1, 1)
        return attrs

    def _model_to_graph(self, case_id: str, model, seed_name: str | None, parent: str | None, metadata: dict) -> GraphModel:
        node_records: dict[tuple[str, int], tuple[Any, Any]] = {}
        deps: dict[tuple[str, int], list[tuple[str, int]]] = {}
        pending = list(self._flatten_tensors(getattr(model, "outputs", [])))

        while pending:
            tensor = pending.pop()
            history = self._keras_history(tensor)
            if history is None:
                continue
            layer, node_index, _ = history
            if layer.__class__.__name__ == "InputLayer":
                continue

            key = (layer.name, node_index)
            if key in node_records:
                continue

            node = layer._inbound_nodes[node_index]
            input_tensors = self._flatten_tensors(getattr(node, "input_tensors", []))
            parents: list[tuple[str, int]] = []
            for input_tensor in input_tensors:
                parent_history = self._keras_history(input_tensor)
                if parent_history is None:
                    continue
                parent_layer, parent_node_index, _ = parent_history
                if parent_layer.__class__.__name__ == "InputLayer":
                    continue
                parent_key = (parent_layer.name, parent_node_index)
                parents.append(parent_key)
                pending.append(input_tensor)

            node_records[key] = (layer, node)
            deps[key] = list(dict.fromkeys(parents))

        topo_keys = self._toposort(deps)
        id_map = {key: idx for idx, key in enumerate(topo_keys)}
        nodes: list[GraphNode] = []
        for key in topo_keys:
            layer, node = node_records[key]
            op = self._canonical_op(layer)
            nodes.append(
                GraphNode(
                    node_id=id_map[key],
                    op_type=f"keras:{op}",
                    inputs=tuple(id_map[parent] for parent in deps.get(key, [])),
                    attrs=self._node_attrs(layer, node, op),
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
        seed_name = request.seed_name or "resnet50"
        project_dir = self._project_dir()
        try:
            cwd = os.getcwd()
            os.chdir(project_dir)
            try:
                tf, baseline_mutate_ops, generate_model_by_model_mutation, MetricsUtils = self._imports()
                from scripts.prediction.custom_objects import custom_objects as comet_custom_objects
            finally:
                os.chdir(cwd)
        except Exception as exc:
            raise RuntimeError(f"COMET imports failed: {exc!r}") from exc

        init_seed = self._initial_seed(seed_name, request.random_seed)
        existing_models = self._ordered_models(existing_models)
        if existing_models:
            init_seed = int(existing_models[0].metadata.get("init_seed", init_seed))
        seed_model = self._build_seed_model(tf, seed_name, init_seed)
        mutators = [item for item in baseline_mutate_ops() if item in self._stable_mutators]
        if not mutators:
            mutators = list(baseline_mutate_ops())
        chooser = _CometMCMC(mutators)
        architecture_pool: list[tuple[str, object]] = [("seed_000", seed_model)]
        last_used: str | None = None

        for existing in existing_models:
            if existing.metadata.get("generation_error"):
                continue
            parent_name = str(existing.parent_case_id or existing.metadata.get("parent_case_id") or "seed_000")
            parent_lookup = {name: model for name, model in architecture_pool}
            parent_model = parent_lookup.get(parent_name)
            if parent_model is None:
                raise RuntimeError(f"Missing replay parent {parent_name!r} for {existing.case_id}")
            replay_seed = self._replay_seed(existing)
            self._seed_everything(tf, replay_seed)
            if str(existing.metadata.get("mutator", "")) == "WS_fallback":
                model = self._fallback_weight_shift(tf, parent_model, np.random.default_rng(replay_seed))
            else:
                parent_clone = self._clone_model(tf, parent_model, comet_custom_objects)
                cwd = os.getcwd()
                os.chdir(project_dir)
                try:
                    model = generate_model_by_model_mutation(
                        model=parent_clone,
                        operator=str(existing.metadata.get("mutator", "")),
                        mutation_operator_mode="diverse",
                        architecture_measure=None,
                    )
                finally:
                    os.chdir(cwd)
            if model is None:
                raise RuntimeError(f"Replay produced no mutant for {existing.case_id}")
            architecture_pool.append((existing.case_id, model))
            mutator = str(existing.metadata.get("mutator", ""))
            if mutator:
                chooser.reward(mutator, float(existing.metadata.get("reward", 0.0) or 0.0))
                last_used = mutator

        max_attempts = int(request.hyper_params.get("max_generation_attempts_per_case", self._max_generation_attempts_per_case))
        for idx in range(len(existing_models), request.num_cases):
            case_id = f"comet_{idx:03d}"
            last_error: str | None = None
            for attempt in range(1, max_attempts + 1):
                replay_seed = stable_seed(self.name, seed_name, request.random_seed, case_id, attempt, "replay")
                attempt_rng = np.random.default_rng(replay_seed)
                parent_name, parent_model = architecture_pool[int(attempt_rng.integers(0, len(architecture_pool)))]
                self._seed_everything(tf, replay_seed)
                mutator = chooser.choose(last_used, rng=attempt_rng)
                try:
                    parent_clone = self._clone_model(tf, parent_model, comet_custom_objects)

                    cwd = os.getcwd()
                    os.chdir(project_dir)
                    try:
                        mutant = generate_model_by_model_mutation(
                            model=parent_clone,
                            operator=mutator,
                            mutation_operator_mode="diverse",
                            architecture_measure=None,
                        )
                    finally:
                        os.chdir(cwd)

                    if mutant is None:
                        raise RuntimeError("mutant_is_none")

                    sample_rng = np.random.default_rng(stable_seed(self.name, case_id, replay_seed, "sample"))
                    sample = sample_rng.standard_normal((1, 64, 64, 3), dtype=np.float32)
                    parent_pred = parent_model(sample, training=False).numpy()
                    mutant_pred = mutant(sample, training=False).numpy()
                    if np.isnan(mutant_pred).any() or np.isinf(mutant_pred).any():
                        raise RuntimeError("mutant_forward_nan_or_inf")
                    if MetricsUtils is not None:
                        reward = float(np.sum(MetricsUtils.Relative_metrics(parent_pred, mutant_pred)))
                    else:
                        reward = float(np.max(np.abs(parent_pred - mutant_pred)))
                except Exception as exc:
                    last_error = repr(exc)
                    chooser.reward(mutator, 0.0)
                    continue

                chooser.reward(mutator, reward)
                architecture_pool.append((case_id, mutant))
                last_used = mutator

                metadata = {
                    "seed_name": seed_name,
                    "parent_case_id": parent_name,
                    "mutator": mutator,
                    "reward": reward,
                    "generation_attempts": attempt,
                    "input_shape": (64, 64, 3),
                    "batch_size": 1,
                    "mutation_depth": idx + 1,
                    "init_seed": init_seed,
                    "replay_seed": replay_seed,
                    "generation_random_seed": request.random_seed,
                }
                yield self._model_to_graph(case_id, mutant, seed_name, parent_name, metadata)
                break
            else:
                parent_name, parent_model = architecture_pool[0]
                mutator = "WS_fallback"
                replay_seed = stable_seed(self.name, seed_name, request.random_seed, case_id, "fallback")
                self._seed_everything(tf, replay_seed)
                mutant = self._fallback_weight_shift(tf, parent_model, np.random.default_rng(replay_seed))
                sample_rng = np.random.default_rng(stable_seed(self.name, case_id, replay_seed, "sample"))
                sample = sample_rng.standard_normal((1, 64, 64, 3), dtype=np.float32)
                parent_pred = parent_model(sample, training=False).numpy()
                mutant_pred = mutant(sample, training=False).numpy()
                if MetricsUtils is not None:
                    reward = float(np.sum(MetricsUtils.Relative_metrics(parent_pred, mutant_pred)))
                else:
                    reward = float(np.max(np.abs(parent_pred - mutant_pred)))

                architecture_pool.append((case_id, mutant))
                last_used = None

                metadata = {
                    "seed_name": seed_name,
                    "parent_case_id": parent_name,
                    "mutator": mutator,
                    "reward": reward,
                    "generation_attempts": max_attempts,
                    "fallback_from_error": last_error,
                    "input_shape": (64, 64, 3),
                    "batch_size": 1,
                    "mutation_depth": idx + 1,
                    "init_seed": init_seed,
                    "replay_seed": replay_seed,
                    "generation_random_seed": request.random_seed,
                }
                yield self._model_to_graph(case_id, mutant, seed_name, parent_name, metadata)

    def detect_bug(
        self,
        model: GraphModel,
        models: list[GraphModel] | None = None,
        diff_threshold: float = 1e-5,
    ) -> BugReport:
        started = time.perf_counter()
        generation_error = model.metadata.get("generation_error")
        if generation_error:
            runtime = time.perf_counter() - started
            return BugReport(
                case_id=model.case_id,
                project=self.name,
                has_bug=True,
                bug_type="generation_runtime_error",
                runtime_s=runtime,
                raw_result={"error": generation_error},
            )

        project_dir = self._project_dir()
        cwd = os.getcwd()
        os.chdir(project_dir)
        try:
            tf, _, _, MetricsUtils = self._imports()
            from scripts.prediction.custom_objects import custom_objects as comet_custom_objects
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
        seed = self._build_seed_model(tf, seed_name, init_seed)

        try:
            del comet_custom_objects
            mutant = self._replay_cache.get(model.case_id, seed)
            sample = np.random.default_rng(stable_seed(self.name, model.case_id, init_seed, "sample")).standard_normal(
                (1, 64, 64, 3),
                dtype=np.float32,
            )
            seed_pred = seed(sample, training=False).numpy()
            mutant_pred = mutant(sample, training=False).numpy()
            if np.isnan(mutant_pred).any() or np.isinf(mutant_pred).any():
                runtime = time.perf_counter() - started
                return BugReport(model.case_id, self.name, True, "nan_or_inf", runtime, {"delta": None})
            if MetricsUtils is not None:
                delta = float(np.sum(MetricsUtils.Relative_metrics(seed_pred, mutant_pred)))
            else:
                delta = float(np.max(np.abs(seed_pred - mutant_pred)))
            runtime = time.perf_counter() - started
            return BugReport(
                case_id=model.case_id,
                project=self.name,
                has_bug=delta > diff_threshold,
                bug_type="differential_inconsistency" if delta > diff_threshold else None,
                runtime_s=runtime,
                raw_result={"delta": delta},
            )
        except Exception as exc:
            runtime = time.perf_counter() - started
            return BugReport(
                case_id=model.case_id,
                project=self.name,
                has_bug=True,
                bug_type="runtime_error",
                runtime_s=runtime,
                raw_result={"error": repr(exc)},
            )
