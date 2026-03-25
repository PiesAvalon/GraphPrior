from __future__ import annotations

import ast
import importlib.util
import json
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from GraphPrior.types import BugReport, GenerationRequest, GraphModel, GraphNode

from .._shared import prepend_path
from ..base import ProjectAdapter


class MuffinAdapter(ProjectAdapter):
    name = "muffin"
    uses_seed_model = False
    _supported_backends = ("tensorflow", "torch")
    _max_generation_attempts_per_case = 50

    def _project_dir(self) -> Path:
        return Path(__file__).resolve().parent

    def _keras_layer_symbol(self, layer_type: str) -> str:
        pieces = layer_type.split("_")
        symbol = pieces[0][0].upper() + pieces[0][1:]
        for piece in pieces[1:]:
            symbol += piece[0].upper() + piece[1:]
        return symbol

    def _prune_unsupported_layers(self, muffin_utils) -> None:
        import keras

        supported = {
            layer_type
            for layer_type in muffin_utils.layer_types
            if hasattr(keras.layers, self._keras_layer_symbol(layer_type))
        }
        if len(supported) == len(muffin_utils.layer_types):
            return

        muffin_utils.layer_types = [layer for layer in muffin_utils.layer_types if layer in supported]
        muffin_utils.seq_layer_types = [layer for layer in muffin_utils.seq_layer_types if layer in supported]
        muffin_utils.RNN_layer_types = [layer for layer in muffin_utils.RNN_layer_types if layer in supported]
        muffin_utils.activation_layer_types = [layer for layer in muffin_utils.activation_layer_types if layer in supported]
        muffin_utils.merging_layer_types = [layer for layer in muffin_utils.merging_layer_types if layer in supported]
        muffin_utils.conv_layer_types = [layer for layer in muffin_utils.conv_layer_types if layer in supported]
        muffin_utils.pooling_layer_types = [layer for layer in muffin_utils.pooling_layer_types if layer in supported]
        muffin_utils.recurrent_layer_types = [layer for layer in muffin_utils.recurrent_layer_types if layer in supported]
        muffin_utils.normalization_layers_types = [layer for layer in muffin_utils.normalization_layers_types if layer in supported]
        muffin_utils.reshape_layer_types = [layer for layer in muffin_utils.reshape_layer_types if layer in supported]
        muffin_utils.locally_connected_layer_types = [layer for layer in muffin_utils.locally_connected_layer_types if layer in supported]
        muffin_utils.normal_layer_types = [layer for layer in muffin_utils.normal_layer_types if layer in supported]
        muffin_utils.reduction_layer_types = [layer for layer in muffin_utils.reduction_layer_types if layer in supported]
        muffin_utils.layer_conditions = {
            layer: cond for layer, cond in muffin_utils.layer_conditions.items() if layer in supported
        }

    def _imports(self):
        prepend_path(self._project_dir())
        from utils.db_manager import DbManager
        from utils.selection import Roulette
        import utils.utils as muffin_utils

        self._prune_unsupported_layers(muffin_utils)
        from cases_generation.model_info_generator import ModelInfoGenerator

        return DbManager, Roulette, muffin_utils.layer_conditions, muffin_utils.layer_types, ModelInfoGenerator

    def _config(self, output_root: Path, generate_mode: str) -> dict:
        db_path = output_root / "muffin.db"
        if (not db_path.exists()) or db_path.stat().st_size == 0:
            schema = (self._project_dir() / "data" / "create_db.sql").read_text(encoding="utf-8")
            conn = sqlite3.connect(db_path)
            try:
                conn.executescript(schema)
                conn.commit()
            finally:
                conn.close()
        return {
            "model": {
                "var": {
                    "tensor_dimension_range": (2, 5),
                    "tensor_element_size_range": (2, 5),
                    "weight_value_range": (-10.0, 10.0),
                    "small_value_range": (0, 1),
                    "vocabulary_size": 1001,
                },
                "node_num_range": (5, 5),
                "dag_io_num_range": (1, 3),
                "dag_max_branch_num": 2,
                "cell_num": 3,
                "node_num_per_normal_cell": 10,
                "node_num_per_reduction_cell": 2,
            },
            "db_path": str(db_path),
            "output_dir": str(output_root / "native_cases"),
            "generate_mode": generate_mode,
        }

    def _canonical_op(self, layer_type: str) -> str:
        mapping = {
            "conv2D": "conv2d",
            "dense": "linear",
            "batch_normalization": "batchnorm",
            "max_pooling2D": "maxpool2d",
            "global_average_pooling2D": "global_avg_pool",
            "flatten": "flatten",
            "add": "add",
            "concatenate": "concat",
        }
        return mapping.get(layer_type, layer_type)

    def _maybe_int(self, value):
        return None if value is None else int(value)

    def _normalize_spatial_attr(self, value):
        if value is None:
            return None
        if not isinstance(value, (list, tuple)):
            return (int(value), int(value))
        normalized = []
        for item in value:
            normalized.append(None if item is None else int(item))
        return tuple(normalized)

    def _json_to_graph(self, case_id: str, json_path: Path, metadata: dict) -> GraphModel:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        structure = payload.get("model_structure", {})
        nodes: list[GraphNode] = []
        for key, item in structure.items():
            nid = int(key)
            pre = tuple(int(x) for x in item.get("pre_layers", []))
            layer_type = str(item.get("type", "unknown"))
            args = dict(item.get("args", {}))
            output_shape = tuple(item.get("output_shape", []))
            attrs = {
                "name": args.get("name", f"node_{nid}"),
                "output_shape": output_shape,
                "dtype": args.get("dtype") or "float32",
            }
            input_shapes = []
            for parent_id in pre:
                parent = structure.get(str(parent_id))
                if parent is not None:
                    input_shapes.append(tuple(parent.get("output_shape", [])))
            if input_shapes:
                attrs["input_shape"] = input_shapes[0] if len(input_shapes) == 1 else tuple(input_shapes)
            out_channels = self._maybe_int(args.get("filters"))
            if out_channels is not None:
                attrs["out_channels"] = out_channels
            out_dim = self._maybe_int(args.get("units"))
            if out_dim is not None:
                attrs["out_dim"] = out_dim
            kernel_size = self._normalize_spatial_attr(args.get("kernel_size"))
            if kernel_size is not None:
                attrs["kernel_size"] = kernel_size
            stride = self._normalize_spatial_attr(args.get("strides"))
            if stride is not None:
                attrs["stride"] = stride
            if "padding" in args:
                attrs["padding"] = args["padding"]
            if "axis" in args:
                attrs["axis"] = args["axis"]
            if layer_type == "batch_normalization" and output_shape:
                axis = args.get("axis", -1)
                axis = axis if axis >= 0 else len(output_shape) + axis
                if 0 <= axis < len(output_shape) and output_shape[axis] is not None:
                    attrs["num_features"] = int(output_shape[axis])
            nodes.append(
                GraphNode(
                    node_id=nid,
                    op_type=self._canonical_op(layer_type),
                    inputs=pre,
                    attrs=attrs,
                )
            )
        nodes.sort(key=lambda n: n.node_id)
        return GraphModel(
            case_id=case_id,
            project=self.name,
            nodes=nodes,
            seed_name=None,
            parent_case_id=None,
            metadata=metadata,
            native_artifact_path=str(json_path),
        )

    def generate_models(
        self,
        request: GenerationRequest,
        output_root: Path,
        existing_models: list[GraphModel] | None = None,
    ):
        DbManager, Roulette, layer_conditions, layer_types, ModelInfoGenerator = self._imports()
        generate_mode = str(request.hyper_params.get("generate_mode", "template"))
        config = self._config(output_root, generate_mode)
        db_manager = DbManager(config["db_path"])
        selector = Roulette(layer_types=layer_types, layer_conditions=layer_conditions, use_heuristic=True)
        generator = ModelInfoGenerator(config["model"], db_manager, selector, generate_mode)

        start_idx = len(existing_models or [])
        max_attempts = int(request.hyper_params.get("max_generation_attempts_per_case", self._max_generation_attempts_per_case))
        for idx in range(start_idx, request.num_cases):
            case_id = f"muffin_{idx:03d}"
            last_error: str | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    json_path, input_shapes, output_shapes, model_id, exp_dir = generator.generate(save_dir=config["output_dir"])
                    first_input_shape = next(iter(input_shapes.values()), None)
                    metadata = {
                        "model_id": model_id,
                        "input_shapes": input_shapes,
                        "output_shapes": output_shapes,
                        "input_shape": first_input_shape[1:] if first_input_shape and len(first_input_shape) > 1 else first_input_shape,
                        "batch_size": first_input_shape[0] if first_input_shape and len(first_input_shape) > 0 and first_input_shape[0] is not None else 1,
                        "generate_mode": generate_mode,
                        "generation_attempts": attempt,
                        "db_path": config["db_path"],
                        "native": {"json_path": str(json_path), "exp_dir": str(exp_dir)},
                    }
                    yield self._json_to_graph(case_id, Path(json_path), metadata)
                    break
                except Exception as exc:
                    last_error = repr(exc)
                    continue
            else:
                raise RuntimeError(f"Muffin failed to generate {case_id} after {max_attempts} attempts: {last_error}")

    def _available_backends(self) -> list[str]:
        module_names = {
            "tensorflow": "tensorflow",
            "torch": "torch",
        }
        available: list[str] = []
        for backend in self._supported_backends:
            if importlib.util.find_spec(module_names[backend]) is not None:
                available.append(backend)
        return available

    def _materialize_backend(self, json_path: Path, backend: str, output_dir: Path) -> subprocess.CompletedProcess[str]:
        output_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "-m",
            "cases_generation.generate_one",
            "--backend",
            backend,
            "--json_path",
            str(json_path),
            "--weight_minv",
            "-10.0",
            "--weight_maxv",
            "10.0",
            "--output_dir",
            str(output_dir),
        ]
        return subprocess.run(cmd, cwd=self._project_dir(), capture_output=True, text=True)

    def _ordered_inputs(self, input_shapes: dict) -> list[tuple[str, tuple[int, ...]]]:
        items = []
        for name, shape in input_shapes.items():
            if isinstance(shape, str):
                shape = ast.literal_eval(shape)
            normalized = tuple(1 if dim is None else int(dim) for dim in shape)
            items.append((str(name), normalized))
        items.sort(key=lambda item: item[0])
        return items

    def _write_inputs(self, model: GraphModel, path: Path) -> list[tuple[str, tuple[int, ...]]]:
        ordered = self._ordered_inputs(model.metadata.get("input_shapes", {}))
        rng = np.random.default_rng(2026)
        arrays = [rng.standard_normal(shape, dtype=np.float32) for _, shape in ordered]
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, *arrays)
        return ordered

    def _run_inference(
        self,
        backend: str,
        model_path: Path,
        weights_dir: Path,
        inputs_path: Path,
        outputs_path: Path,
        meta_path: Path,
    ) -> subprocess.CompletedProcess[str]:
        cmd = [
            sys.executable,
            str(self._project_dir() / "runtime_infer.py"),
            "--backend",
            backend,
            "--model-path",
            str(model_path),
            "--weights-dir",
            str(weights_dir),
            "--inputs-path",
            str(inputs_path),
            "--outputs-path",
            str(outputs_path),
            "--meta-path",
            str(meta_path),
        ]
        return subprocess.run(cmd, cwd=self._project_dir(), capture_output=True, text=True)

    def _load_output_arrays(self, path: Path) -> list[np.ndarray]:
        data = np.load(path, allow_pickle=False)
        try:
            keys = sorted(data.files, key=lambda key: int(key.split("_")[-1]))
        except ValueError:
            keys = sorted(data.files)
        return [np.asarray(data[key]) for key in keys]

    def _pairwise_delta(self, lhs: list[np.ndarray], rhs: list[np.ndarray]) -> float | None:
        if len(lhs) != len(rhs):
            return None
        deltas: list[float] = []
        for left, right in zip(lhs, rhs):
            if left.shape != right.shape:
                return None
            deltas.append(float(np.max(np.abs(left - right))))
        return max(deltas) if deltas else 0.0

    def _update_db_records(
        self,
        db_path: Path | None,
        model_id: int | None,
        fail_backends: list[str],
        crash_backends: list[str],
        nan_backends: list[str],
        inf_backends: list[str],
        deltas: dict[str, float | None],
    ) -> None:
        if db_path is None or model_id is None or not db_path.exists():
            return
        DbManager, _, _, _, _ = self._imports()
        db = DbManager(str(db_path))
        if fail_backends:
            db.update_model_generate_fail_backends(model_id, fail_backends)
        if crash_backends:
            db.update_model_crash_backends(model_id, crash_backends)
        if nan_backends:
            db.update_model_nan_backends(model_id, nan_backends)
        if inf_backends:
            db.update_model_inf_backends(model_id, inf_backends)
        if deltas:
            conn = sqlite3.connect(db_path, check_same_thread=False)
            try:
                conn.executemany(
                    """
                    insert or replace into inconsistency(model_id, backend_pair, model_output_delta, loss_delta, loss_grads_delta, weights_delta)
                    values(?, ?, ?, null, null, null)
                    """,
                    [(int(model_id), backend_pair, delta) for backend_pair, delta in deltas.items()],
                )
                conn.commit()
            finally:
                conn.close()

    def detect_bug(
        self,
        model: GraphModel,
        models: list[GraphModel] | None = None,
        diff_threshold: float = 1e-5,
    ) -> BugReport:
        del models
        started = time.perf_counter()
        if model.metadata.get("generation_error"):
            runtime = time.perf_counter() - started
            return BugReport(model.case_id, self.name, True, "generation_runtime_error", runtime, {"error": model.metadata["generation_error"]})

        available_backends = self._available_backends()
        exp_dir = Path(model.metadata.get("native", {}).get("exp_dir", "."))
        artifact_root = exp_dir / "backend_eval"
        artifact_root.mkdir(parents=True, exist_ok=True)
        inputs_path = artifact_root / "inputs.npz"
        ordered_inputs = self._write_inputs(model, inputs_path)

        materialized: dict[str, Path] = {}
        fail_backends: list[str] = []
        crash_backends: list[str] = []
        nan_backends: list[str] = []
        inf_backends: list[str] = []
        deltas: dict[str, float | None] = {}
        stderr_by_backend: dict[str, str] = {}
        stdout_by_backend: dict[str, str] = {}
        output_arrays: dict[str, list[np.ndarray]] = {}

        for backend in available_backends:
            backend_dir = artifact_root / backend
            result = self._materialize_backend(Path(model.native_artifact_path), backend, backend_dir)
            stdout_by_backend[backend] = result.stdout[-2000:]
            stderr_by_backend[backend] = result.stderr[-2000:]
            if result.returncode != 0:
                fail_backends.append(backend)
                continue

            model_path = backend_dir / f"{backend}.keras"
            weights_dir = backend_dir / "initial_weights"
            outputs_path = backend_dir / "outputs.npz"
            meta_path = backend_dir / "outputs_meta.json"
            infer = self._run_inference(backend, model_path, weights_dir, inputs_path, outputs_path, meta_path)
            stdout_by_backend[f"{backend}:infer"] = infer.stdout[-2000:]
            stderr_by_backend[f"{backend}:infer"] = infer.stderr[-2000:]
            if infer.returncode != 0:
                crash_backends.append(backend)
                continue

            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("nan"):
                nan_backends.append(backend)
            if meta.get("inf"):
                inf_backends.append(backend)
            materialized[backend] = backend_dir
            output_arrays[backend] = self._load_output_arrays(outputs_path)

        successful_backends = sorted(output_arrays.keys())
        for idx, left in enumerate(successful_backends):
            for right in successful_backends[idx + 1:]:
                delta = self._pairwise_delta(output_arrays[left], output_arrays[right])
                deltas[f"{left}|{right}"] = delta

        model_id = model.metadata.get("model_id")
        try:
            model_id = int(model_id) if model_id is not None else None
        except Exception:
            model_id = None
        db_path = model.metadata.get("db_path")
        self._update_db_records(
            Path(db_path) if db_path else None,
            model_id,
            fail_backends,
            crash_backends,
            nan_backends,
            inf_backends,
            deltas,
        )

        runtime = time.perf_counter() - started
        if fail_backends:
            return BugReport(
                case_id=model.case_id,
                project=self.name,
                has_bug=True,
                bug_type="generation_runtime_error",
                runtime_s=runtime,
                raw_result={
                    "available_backends": available_backends,
                    "failed_backends": fail_backends,
                    "stdout": stdout_by_backend,
                    "stderr": stderr_by_backend,
                    "input_shapes": ordered_inputs,
                },
            )
        if crash_backends:
            return BugReport(
                case_id=model.case_id,
                project=self.name,
                has_bug=True,
                bug_type="crash_in_backend",
                runtime_s=runtime,
                raw_result={
                    "available_backends": available_backends,
                    "crash_backends": crash_backends,
                    "stdout": stdout_by_backend,
                    "stderr": stderr_by_backend,
                    "input_shapes": ordered_inputs,
                },
            )
        if nan_backends:
            return BugReport(
                case_id=model.case_id,
                project=self.name,
                has_bug=True,
                bug_type="nan_or_inf",
                runtime_s=runtime,
                raw_result={
                    "available_backends": available_backends,
                    "nan_backends": nan_backends,
                    "inf_backends": inf_backends,
                    "backend_deltas": deltas,
                    "input_shapes": ordered_inputs,
                },
            )
        if inf_backends:
            return BugReport(
                case_id=model.case_id,
                project=self.name,
                has_bug=True,
                bug_type="nan_or_inf",
                runtime_s=runtime,
                raw_result={
                    "available_backends": available_backends,
                    "nan_backends": nan_backends,
                    "inf_backends": inf_backends,
                    "backend_deltas": deltas,
                    "input_shapes": ordered_inputs,
                },
            )

        inconsistent_pairs = {pair: delta for pair, delta in deltas.items() if delta is None or delta > diff_threshold}
        return BugReport(
            case_id=model.case_id,
            project=self.name,
            has_bug=bool(inconsistent_pairs),
            bug_type="differential_inconsistency" if inconsistent_pairs else None,
            runtime_s=runtime,
            raw_result={
                "available_backends": available_backends,
                "successful_backends": successful_backends,
                "backend_deltas": deltas,
                "inconsistent_pairs": inconsistent_pairs,
                "input_shapes": ordered_inputs,
                "differential_skipped": len(successful_backends) < 2,
            },
        )
