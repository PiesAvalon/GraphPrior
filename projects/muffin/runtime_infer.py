from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from utils.utils import switch_backend


def _load_inputs(path: Path) -> list[np.ndarray]:
    data = np.load(path, allow_pickle=False)
    try:
        keys = sorted(data.files, key=lambda key: int(key.split("_")[-1]))
    except ValueError:
        keys = sorted(data.files)
    return [np.asarray(data[key]) for key in keys]


def _load_weights(weights_dir: Path, layer_name: str) -> list[np.ndarray] | None:
    weight_path = weights_dir / f"{layer_name}.npz"
    if not weight_path.exists():
        return None
    data = np.load(weight_path, allow_pickle=False)
    try:
        keys = sorted(data.files, key=lambda key: int(key.split("_")[-1]))
    except ValueError:
        keys = sorted(data.files)
    return [np.asarray(data[key]) for key in keys]


def _to_numpy(output) -> np.ndarray:
    if hasattr(output, "detach"):
        output = output.detach()
    if hasattr(output, "cpu"):
        output = output.cpu()
    if hasattr(output, "numpy"):
        return np.asarray(output.numpy())
    return np.asarray(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--weights-dir", required=True)
    parser.add_argument("--inputs-path", required=True)
    parser.add_argument("--outputs-path", required=True)
    parser.add_argument("--meta-path", required=True)
    args = parser.parse_args()

    switch_backend(args.backend)
    import keras

    model = keras.models.load_model(args.model_path, compile=False)
    weights_dir = Path(args.weights_dir)
    for layer in model.layers:
        weights = _load_weights(weights_dir, layer.name)
        if weights is None:
            continue
        existing = layer.get_weights()
        if len(existing) != len(weights):
            continue
        if any(tuple(src.shape) != tuple(dst.shape) for src, dst in zip(weights, existing)):
            continue
        layer.set_weights(weights)

    inputs = _load_inputs(Path(args.inputs_path))
    outputs = model(inputs[0] if len(inputs) == 1 else inputs, training=False)
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]

    arrays = [_to_numpy(output) for output in outputs]
    np.savez(Path(args.outputs_path), *arrays)
    meta = {
        "backend": args.backend,
        "nan": any(np.isnan(array).any() for array in arrays),
        "inf": any(np.isinf(array).any() for array in arrays),
        "num_outputs": len(arrays),
        "shapes": [list(array.shape) for array in arrays],
    }
    Path(args.meta_path).write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
