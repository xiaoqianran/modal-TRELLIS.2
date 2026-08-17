from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from modal_trellis2.modal.weights import MODELS_512


def _nonempty(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def inspect_trellis_bundle(root: Path) -> dict[str, Any]:
    """Validate every local file needed by the production 512 pipeline."""
    root = Path(root)
    pipeline_path = root / "pipeline.json"
    missing: list[str] = []
    invalid: list[str] = []
    model_refs: dict[str, str] = {}

    if not _nonempty(pipeline_path):
        missing.append(str(pipeline_path))
        return {"ok": False, "missing": missing, "invalid": invalid, "models": model_refs}

    try:
        payload = _read_json(pipeline_path)
        models = payload["args"]["models"]
    except (OSError, ValueError, TypeError, KeyError) as exc:
        invalid.append(f"{pipeline_path}: {exc}")
        return {"ok": False, "missing": missing, "invalid": invalid, "models": model_refs}

    if not isinstance(models, dict):
        invalid.append(f"{pipeline_path}: args.models is not an object")
        return {"ok": False, "missing": missing, "invalid": invalid, "models": model_refs}

    for name in MODELS_512:
        ref = models.get(name)
        if not isinstance(ref, str) or not ref.strip():
            invalid.append(f"{pipeline_path}: missing model ref {name!r}")
            continue
        model_refs[name] = ref
        base = root / ref
        for suffix in (".json", ".safetensors"):
            path = Path(f"{base}{suffix}")
            if not _nonempty(path):
                missing.append(str(path))

    return {
        "ok": not missing and not invalid,
        "missing": missing,
        "invalid": invalid,
        "models": model_refs,
    }


def require_trellis_bundle(root: Path) -> dict[str, Any]:
    report = inspect_trellis_bundle(root)
    if not report["ok"]:
        details = "; ".join([*report["missing"], *report["invalid"]])
        raise RuntimeError(
            "Incomplete TRELLIS.2 offline bundle. Run `uv run modal-trellis2 prefetch` again. "
            f"{details}"
        )
    return report


def _weight_files_from_index(root: Path, index_path: Path) -> tuple[list[Path], list[str]]:
    invalid: list[str] = []
    try:
        payload = _read_json(index_path)
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("weight_map is missing or empty")
        filenames = sorted({value for value in weight_map.values() if isinstance(value, str)})
        if not filenames:
            raise ValueError("weight_map contains no files")
        return [root / name for name in filenames], invalid
    except (OSError, ValueError, TypeError) as exc:
        invalid.append(f"{index_path}: {exc}")
        return [], invalid


def inspect_hf_model_dir(root: Path) -> dict[str, Any]:
    """Validate a local Transformers model directory without importing Transformers."""
    root = Path(root)
    missing: list[str] = []
    invalid: list[str] = []
    config = root / "config.json"
    if not _nonempty(config):
        missing.append(str(config))

    weight_paths: list[Path] = []
    index_candidates = (
        root / "model.safetensors.index.json",
        root / "pytorch_model.bin.index.json",
    )
    index_path = next((path for path in index_candidates if _nonempty(path)), None)
    if index_path is not None:
        weight_paths, index_invalid = _weight_files_from_index(root, index_path)
        invalid.extend(index_invalid)
    else:
        direct_candidates = [
            root / "model.safetensors",
            root / "pytorch_model.bin",
        ]
        weight_paths = [path for path in direct_candidates if _nonempty(path)]
        if not weight_paths:
            weight_paths = [path for path in root.glob("*.safetensors") if _nonempty(path)]
        if not weight_paths:
            weight_paths = [path for path in root.glob("pytorch_model*.bin") if _nonempty(path)]
        if not weight_paths:
            missing.append(f"{root}: model weights")

    for path in weight_paths:
        if not _nonempty(path):
            missing.append(str(path))

    return {
        "ok": not missing and not invalid,
        "missing": missing,
        "invalid": invalid,
        "weight_files": [str(path) for path in weight_paths],
    }


def hf_model_ready(root: Path) -> bool:
    return bool(inspect_hf_model_dir(root)["ok"])


def require_hf_model_dir(root: Path, *, label: str) -> dict[str, Any]:
    report = inspect_hf_model_dir(root)
    if not report["ok"]:
        details = "; ".join([*report["missing"], *report["invalid"]])
        raise RuntimeError(
            f"Incomplete offline {label} bundle. Run `uv run modal-trellis2 prefetch` again. "
            f"{details}"
        )
    return report
