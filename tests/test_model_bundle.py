from __future__ import annotations

import json
from pathlib import Path

from modal_trellis2.modal.model_bundle import inspect_hf_model_dir, inspect_trellis_bundle
from modal_trellis2.modal.weights import MODELS_512


def test_trellis_bundle_requires_every_512_config_and_weight(tmp_path: Path) -> None:
    refs = {name: f"ckpts/{name}" for name in MODELS_512}
    (tmp_path / "ckpts").mkdir()
    (tmp_path / "pipeline.json").write_text(
        json.dumps({"args": {"models": refs}}), encoding="utf-8"
    )
    for ref in refs.values():
        Path(f"{tmp_path / ref}.json").write_text("{}", encoding="utf-8")
        Path(f"{tmp_path / ref}.safetensors").write_bytes(b"weights")

    report = inspect_trellis_bundle(tmp_path)
    assert report["ok"] is True

    Path(f"{tmp_path / refs[MODELS_512[-1]]}.safetensors").unlink()
    report = inspect_trellis_bundle(tmp_path)
    assert report["ok"] is False
    assert report["missing"]


def test_hf_bundle_rejects_config_only_directory(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    assert inspect_hf_model_dir(tmp_path)["ok"] is False
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    assert inspect_hf_model_dir(tmp_path)["ok"] is True


def test_hf_sharded_bundle_requires_every_indexed_shard(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "a": "model-00001-of-00002.safetensors",
                    "b": "model-00002-of-00002.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"one")
    assert inspect_hf_model_dir(tmp_path)["ok"] is False
    (tmp_path / "model-00002-of-00002.safetensors").write_bytes(b"two")
    assert inspect_hf_model_dir(tmp_path)["ok"] is True
